"""Step 5 — Genetic Algorithm for 5x5 EvoGym body evolution (mutation-only).

Evolves soft robot morphologies starting from 100% random valid connected blobs
(never seeding the hand-designed reference robot) using the repository's `ppo.py`
unchanged as the inner-loop controller.

Fitness favors progression, execution speed once the goal is reached, and
organic energy efficiency (normalized actuation effort + temporal smoothness),
while logging biomechanical metrics such as Cost of Transport (CoT) and full
genealogy (`parent_ids`, `operator`, per-generation artifacts) for Steps 6–8.

Example:
    caffeinate -i uv run --python .venv/bin/python evolve.py \
        --env Walker-v0 --pop-size 8 --generations 10 --steps 500000 \
        --elites 2 --survivors 3 --mutation-rate 0.12 --mutation-rate-end 0.06 \
        --reval-top-k 3 --reval-seeds 31 32 33 --reval-ref walker \
        --n-envs 8 --seed 31 --out runs/evo_walker_500k_s31
"""

import argparse
import inspect
import json
import time
from pathlib import Path

import evogym
import numpy as np
import pandas as pd
import torch

from ppo import PPOAgent, load_body, make_env, to_env_action, train_agent

# EvoGym default sampling distribution over voxels:
# 0 empty, 1 rigid, 2 soft, 3 horizontal actuator, 4 vertical actuator.
DEFAULT_VOXEL_PROBS = np.array([0.6, 0.2, 0.2, 0.2, 0.2], dtype=np.float64)
DEFAULT_VOXEL_PROBS /= DEFAULT_VOXEL_PROBS.sum()

ACTUATOR_REST_LENGTH = 1.0
ACTUATOR_MAX_STRAIN = 0.6  # bounds are [0.6, 1.6] -> max |a - 1.0| = 0.6


# ---------------------------------------------------------------------------
# Body validity, canonical hashing, random init, and mutation-only operator
# ---------------------------------------------------------------------------


def is_valid_body(
    body: np.ndarray, min_voxels: int = 2, min_actuators: int = 1
) -> bool:
    """Return True if a 2D voxel grid is 4-connected with enough mass and actuators."""
    arr = np.asarray(body, dtype=int)
    n_voxels = int((arr > 0).sum())
    n_actuators = int((arr >= 3).sum())
    if n_voxels < min_voxels or n_actuators < min_actuators:
        return False
    return bool(evogym.is_connected(arr) and evogym.has_actuator(arr))


def canonical_body_hash(body: np.ndarray) -> str:
    """Translation-invariant canonical hash of a voxel body.

    Cropping to the tight non-zero bounding box guarantees that shifting the
    same robot inside the 5x5 grid maps to the same hash, avoiding redundant
    PPO trainings.
    """
    arr = np.asarray(body, dtype=int)
    nz = np.argwhere(arr > 0)
    if nz.size == 0:
        return "empty"
    r0, c0 = nz.min(axis=0)
    r1, c1 = nz.max(axis=0)
    cropped = arr[r0 : r1 + 1, c0 : c1 + 1]
    h, w = cropped.shape
    digits = "".join(str(int(v)) for v in cropped.ravel())
    return f"{h}x{w}:{digits}"


def body_stats(body: np.ndarray) -> dict:
    """Return structural counts for a voxel body."""
    arr = np.asarray(body, dtype=int)
    return {
        "n_voxels": int((arr > 0).sum()),
        "n_rigid": int((arr == 1).sum()),
        "n_soft": int((arr == 2).sum()),
        "n_h_act": int((arr == 3).sum()),
        "n_v_act": int((arr == 4).sum()),
        "n_actuators": int((arr >= 3).sum()),
    }


def sample_random_body(
    rng: np.random.Generator,
    shape: tuple[int, int] = (5, 5),
    seen_hashes: set[str] | None = None,
    voxel_probs: np.ndarray = DEFAULT_VOXEL_PROBS,
    min_voxels: int = 2,
    min_actuators: int = 1,
    max_attempts: int = 10_000,
) -> np.ndarray:
    """Sample a random valid 4-connected voxel body not present in seen_hashes."""
    seen = seen_hashes or set()
    for _ in range(max_attempts):
        body = rng.choice(5, size=shape, p=voxel_probs).astype(int)
        if not is_valid_body(body, min_voxels=min_voxels, min_actuators=min_actuators):
            continue
        if canonical_body_hash(body) in seen:
            continue
        return body
    raise RuntimeError("Failed to sample a novel valid random body.")


def mutate_body(
    parent: np.ndarray,
    rng: np.random.Generator,
    mutation_rate: float = 0.12,
    seen_hashes: set[str] | None = None,
    voxel_probs: np.ndarray = DEFAULT_VOXEL_PROBS,
    min_voxels: int = 2,
    min_actuators: int = 1,
    max_attempts: int = 5_000,
) -> tuple[np.ndarray, int]:
    """Mutate a voxel body cell-by-cell until a novel valid child is produced.

    Returns (child_body, n_changed_voxels).
    """
    parent_arr = np.asarray(parent, dtype=int)
    parent_hash = canonical_body_hash(parent_arr)
    seen = seen_hashes or set()
    n_cells = parent_arr.size

    for attempt in range(max_attempts):
        # Gradually widen mutation if local neighborhood is exhausted.
        eff_rate = (
            mutation_rate
            if attempt < 1_000
            else max(mutation_rate, min(0.5, mutation_rate * 2.0))
        )
        child = parent_arr.copy()
        mask = rng.random(parent_arr.shape) < eff_rate
        if not mask.any():
            flat_idx = int(rng.integers(0, n_cells))
            mask.ravel()[flat_idx] = True

        for r, c in np.argwhere(mask):
            current = int(child[r, c])
            alt_probs = voxel_probs.copy()
            alt_probs[current] = 0.0
            alt_probs /= alt_probs.sum()
            child[r, c] = int(rng.choice(5, p=alt_probs))

        if not is_valid_body(child, min_voxels=min_voxels, min_actuators=min_actuators):
            continue
        h = canonical_body_hash(child)
        if h == parent_hash or h in seen:
            continue
        return child, int((child != parent_arr).sum())

    raise RuntimeError("Failed to produce a novel valid mutant from parent body.")


# ---------------------------------------------------------------------------
# Deterministic evaluation & organic fitness (speed + energy efficiency)
# ---------------------------------------------------------------------------


def evaluate_policy(
    agent: PPOAgent,
    body: np.ndarray,
    env_id: str = "Walker-v0",
    episodes: int = 1,
    seed: int = 0,
    w_speed: float = 3.0,
    w_energy: float = 0.5,
    w_jerk: float = 0.25,
    gif_path: Path | None = None,
) -> dict:
    """Evaluate a trained policy deterministically (action = mean mu) and compute
    progression, speed, actuation energy, smoothness (jerk), Cost of Transport,
    and composite organic fitness.
    """
    env = make_env(
        env_id, {"body": body}, render_mode="rgb_array" if gif_path else None
    )
    try:
        max_steps = int(env.spec.max_episode_steps or 500)
        grid_cells = int(np.asarray(body).size)
        stats = body_stats(body)

        ep_metrics = []
        frames = []

        agent.network.eval()
        for ep in range(episodes):
            state, _ = env.reset(seed=seed + ep)
            u = env.unwrapped
            x_init = float(u.object_pos_at_time(u.get_time(), "robot")[0].mean())
            last_valid_x = x_init

            done = False
            steps = 0
            total_reward = 0.0
            last_reward = 0.0
            terminated = False
            non_finite = False
            total_effort = 0.0
            total_jerk = 0.0
            prev_action = None

            while not done:
                s_t = torch.as_tensor(
                    state, dtype=torch.float32, device=agent.device
                ).unsqueeze(0)
                with torch.no_grad():
                    mean, _, _ = agent.network(s_t)
                env_action = to_env_action(
                    mean.squeeze(0).cpu().numpy(), env.action_space
                )

                # Actuation strain squared relative to physical rest length (1.0), normalized to [0, 1] per muscle.
                strain = (env_action - ACTUATOR_REST_LENGTH) / ACTUATOR_MAX_STRAIN
                total_effort += float(np.sum(strain * strain))
                if prev_action is not None:
                    delta_a = env_action - prev_action
                    total_jerk += float(np.sum(delta_a * delta_a))
                prev_action = env_action

                state, r, terminated, truncated, _ = env.step(env_action)
                done = bool(terminated or truncated)
                r_val = float(r)
                if np.isfinite(r_val):
                    last_reward = r_val
                    total_reward += r_val
                else:
                    non_finite = True
                    done = True

                x_cur = float(u.object_pos_at_time(u.get_time(), "robot")[0].mean())
                if np.isfinite(x_cur):
                    last_valid_x = x_cur
                else:
                    non_finite = True
                    done = True

                steps += 1

                if gif_path and ep == 0 and len(frames) < max_steps:
                    frames.append(env.render()[::2, ::2])

            x_final = float(u.object_pos_at_time(u.get_time(), "robot")[0].mean())
            if not np.isfinite(x_final):
                non_finite = True
                x_final = last_valid_x
            distance = (
                float(x_final - x_init)
                if np.isfinite(x_final) and np.isfinite(x_init)
                else 0.0
            )

            # EvoGym sets terminated=True both on goal completion (+1.0 bonus) and on
            # unstable simulation blow-up (-3.0 penalty). Separate the two cleanly.
            unstable = bool(non_finite or (terminated and last_reward < -1.0))
            goal_reached = bool(terminated and not unstable and last_reward > 0.5)

            norm_effort = total_effort / max(steps * grid_cells, 1)
            norm_jerk = (
                total_jerk / max((steps - 1) * grid_cells, 1) if steps > 1 else 0.0
            )
            avg_speed = distance / max(steps, 1)
            cot = (
                float(total_effort / distance)
                if (not non_finite and distance > 0.1)
                else None
            )
            cot_per_mass = (
                float(total_effort / (stats["n_voxels"] * distance))
                if (not non_finite and distance > 0.1)
                else None
            )

            speed_bonus = (
                w_speed * ((max_steps - steps) / max_steps) if goal_reached else 0.0
            )
            energy_penalty = w_energy * norm_effort + w_jerk * norm_jerk
            if non_finite or not np.isfinite(total_reward):
                total_reward = -10.0
                fitness = -1e6
            else:
                fitness = total_reward + speed_bonus - energy_penalty
                if not np.isfinite(fitness):
                    fitness = -1e6

            ep_metrics.append(
                {
                    "det_return": total_reward,
                    "distance": distance,
                    "steps": steps,
                    "max_steps": max_steps,
                    "goal_reached": goal_reached,
                    "unstable": unstable,
                    "avg_speed": avg_speed,
                    "total_energy": total_effort,
                    "norm_effort": norm_effort,
                    "norm_jerk": norm_jerk,
                    "cost_of_transport": cot,
                    "cot_per_mass": cot_per_mass,
                    "speed_bonus": speed_bonus,
                    "energy_penalty": energy_penalty,
                    "fitness": fitness,
                }
            )
    finally:
        env.close()

    if gif_path and frames:
        import imageio

        gif_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(gif_path, frames[::2], duration=1 / 25, loop=0)

    def mean_key(key: str) -> float:
        return float(np.mean([m[key] for m in ep_metrics]))

    valid_cots = [
        m["cost_of_transport"] for m in ep_metrics if m["cost_of_transport"] is not None
    ]
    valid_cots_mass = [
        m["cot_per_mass"] for m in ep_metrics if m["cot_per_mass"] is not None
    ]

    return {
        "det_return": round(mean_key("det_return"), 4),
        "distance": round(mean_key("distance"), 4),
        "steps": round(mean_key("steps"), 1),
        "max_steps": max_steps,
        "goal_reached": bool(all(m["goal_reached"] for m in ep_metrics)),
        "goal_rate": round(float(np.mean([m["goal_reached"] for m in ep_metrics])), 3),
        "unstable": bool(any(m["unstable"] for m in ep_metrics)),
        "avg_speed": round(mean_key("avg_speed"), 5),
        "total_energy": round(mean_key("total_energy"), 3),
        "norm_effort": round(mean_key("norm_effort"), 5),
        "norm_jerk": round(mean_key("norm_jerk"), 5),
        "cost_of_transport": round(float(np.mean(valid_cots)), 4)
        if valid_cots
        else None,
        "cot_per_mass": round(float(np.mean(valid_cots_mass)), 4)
        if valid_cots_mass
        else None,
        "speed_bonus": round(mean_key("speed_bonus"), 4),
        "energy_penalty": round(mean_key("energy_penalty"), 4),
        "fitness": round(mean_key("fitness"), 4),
        **stats,
    }


def _load_agent_from_dir(run_dir: Path, env_id: str, body: np.ndarray) -> PPOAgent:
    """Reconstruct a PPOAgent and load saved weights from a completed run directory."""
    env = make_env(env_id, {"body": body})
    try:
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]
    finally:
        env.close()
    agent = PPOAgent(state_size=state_size, action_size=action_size)
    agent.network.load_state_dict(
        torch.load(run_dir / "model.pt", map_location=agent.device, weights_only=True)
    )
    agent.network.eval()
    return agent


RESERVED_PPO_KEYS = {"seed", "out_dir", "env_kwargs"}


def _filter_ppo_kwargs(raw_kwargs: dict) -> dict:
    """Forward only parameters accepted by the current `ppo.train_agent` signature.

    Excludes per-robot orchestration keys (`seed`, `out_dir`, `env_kwargs`) so
    passing a saved `config.json` from Step 4 via `--ppo-config` never collides.
    """
    sig = inspect.signature(train_agent)
    accepted = set(sig.parameters.keys()) - RESERVED_PPO_KEYS
    unknown = sorted(set(raw_kwargs.keys()) - accepted - RESERVED_PPO_KEYS)
    if unknown:
        print(
            f"[warn] Ignoring PPO config keys not in current ppo.train_agent: {unknown}"
        )
    return {k: v for k, v in raw_kwargs.items() if k in accepted}


# ---------------------------------------------------------------------------
# Evolution loop with per-robot checkpointing, deduplication & lineage export
# ---------------------------------------------------------------------------


def _build_parent_schedule(survivors: list[dict], n_children: int) -> list[dict]:
    """Allocate mutated children deterministically across sorted top survivors.

    Uses rank-proportional largest-remainder allocation (weights k, k-1, ..., 1)
    so the #1 champion always receives the most offspring and every top survivor
    gets at least one child whenever n_children >= len(survivors).
    Example: 3 survivors, 6 children -> [3, 2, 1].
    """
    if n_children <= 0:
        return []
    if not survivors:
        raise ValueError("Cannot allocate children from an empty survivors list.")
    k = len(survivors)
    ranks = np.arange(k, 0, -1, dtype=np.float64)
    raw = ranks / ranks.sum() * n_children
    counts = np.floor(raw).astype(int)
    if n_children >= k:
        zeros = counts == 0
        if zeros.any():
            counts[zeros] = 1
            while int(counts.sum()) > n_children:
                idx = int(np.argmax(counts))
                counts[idx] -= 1
    rem = n_children - int(counts.sum())
    if rem > 0:
        order = np.argsort(-(raw - counts))
        for i in order[:rem]:
            counts[int(i)] += 1
    schedule: list[dict] = []
    for surv, count in zip(survivors, counts):
        schedule.extend([surv] * int(count))
    return schedule


def _generation_mutation_rate(
    gen: int, total_generations: int, start_rate: float, end_rate: float
) -> float:
    """Linearly anneal mutation rate from start_rate (gen 1) to end_rate (final gen)."""
    if total_generations <= 2 or gen <= 1:
        return float(start_rate)
    progress = min(1.0, (gen - 1) / (total_generations - 2))
    return float(start_rate + progress * (end_rate - start_rate))


def _write_summaries(out_dir: Path, history: dict) -> None:
    """Write evolution_history.json, summary.csv, and best_per_gen.csv."""
    (out_dir / "evolution_history.json").write_text(json.dumps(history, indent=2))

    rows = []
    best_rows = []
    for gen_rec in history["generations"]:
        gen_idx = gen_rec["generation"]
        inds = gen_rec["individuals"]
        for ind in inds:
            row = {
                "generation": gen_idx,
                "ind_idx": ind["ind_idx"],
                "robot_id": ind["robot_id"],
                "origin_id": ind["origin_id"],
                "body_hash": ind["body_hash"],
                "operator": ind["operator"],
                "parent_ids": ",".join(ind["parent_ids"]),
                "n_changed_voxels": ind.get("n_changed_voxels", 0),
                "reused": ind["reused"],
                "train_time_s": ind.get("train_time_s", 0.0),
                "train_avg100": ind.get("train_avg100"),
                "model_dir": ind["model_dir"],
            }
            row.update(ind["metrics"])
            rows.append(row)

        if inds:
            best = max(inds, key=lambda x: x["metrics"]["fitness"])
            fitnesses = [x["metrics"]["fitness"] for x in inds]
            returns = [x["metrics"]["det_return"] for x in inds]
            best_rows.append(
                {
                    "generation": gen_idx,
                    "best_robot_id": best["robot_id"],
                    "best_origin_id": best["origin_id"],
                    "best_body_hash": best["body_hash"],
                    "best_fitness": best["metrics"]["fitness"],
                    "mean_fitness": round(float(np.mean(fitnesses)), 4),
                    "min_fitness": round(float(np.min(fitnesses)), 4),
                    "best_det_return": best["metrics"]["det_return"],
                    "mean_det_return": round(float(np.mean(returns)), 4),
                    "best_steps": best["metrics"]["steps"],
                    "best_goal_reached": best["metrics"]["goal_reached"],
                    "best_cot": best["metrics"]["cost_of_transport"],
                    "best_norm_effort": best["metrics"]["norm_effort"],
                    "best_n_voxels": best["metrics"]["n_voxels"],
                    "best_n_actuators": best["metrics"]["n_actuators"],
                    "best_model_dir": best["model_dir"],
                }
            )

    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False)
    if best_rows:
        pd.DataFrame(best_rows).to_csv(out_dir / "best_per_gen.csv", index=False)


def _validate_args(args: argparse.Namespace) -> None:
    """Fail fast on invalid CLI bounds before starting any training."""
    if args.generations < 1:
        raise ValueError("--generations must be >= 1.")
    if args.pop_size < 1:
        raise ValueError("--pop-size must be >= 1.")
    if args.elites < 0 or (args.pop_size > 1 and args.elites >= args.pop_size):
        raise ValueError("--elites must satisfy 0 <= elites < pop_size.")
    if args.pop_size == 1 and args.elites != 0:
        raise ValueError("--elites must be 0 when --pop-size is 1.")
    if args.survivors < 1:
        raise ValueError("--survivors must be >= 1.")
    if args.grid_size < 2:
        raise ValueError("--grid-size must be >= 2.")
    if args.eval_episodes < 1:
        raise ValueError("--eval-episodes must be >= 1.")
    if not (0.0 <= args.mutation_rate <= 1.0):
        raise ValueError("--mutation-rate must be in [0.0, 1.0].")
    if args.mutation_rate_end is not None and not (
        0.0 <= args.mutation_rate_end <= 1.0
    ):
        raise ValueError("--mutation-rate-end must be in [0.0, 1.0].")
    if args.steps < 1:
        raise ValueError("--steps must be >= 1.")
    if args.n_envs < 1:
        raise ValueError("--n-envs must be >= 1.")
    if args.reval_top_k < 0:
        raise ValueError("--reval-top-k must be >= 0.")
    if args.reval_steps is not None and args.reval_steps < 1:
        raise ValueError("--reval-steps must be >= 1.")


def _build_ppo_kwargs(args: argparse.Namespace) -> dict:
    """Build filtered PPO kwargs from CLI flags and optional --ppo-config JSON."""
    ppo_kwargs = {
        "total_steps": args.steps,
        "update_freq": args.update_freq,
        "env_id": args.env,
        "lr": args.lr,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "entropy_beta": args.entropy_beta,
        "clip_epsilon": args.clip_epsilon,
        "ppo_epochs": args.ppo_epochs,
        "batch_size": args.batch_size,
        "norm_reward": args.norm_reward,
        "n_envs": args.n_envs,
        "vectorization": args.vectorization,
        "verbose": args.verbose_ppo,
    }
    if args.ppo_config:
        overrides = json.loads(Path(args.ppo_config).read_text())
        ppo_kwargs.update(overrides)
        ppo_kwargs["env_id"] = args.env
        ppo_kwargs["total_steps"] = args.steps
    return _filter_ppo_kwargs(ppo_kwargs)


def run_evolution(args: argparse.Namespace) -> dict:
    """Run (or resume) the mutation-only genetic algorithm."""
    _validate_args(args)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    ppo_kwargs = _build_ppo_kwargs(args)
    mut_end = (
        args.mutation_rate if args.mutation_rate_end is None else args.mutation_rate_end
    )

    evo_config = {
        **vars(args),
        "mutation_rate_end": mut_end,
        "effective_ppo_kwargs": ppo_kwargs,
        "fitness_formula": "det_return + w_speed * 1[goal] * (max_steps - steps)/max_steps - w_energy * norm_effort - w_jerk * norm_jerk",
    }
    (out_dir / "evo_config.json").write_text(json.dumps(evo_config, indent=2))

    shape = (args.grid_size, args.grid_size)
    n_elites = min(args.elites, args.pop_size - 1)
    n_survivors = max(n_elites, min(args.survivors, args.pop_size))

    # Cache of already-trained bodies keyed by canonical hash -> record with metrics & model_dir
    evaluated_cache: dict[str, dict] = {}
    seen_hashes: set[str] = set()
    history: dict = {"config": evo_config, "generations": []}

    t_evo_start = time.perf_counter()

    for gen in range(args.generations):
        # Per-generation RNG ensures identical populations even after an interrupted resume.
        rng = np.random.default_rng(args.seed + gen * 10_000)
        gen_dir = out_dir / f"gen_{gen:02d}"
        gen_dir.mkdir(parents=True, exist_ok=True)
        plan_path = gen_dir / "population.json"

        # 1. Build or reload this generation's planned bodies (for exact resumability)
        if plan_path.exists():
            planned = json.loads(plan_path.read_text())
            for spec in planned:
                seen_hashes.add(spec["body_hash"])
            print(
                f"\n=== Generation {gen:02d}/{args.generations - 1:02d} (resuming planned population) ==="
            )
        else:
            planned = []
            if gen == 0:
                print(
                    f"\n=== Generation 00/{args.generations - 1:02d} (sampling {args.pop_size} random blobs) ==="
                )
                for ind_idx in range(args.pop_size):
                    body = sample_random_body(rng, shape=shape, seen_hashes=seen_hashes)
                    bhash = canonical_body_hash(body)
                    seen_hashes.add(bhash)
                    planned.append(
                        {
                            "ind_idx": ind_idx,
                            "robot_id": f"gen{gen:02d}_ind{ind_idx:02d}",
                            "body": body.tolist(),
                            "body_hash": bhash,
                            "operator": "random",
                            "parent_ids": [],
                            "n_changed_voxels": 0,
                        }
                    )
            else:
                prev_inds = history["generations"][gen - 1]["individuals"]
                ranked = sorted(
                    prev_inds, key=lambda x: x["metrics"]["fitness"], reverse=True
                )
                elites = ranked[:n_elites]
                survivors = ranked[:n_survivors]

                gen_mut_rate = _generation_mutation_rate(
                    gen, args.generations, args.mutation_rate, mut_end
                )
                n_mutants = args.pop_size - n_elites
                parent_schedule = _build_parent_schedule(survivors, n_mutants)
                print(
                    f"\n=== Generation {gen:02d}/{args.generations - 1:02d} "
                    f"(elites={n_elites}, mutants={n_mutants}, survivors={n_survivors}, p_mut={gen_mut_rate:.3f}) ==="
                )

                for ind_idx, elite in enumerate(elites):
                    planned.append(
                        {
                            "ind_idx": ind_idx,
                            "robot_id": f"gen{gen:02d}_ind{ind_idx:02d}",
                            "body": elite["body"],
                            "body_hash": elite["body_hash"],
                            "operator": "elite",
                            "parent_ids": [elite["robot_id"]],
                            "n_changed_voxels": 0,
                        }
                    )

                for ind_idx, parent in enumerate(parent_schedule, start=n_elites):
                    child, n_changed = mutate_body(
                        np.array(parent["body"], dtype=int),
                        rng=rng,
                        mutation_rate=gen_mut_rate,
                        seen_hashes=seen_hashes,
                    )
                    bhash = canonical_body_hash(child)
                    seen_hashes.add(bhash)
                    planned.append(
                        {
                            "ind_idx": ind_idx,
                            "robot_id": f"gen{gen:02d}_ind{ind_idx:02d}",
                            "body": child.tolist(),
                            "body_hash": bhash,
                            "operator": "mutation",
                            "parent_ids": [parent["robot_id"]],
                            "n_changed_voxels": n_changed,
                        }
                    )

            plan_path.write_text(json.dumps(planned, indent=2))

        # 2. Train / reuse & evaluate every individual in this generation
        gen_records = []
        for spec in planned:
            ind_idx = spec["ind_idx"]
            robot_id = spec["robot_id"]
            body = np.array(spec["body"], dtype=int)
            bhash = spec["body_hash"]
            ind_dir = gen_dir / f"ind_{ind_idx:02d}"
            ind_dir.mkdir(parents=True, exist_ok=True)
            (ind_dir / "body.json").write_text(json.dumps(spec["body"], indent=2))
            eval_path = ind_dir / "eval.json"

            # Case A: Already completed on disk (resume)
            if eval_path.exists():
                rec = json.loads(eval_path.read_text())
                if bhash not in evaluated_cache:
                    evaluated_cache[bhash] = rec
                gen_records.append(rec)
                m = rec["metrics"]
                print(
                    f"  [resume] {robot_id} ({rec['operator']:8s}) | "
                    f"fit={m['fitness']:6.2f} ret={m['det_return']:6.2f} "
                    f"steps={m['steps']:5.0f} goal={int(m['goal_reached'])} "
                    f"effort={m['norm_effort']:.3f} cot={m['cost_of_transport']}"
                )
                continue

            # Case B: Elite or duplicate canonical body already evaluated -> reuse without retraining
            if bhash in evaluated_cache:
                cached = evaluated_cache[bhash]
                rec = {
                    **spec,
                    "origin_id": cached["origin_id"],
                    "reused": True,
                    "train_time_s": 0.0,
                    "train_avg100": cached["train_avg100"],
                    "model_dir": cached["model_dir"],
                    "metrics": cached["metrics"],
                }
                eval_path.write_text(json.dumps(rec, indent=2))
                gen_records.append(rec)
                m = rec["metrics"]
                print(
                    f"  [reuse ] {robot_id} ({spec['operator']:8s} <- {cached['origin_id']}) | "
                    f"fit={m['fitness']:6.2f} ret={m['det_return']:6.2f} "
                    f"steps={m['steps']:5.0f} goal={int(m['goal_reached'])} "
                    f"effort={m['norm_effort']:.3f} cot={m['cost_of_transport']}"
                )
                continue

            # Case C: Fresh robot -> train with PPO (unchanged) and evaluate deterministically
            # Space seeds by at least max(100, n_envs) because ppo.py seeds worker i with seed + i.
            run_seed = args.seed + gen * 100_000 + ind_idx * max(100, args.n_envs)
            run_cfg = {
                **ppo_kwargs,
                "seed": run_seed,
                "robot_id": robot_id,
                "body_hash": bhash,
                "body": spec["body"],
            }
            (ind_dir / "config.json").write_text(json.dumps(run_cfg, indent=2))

            t0 = time.perf_counter()
            agent, rewards, *_ = train_agent(
                **ppo_kwargs,
                seed=run_seed,
                out_dir=ind_dir,
                env_kwargs={"body": body},
            )
            train_time_s = round(time.perf_counter() - t0, 1)
            train_avg100 = round(float(np.mean(rewards[-100:])), 4) if rewards else None

            metrics = evaluate_policy(
                agent=agent,
                body=body,
                env_id=args.env,
                episodes=args.eval_episodes,
                seed=args.eval_seed,
                w_speed=args.w_speed,
                w_energy=args.w_energy,
                w_jerk=args.w_jerk,
            )

            rel_model_dir = str(ind_dir.relative_to(out_dir))
            rec = {
                **spec,
                "origin_id": robot_id,
                "reused": False,
                "train_seed": run_seed,
                "train_time_s": train_time_s,
                "train_avg100": train_avg100,
                "model_dir": rel_model_dir,
                "metrics": metrics,
            }
            eval_path.write_text(json.dumps(rec, indent=2))
            evaluated_cache[bhash] = rec
            gen_records.append(rec)

            print(
                f"  [train ] {robot_id} ({spec['operator']:8s}, vox={metrics['n_voxels']:2d}, act={metrics['n_actuators']:2d}) | "
                f"fit={metrics['fitness']:6.2f} ret={metrics['det_return']:6.2f} "
                f"steps={metrics['steps']:5.0f} goal={int(metrics['goal_reached'])} "
                f"effort={metrics['norm_effort']:.3f} cot={metrics['cost_of_transport']} ({train_time_s:.1f}s)"
            )

        # Save GIF of this generation's best robot if requested
        best_gen = max(gen_records, key=lambda x: x["metrics"]["fitness"])
        if args.save_gen_gifs:
            gif_out = gen_dir / f"best_{best_gen['robot_id']}.gif"
            if not gif_out.exists():
                best_model_dir = out_dir / best_gen["model_dir"]
                best_body = np.array(best_gen["body"], dtype=int)
                best_agent = _load_agent_from_dir(best_model_dir, args.env, best_body)
                evaluate_policy(
                    agent=best_agent,
                    body=best_body,
                    env_id=args.env,
                    episodes=1,
                    seed=args.eval_seed,
                    w_speed=args.w_speed,
                    w_energy=args.w_energy,
                    w_jerk=args.w_jerk,
                    gif_path=gif_out,
                )

        gen_summary = {
            "generation": gen,
            "best_robot_id": best_gen["robot_id"],
            "best_fitness": best_gen["metrics"]["fitness"],
            "mean_fitness": round(
                float(np.mean([r["metrics"]["fitness"] for r in gen_records])), 4
            ),
            "individuals": gen_records,
        }
        history["generations"].append(gen_summary)
        _write_summaries(out_dir, history)

        print(
            f"--> Gen {gen:02d} summary | best={best_gen['robot_id']} "
            f"fit={best_gen['metrics']['fitness']:.2f} "
            f"(ret={best_gen['metrics']['det_return']:.2f}, steps={best_gen['metrics']['steps']:.0f}, "
            f"cot={best_gen['metrics']['cost_of_transport']}) | "
            f"mean_fit={gen_summary['mean_fitness']:.2f}"
        )

    total_min = (time.perf_counter() - t_evo_start) / 60.0
    all_inds = [ind for g in history["generations"] for ind in g["individuals"]]
    champion = max(all_inds, key=lambda x: x["metrics"]["fitness"])
    (out_dir / "champion_body.json").write_text(json.dumps(champion["body"], indent=2))
    print(
        f"\nEvolution complete in {total_min:.1f} min! "
        f"Champion: {champion['origin_id']} | fit={champion['metrics']['fitness']:.2f} "
        f"| ret={champion['metrics']['det_return']:.2f} | steps={champion['metrics']['steps']:.0f} "
        f"| cot={champion['metrics']['cost_of_transport']} | saved to {out_dir}/"
    )
    if args.reval_top_k > 0 and args.reval_seeds:
        run_reevaluation(out_dir, history, args, ppo_kwargs)
    return history


def run_reevaluation(
    out_dir: Path,
    history: dict,
    args: argparse.Namespace,
    ppo_kwargs: dict | None = None,
) -> pd.DataFrame:
    """Re-train the top-K unique evolved bodies, best Gen-0 body, and optional
    hand-designed reference across multiple PPO seeds to eliminate lucky-seed
    elite bias and compare pure speed alongside energy efficiency.
    """
    torch.set_num_threads(1)
    reval_dir = out_dir / "reval"
    reval_dir.mkdir(parents=True, exist_ok=True)

    base_ppo = dict(ppo_kwargs or _build_ppo_kwargs(args))
    reval_steps = int(args.reval_steps or base_ppo.get("total_steps", args.steps))
    base_ppo["total_steps"] = reval_steps
    env_id = base_ppo.get("env_id", args.env)

    # Deduplicate all unique morphologies trained during evolution by canonical hash.
    unique_by_hash: dict[str, dict] = {}
    for gen_rec in history["generations"]:
        for ind in gen_rec["individuals"]:
            h = ind["body_hash"]
            if (
                h not in unique_by_hash
                or ind["metrics"]["fitness"] > unique_by_hash[h]["metrics"]["fitness"]
            ):
                unique_by_hash[h] = ind

    ranked_unique = sorted(
        unique_by_hash.values(), key=lambda x: x["metrics"]["fitness"], reverse=True
    )

    candidates: list[dict] = []
    added_hashes: set[str] = set()
    for rank, ind in enumerate(ranked_unique[: args.reval_top_k], start=1):
        candidates.append(
            {
                "label": f"top{rank}_{ind['origin_id']}",
                "group": "evolved",
                "origin_id": ind["origin_id"],
                "body_hash": ind["body_hash"],
                "body": ind["body"],
            }
        )
        added_hashes.add(ind["body_hash"])

    if history["generations"]:
        gen0_best = max(
            history["generations"][0]["individuals"],
            key=lambda x: x["metrics"]["fitness"],
        )
        if gen0_best["body_hash"] not in added_hashes:
            candidates.append(
                {
                    "label": f"gen0_best_{gen0_best['origin_id']}",
                    "group": "gen0_random",
                    "origin_id": gen0_best["origin_id"],
                    "body_hash": gen0_best["body_hash"],
                    "body": gen0_best["body"],
                }
            )
            added_hashes.add(gen0_best["body_hash"])

    if args.reval_ref and args.reval_ref.lower() != "none":
        ref_body = load_body(args.reval_ref).astype(int)
        ref_name = Path(args.reval_ref).stem
        candidates.append(
            {
                "label": f"ref_{ref_name}",
                "group": "hand_designed",
                "origin_id": ref_name,
                "body_hash": canonical_body_hash(ref_body),
                "body": ref_body.tolist(),
            }
        )

    print(
        f"\n=== Multi-Seed Re-Evaluation ({len(candidates)} bodies x "
        f"{len(args.reval_seeds)} seeds @ {reval_steps:,} steps) ==="
    )

    run_rows = []
    for cand in candidates:
        body = np.array(cand["body"], dtype=int)
        for seed in args.reval_seeds:
            run_dir = reval_dir / f"{cand['label']}_s{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "body.json").write_text(json.dumps(cand["body"], indent=2))
            eval_path = run_dir / "eval.json"

            if eval_path.exists():
                rec = json.loads(eval_path.read_text())
                cached_steps = rec.get("reval_steps")
                if cached_steps is not None and cached_steps != reval_steps:
                    print(
                        f"  [warn  ] {cand['label']} s{seed}: cached eval used {cached_steps:,} steps "
                        f"(requested {reval_steps:,}); keeping cached result."
                    )
                tag = "resume"
            else:
                run_cfg = {
                    **base_ppo,
                    "seed": seed,
                    "label": cand["label"],
                    "group": cand["group"],
                    "body_hash": cand["body_hash"],
                    "body": cand["body"],
                }
                (run_dir / "config.json").write_text(json.dumps(run_cfg, indent=2))
                t0 = time.perf_counter()
                agent, rewards, *_ = train_agent(
                    **base_ppo,
                    seed=seed,
                    out_dir=run_dir,
                    env_kwargs={"body": body},
                )
                train_time_s = round(time.perf_counter() - t0, 1)
                train_avg100 = (
                    round(float(np.mean(rewards[-100:])), 4) if rewards else None
                )
                metrics = evaluate_policy(
                    agent=agent,
                    body=body,
                    env_id=env_id,
                    episodes=args.eval_episodes,
                    seed=args.eval_seed,
                    w_speed=args.w_speed,
                    w_energy=args.w_energy,
                    w_jerk=args.w_jerk,
                )
                rec = {
                    "label": cand["label"],
                    "group": cand["group"],
                    "origin_id": cand["origin_id"],
                    "body_hash": cand["body_hash"],
                    "seed": seed,
                    "reval_steps": reval_steps,
                    "train_time_s": train_time_s,
                    "train_avg100": train_avg100,
                    "metrics": metrics,
                }
                eval_path.write_text(json.dumps(rec, indent=2))
                tag = "train "

            m = rec["metrics"]
            print(
                f"  [{tag}] {cand['label']:22s} s{seed} | "
                f"ret={m['det_return']:6.2f} steps={m['steps']:5.0f} goal={int(m['goal_reached'])} "
                f"spd={m['avg_speed'] * 100:.2f} | cot={m['cost_of_transport']} "
                f"eff={m['norm_effort']:.3f} | fit={m['fitness']:6.2f}"
            )
            run_rows.append(
                {
                    "label": cand["label"],
                    "group": cand["group"],
                    "origin_id": cand["origin_id"],
                    "body_hash": cand["body_hash"],
                    "seed": seed,
                    "reval_steps": rec.get("reval_steps", reval_steps),
                    "train_time_s": rec["train_time_s"],
                    "train_avg100": rec["train_avg100"],
                    **m,
                }
            )

    runs_df = pd.DataFrame(run_rows)
    runs_df.to_csv(reval_dir / "reval_runs.csv", index=False)

    summary_rows = []
    for cand in candidates:
        sub = runs_df[runs_df["label"] == cand["label"]]
        valid_cot = sub["cost_of_transport"].dropna()
        summary_rows.append(
            {
                "label": cand["label"],
                "group": cand["group"],
                "n_seeds": len(sub),
                "reval_steps": reval_steps,
                # Pure performance / speed metrics (no energy penalty)
                "det_return_mean": round(float(sub["det_return"].mean()), 3),
                "det_return_std": round(float(sub["det_return"].std(ddof=0)), 3),
                "steps_mean": round(float(sub["steps"].mean()), 1),
                "steps_std": round(float(sub["steps"].std(ddof=0)), 1),
                "avg_speed_mean": round(float(sub["avg_speed"].mean()), 5),
                "goal_rate": round(float(sub["goal_reached"].mean()), 3),
                # Energy efficiency metrics
                "cot_mean": round(float(valid_cot.mean()), 2)
                if len(valid_cot)
                else None,
                "cot_std": round(float(valid_cot.std(ddof=0)), 2)
                if len(valid_cot)
                else None,
                "norm_effort_mean": round(float(sub["norm_effort"].mean()), 4),
                "n_voxels": int(sub["n_voxels"].iloc[0]),
                "n_actuators": int(sub["n_actuators"].iloc[0]),
                # Composite organic fitness
                "fitness_mean": round(float(sub["fitness"].mean()), 3),
                "fitness_std": round(float(sub["fitness"].std(ddof=0)), 3),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(reval_dir / "reval_summary.csv", index=False)
    print("\n=== Multi-Seed Re-Evaluation Summary (Pure Speed + Energy + Fitness) ===")
    print(summary_df.to_string(index=False))
    return summary_df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 5 — Genetic Algorithm for 5x5 EvoGym body evolution (mutation-only)."
    )
    # Task & output
    p.add_argument(
        "--env",
        default="Walker-v0",
        help="EvoGym environment id (e.g. Walker-v0, UpStepper-v0)",
    )
    p.add_argument(
        "--out", required=True, help="Output directory (supports automatic resume)"
    )

    # GA hyperparameters
    p.add_argument("--pop-size", type=int, default=8, help="Robots per generation")
    p.add_argument(
        "--generations",
        type=int,
        default=10,
        help="Number of generations (incl. gen 0)",
    )
    p.add_argument(
        "--elites",
        type=int,
        default=2,
        help="Top elites kept unchanged without retraining",
    )
    p.add_argument(
        "--survivors",
        type=int,
        default=3,
        help="Top survivors with guaranteed rank-proportional offspring",
    )
    p.add_argument(
        "--mutation-rate",
        type=float,
        default=0.12,
        help="Initial per-voxel mutation probability (gen 1)",
    )
    p.add_argument(
        "--mutation-rate-end",
        type=float,
        default=0.06,
        help="Final per-voxel mutation probability (linearly annealed to final gen)",
    )
    p.add_argument(
        "--grid-size", type=int, default=5, help="Square voxel grid size (default 5x5)"
    )
    p.add_argument("--seed", type=int, default=31, help="Master random seed")

    # Organic fitness weights (evaluated deterministically, PPO reward untouched)
    p.add_argument(
        "--w-speed",
        type=float,
        default=3.0,
        help="Speed bonus weight when goal is reached",
    )
    p.add_argument(
        "--w-energy",
        type=float,
        default=0.5,
        help="Normalized actuation effort penalty weight",
    )
    p.add_argument(
        "--w-jerk",
        type=float,
        default=0.25,
        help="Normalized action jerk (smoothness) penalty weight",
    )
    p.add_argument(
        "--eval-episodes",
        type=int,
        default=1,
        help="Deterministic eval episodes per robot",
    )
    p.add_argument(
        "--eval-seed", type=int, default=0, help="Deterministic evaluation seed"
    )
    p.add_argument(
        "--save-gen-gifs",
        action="store_true",
        help="Save a GIF of each generation's best robot",
    )

    # Post-evolution multi-seed re-evaluation (eliminates lucky-seed elite bias)
    p.add_argument(
        "--reval-top-k",
        type=int,
        default=0,
        help="Re-train top-K unique evolved bodies across --reval-seeds (0 disables)",
    )
    p.add_argument(
        "--reval-seeds",
        type=int,
        nargs="+",
        default=[31, 32, 33],
        help="Seeds for post-evolution multi-seed re-evaluation",
    )
    p.add_argument(
        "--reval-steps",
        type=int,
        default=None,
        help="PPO env steps per seed during re-evaluation (defaults to --steps)",
    )
    p.add_argument(
        "--reval-ref",
        default="walker",
        help="External reference robot to include in re-evaluation ('none' to skip)",
    )
    p.add_argument(
        "--reval-only",
        action="store_true",
        help="Skip evolution and run multi-seed re-evaluation on an existing --out run",
    )

    # Inner-loop PPO settings (forwarded directly to ppo.train_agent without modifying ppo.py)
    p.add_argument(
        "--ppo-config",
        default=None,
        help="Optional JSON file overriding PPO kwargs (e.g. Step 4 best PPO)",
    )
    p.add_argument("--steps", type=int, default=500_000, help="PPO env steps per robot")
    p.add_argument("--n-envs", type=int, default=8, help="Parallel envs per robot")
    p.add_argument("--vectorization", choices=["sync", "async"], default="async")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--entropy-beta", type=float, default=0.0)
    p.add_argument("--clip-epsilon", type=float, default=0.2)
    p.add_argument("--ppo-epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--update-freq", type=int, default=2048)
    p.add_argument("--norm-reward", action="store_true")
    p.add_argument(
        "--verbose-ppo", action="store_true", help="Print PPO progress logs per robot"
    )
    return p.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.reval_only:
        _validate_args(cli_args)
        out_path = Path(cli_args.out)
        hist_path = out_path / "evolution_history.json"
        if not hist_path.exists():
            raise FileNotFoundError(f"Cannot run --reval-only: missing {hist_path}.")
        saved_history = json.loads(hist_path.read_text())
        saved_cfg = saved_history.get("config", {})
        # Inherit original evolution's env, fitness weights, and eval config for apples-to-apples re-eval.
        for key in (
            "env",
            "w_speed",
            "w_energy",
            "w_jerk",
            "eval_episodes",
            "eval_seed",
        ):
            if key in saved_cfg:
                setattr(cli_args, key, saved_cfg[key])
        if cli_args.reval_top_k == 0:
            cli_args.reval_top_k = 3
        saved_ppo = saved_cfg.get("effective_ppo_kwargs")
        if saved_ppo and not cli_args.ppo_config:
            ppo_for_reval = _filter_ppo_kwargs(saved_ppo)
        else:
            ppo_for_reval = _build_ppo_kwargs(cli_args)
        run_reevaluation(out_path, saved_history, cli_args, ppo_kwargs=ppo_for_reval)
    else:
        run_evolution(cli_args)
