"""Shared metrics for the Walker-v0 PPO experiments (step 3 baseline and step 4 optimisations).

Walker-v0 returns saturate at ≈ 10.5 once the robot reaches the end of the track, so besides the
return we measure speed (steps to goal) and learning efficiency (training steps to 90 % success).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ppo import PPOAgent, make_env, to_env_action

LIMIT = 500  # Walker-v0 episode limit: a shorter episode means the goal was reached


def load_run(path):
    """episodes.csv + log.csv of one run, with a success flag and the approximate training step of each episode."""
    path = Path(path)
    ep = pd.read_csv(path / "episodes.csv")
    lg = pd.read_csv(path / "log.csv")
    ep["success"] = ep["length"] < LIMIT
    ep["end_step"] = ep["length"].cumsum() * (lg["steps"].iloc[-1] / ep["length"].sum())
    return ep, lg


def steps_to_reliable_success(ep, window=50, threshold=0.9):
    """Training steps before >= threshold of the last `window` episodes reach the goal (NaN if never)."""
    rate = ep["success"].rolling(window).mean().to_numpy()
    hit = np.flatnonzero(rate >= threshold)
    return float(ep["end_step"].iloc[hit[0]]) if len(hit) else np.nan


def deterministic_test(path, body, env_id="Walker-v0"):
    """(steps, return) of one deterministic episode (action = mean μ) with the saved weights."""
    env = make_env(env_id, {"body": body})
    agent = PPOAgent(state_size=env.observation_space.shape[0], action_size=env.action_space.shape[0])
    agent.network.load_state_dict(torch.load(Path(path) / "model.pt"))
    s, _ = env.reset(seed=0)
    n, total, done = 0, 0.0, False
    while not done:
        with torch.no_grad():
            mean = agent.network(torch.as_tensor(s, dtype=torch.float32).unsqueeze(0))[0].numpy()[0]
        s, r, terminated, truncated, _ = env.step(to_env_action(mean, env.action_space))
        n, total, done = n + 1, total + r, terminated or truncated
    env.close()
    return n, total


def run_metrics(path, body):
    """One row of metrics for one run."""
    ep, _ = load_run(path)
    last = ep.tail(100)
    test_steps, test_return = deterministic_test(path, body)
    return {"return_last100": last["episode_return"].mean(),
            "success_last100_%": 100 * last["success"].mean(),
            "steps_to_goal_last100": last.loc[last["success"], "length"].mean(),
            "test_steps_to_goal": test_steps,
            "test_return": test_return,
            "train_steps_to_90%_success": steps_to_reliable_success(ep)}


def experiment_table(prefix, body, seeds=(31, 32, 33), runs_dir="../runs"):
    """Per-seed metrics of runs named `{prefix}_s{seed}`."""
    rows = [{"seed": s, **run_metrics(Path(runs_dir) / f"{prefix}_s{s}", body)} for s in seeds]
    return pd.DataFrame(rows).set_index("seed")
