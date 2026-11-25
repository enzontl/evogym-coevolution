"""PPO with continuous actions (Gaussian policy), adapted from the course PPO (TD6).

Lines marked `# CHANGED` / `# ADDED` differ from the TD6 version (discrete actions, LunarLander).
Contents: ActorCriticNetwork, PPOAgent, to_env_action, RunningMeanStd, make_vec_env, train_agent, test_agent.
Training runs n_envs copies of the environment in parallel (buffers are (T, n_envs)).

Command line (see `python ppo.py --help`):
    caffeinate -i python ppo.py --env Pendulum-v1 --steps 200000 --vectorization sync \\
        --norm-reward --seed 31 --out runs/pendulum_norm_s31
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ActorCriticNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_size, action_size)      # CHANGED: mean μ of a Gaussian, no softmax
        self.critic = nn.Linear(hidden_size, 1)
        self.log_std = nn.Parameter(torch.zeros(action_size))  # ADDED: learned log σ (state-independent)

    def forward(self, x):
        features = self.shared(x)
        mean = self.actor(features)
        state_value = self.critic(features)
        return mean, self.log_std.expand_as(mean), state_value  # CHANGED: (μ, log σ, V) instead of (probs, V)


class PPOAgent:
    def __init__(
        self, lr=3e-4, gamma=0.99, state_size=3, action_size=1,
        gae_lambda=0.95, entropy_beta=0.0, clip_epsilon=0.2, ppo_epochs=10, batch_size=64
    ):
        # Hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_beta = entropy_beta
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size

        # Environment parameters
        self.state_size = state_size
        self.action_size = action_size  # CHANGED: number of action dimensions (was n_actions)

        # Tiny MLP: CPU is faster than MPS on a Mac
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f"Using device: {self.device}")

        self.network = ActorCriticNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.reset_memory()

    def reset_memory(self):
        """Clear rollout buffers."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.log_probs = []

    def get_dist(self, states):
        """ADDED: Gaussian policy distribution and state values for a batch of states."""
        mean, log_std, value = self.network(states)
        return torch.distributions.Normal(mean, log_std.exp()), value

    def choose_action(self, states):
        """Sample one action per parallel env. states: (n_envs, state_size)."""
        state_tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)  # CHANGED: batch of envs
        with torch.no_grad():
            dist, value = self.get_dist(state_tensor)    # CHANGED: Normal instead of Categorical
        action = dist.sample()                            # (n_envs, action_size)
        log_prob = dist.log_prob(action).sum(-1)          # CHANGED: sum over independent action dims -> (n_envs,)
        return action.cpu().numpy(), value.squeeze(-1).cpu().numpy(), log_prob.cpu().numpy()

    def get_value(self, states):
        """ADDED: V(s) for a batch of states (bootstrapping), shape (n_envs,)."""
        with torch.no_grad():
            return self.network(torch.as_tensor(states, dtype=torch.float32, device=self.device))[2].squeeze(-1).cpu().numpy()

    def store(self, state, action, reward, value, done, log_prob):
        """Store one step of all parallel envs (arrays of length n_envs).

        done marks termination OR time-limit truncation; the training loop adds the
        bootstrap value to time-limit rewards.
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.log_probs.append(log_prob)

    def compute_gae(self, next_value):
        """Compute returns and advantages with GAE: same recursion as TD6, on (T, n_envs) arrays."""
        rewards = np.array(self.rewards, dtype=np.float32)                 # (T, n_envs)
        values = np.array(self.values + [next_value], dtype=np.float32)    # (T+1, n_envs)
        dones = np.array(self.dones, dtype=np.float32)                     # (T, n_envs)

        T = len(rewards)
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)

        gae = np.zeros(rewards.shape[1:], dtype=np.float32)  # CHANGED: one running GAE per env
        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * values[t + 1] * (1.0 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        return returns, advantages

    def update(self, next_value):
        """Perform one PPO update step."""
        if len(self.states) == 0:
            return {"actor_loss": 0.0, "critic_loss": 0.0}

        returns, advantages = self.compute_gae(next_value)

        def flat(x):
            """ADDED: (T, n_envs, ...) -> (T * n_envs, ...); after this, the update is the TD6 one."""
            x = np.asarray(x, dtype=np.float32)
            return torch.as_tensor(x.reshape(-1, *x.shape[2:]), device=self.device)

        states = flat(self.states)            # (N, state_size)
        actions = flat(self.actions)          # CHANGED: (N, action_size) floats
        returns = flat(returns)               # (N,)
        advantages = flat(advantages)         # (N,)
        old_log_probs = flat(self.log_probs)  # (N,)

        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_actor_loss = 0
        total_critic_loss = 0
        updates = 0

        for _ in range(self.ppo_epochs):
            indices = torch.randperm(len(states))
            for start in range(0, len(states), self.batch_size):
                idx = indices[start:start + self.batch_size]

                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_returns = returns[idx]
                batch_advantages = advantages[idx]

                dist, values = self.get_dist(batch_states)               # CHANGED: Normal
                log_probs = dist.log_prob(batch_actions).sum(-1)         # CHANGED: sum over action dims
                entropy = dist.entropy().sum(-1).mean()                  # CHANGED: sum over action dims

                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = F.mse_loss(values.squeeze(-1), batch_returns)

                loss = actor_loss + 0.5 * critic_loss - self.entropy_beta * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                updates += 1

        self.reset_memory()

        return {
            'actor_loss': total_actor_loss / updates,
            'critic_loss': total_critic_loss / updates
        }


def to_env_action(action, action_space):
    """ADDED: map a policy action in [-1, 1] to [low, high] of the environment."""
    a = np.clip(action, -1.0, 1.0)
    return action_space.low + (a + 1.0) * 0.5 * (action_space.high - action_space.low)


class RunningMeanStd:
    """ADDED: running mean / variance (parallel update formula)."""
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x):
        x = np.asarray(x, dtype=np.float64).reshape(-1, *np.shape(self.mean))
        batch_mean, batch_var, batch_count = x.mean(0), x.var(0), x.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        self.var = (self.var * self.count + batch_var * batch_count
                    + delta ** 2 * self.count * batch_count / total) / total
        self.count = total


def make_vec_env(env_id, n_envs, vectorization="async"):
    """ADDED: n copies of the env. 'async' = one process per env (slow envs like EvoGym),
    'sync' = all in this process (fast envs like Pendulum, where process overhead dominates).
    SAME_STEP autoreset: a finished env restarts in the same step() call and its last
    observation is returned in info["final_obs"]."""
    vec_cls = gym.vector.AsyncVectorEnv if vectorization == "async" else gym.vector.SyncVectorEnv
    return vec_cls([lambda: gym.make(env_id) for _ in range(n_envs)],
                   autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)


def train_agent(total_steps=200_000, update_freq=2048, env_id="Pendulum-v1",
                lr=3e-4, gamma=0.99, gae_lambda=0.95, entropy_beta=0.0,
                clip_epsilon=0.2, ppo_epochs=10, batch_size=64, seed=31,
                norm_reward=False, n_envs=8, vectorization="async", verbose=True,
                out_dir=None, save_every=25):
    """Train a PPO agent on n_envs parallel copies of the env.

    CHANGED vs TD6: the loop counts env steps (total_steps) instead of episodes, because the
    parallel envs finish their episodes at different times. Each rollout = update_freq
    transitions in total (update_freq // n_envs steps of every env), then one PPO update.
    If out_dir is given, logs and weights are saved there every `save_every` updates and at the end.
    """
    if total_steps < 1 or update_freq < n_envs:
        raise ValueError("total_steps must be positive and update_freq >= n_envs.")

    np.random.seed(seed)
    torch.manual_seed(seed)

    envs = make_vec_env(env_id, n_envs, vectorization)
    action_space = envs.single_action_space
    state_size = envs.single_observation_space.shape[0]
    action_size = action_space.shape[0]   # CHANGED: continuous action dimensions
    if verbose:
        print(f"State size: {state_size}, Action size: {action_size}, bounds {action_space.low}..{action_space.high}, "
              f"{n_envs} envs ({vectorization})")

    agent = PPOAgent(state_size=state_size, action_size=action_size, lr=lr, gamma=gamma,
                     gae_lambda=gae_lambda, entropy_beta=entropy_beta, clip_epsilon=clip_epsilon,
                     ppo_epochs=ppo_epochs, batch_size=batch_size)

    rewards_history, actor_loss_history, critic_loss_history = [], [], []
    std_history = []  # ADDED: policy σ after each update (exploration should shrink)

    reward_rms = RunningMeanStd() if norm_reward else None  # ADDED: std of the discounted return (shared)
    disc_return = np.zeros(n_envs)                          # one discounted return per env
    ep_reward = np.zeros(n_envs)                            # raw episode return per env

    rollout_steps = update_freq // n_envs
    state, _ = envs.reset(seed=seed)  # env i is seeded with seed + i
    steps, next_log = 0, 50
    update_log, t0 = [], time.perf_counter()  # ADDED: one row per PPO update (saved as log.csv)

    while steps < total_steps:
        for _ in range(rollout_steps):
            action, value, log_prob = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = envs.step(to_env_action(action, action_space))  # CHANGED
            done = terminated | truncated

            # ADDED: reward normalisation (training signal only; ep_reward below stays raw)
            stored_reward = reward.astype(np.float64)
            if reward_rms is not None:
                disc_return = disc_return * agent.gamma + reward
                reward_rms.update(disc_return)
                stored_reward = reward / np.sqrt(reward_rms.var + 1e-8)
                disc_return[done] = 0.0

            # Bootstrap a time limit from the episode's real last observation (info["final_obs"]),
            # not from next_state, which is already the first observation of the next episode.
            cut = truncated & ~terminated
            if cut.any():
                final_obs = np.stack(info["final_obs"][cut])
                stored_reward[cut] += agent.gamma * agent.get_value(final_obs)
            agent.store(state, action, stored_reward, value, done, log_prob)  # unclipped action stored

            ep_reward += reward
            for i in np.flatnonzero(done):
                rewards_history.append(ep_reward[i])
                ep_reward[i] = 0.0
            state = next_state
            steps += n_envs

        losses = agent.update(agent.get_value(state))  # GAE masks envs whose episode just ended
        actor_loss_history.append(losses["actor_loss"])
        critic_loss_history.append(losses["critic_loss"])
        std_history.append(agent.network.log_std.exp().mean().item())
        update_log.append({"steps": steps, "episodes": len(rewards_history),
                           "avg100": float(np.mean(rewards_history[-100:])) if rewards_history else np.nan,
                           "actor_loss": losses["actor_loss"], "critic_loss": losses["critic_loss"],
                           "sigma": std_history[-1], "time_s": round(time.perf_counter() - t0, 1)})
        if out_dir and len(update_log) % save_every == 0:
            save_run(out_dir, agent, update_log, rewards_history)  # checkpoint: a crash keeps the progress

        if verbose and len(rewards_history) >= next_log:
            print(f"Steps {steps:>8,} | Ep {len(rewards_history)}: Avg100 = {np.mean(rewards_history[-100:]):.1f}, "
                  f"sigma = {std_history[-1]:.3f}")
            next_log += 50

    envs.close()
    if out_dir:
        save_run(out_dir, agent, update_log, rewards_history)
    if verbose:
        print("Training complete!")
    return agent, rewards_history, actor_loss_history, critic_loss_history, std_history


def save_run(out_dir, agent, update_log, rewards_history):
    """ADDED: write log.csv (per update), episodes.csv (per episode) and model.pt (network weights)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(update_log).to_csv(out / "log.csv", index=False)
    pd.DataFrame({"episode_return": rewards_history}).to_csv(out / "episodes.csv", index=False)
    torch.save(agent.network.state_dict(), out / "model.pt")


def test_agent(agent, env_id="Pendulum-v1", episodes=5, seed=0):
    """Run a trained PPO agent in test mode: action = mean μ (CHANGED: was argmax)."""
    env = gym.make(env_id)
    totals = []
    for ep in range(episodes):
        state, _ = env.reset(seed=seed + ep)
        done, total_r = False, 0.0
        while not done:
            s_t = torch.as_tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
            with torch.no_grad():
                mean, _, _ = agent.network(s_t)
            state, r, terminated, truncated, _ = env.step(to_env_action(mean.squeeze(0).cpu().numpy(), env.action_space))
            done = terminated or truncated
            total_r += r
        totals.append(total_r)
        print(f"Test Episode {ep+1}: {total_r:.1f}")
    env.close()
    return totals


def parse_args():
    p = argparse.ArgumentParser(description="Train continuous-action PPO (course TD6 defaults).")
    p.add_argument("--env", default="Pendulum-v1", help="Gymnasium env id")
    p.add_argument("--steps", type=int, default=200_000, help="total env steps (all envs together)")
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--vectorization", choices=["sync", "async"], default="async",
                   help="sync for fast envs (Pendulum), async for slow ones (EvoGym)")
    p.add_argument("--seed", type=int, default=31)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--entropy-beta", type=float, default=0.0)
    p.add_argument("--clip-epsilon", type=float, default=0.2)
    p.add_argument("--ppo-epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--update-freq", type=int, default=2048, help="transitions per PPO update")
    p.add_argument("--norm-reward", action="store_true", help="optimisation: reward normalisation")
    p.add_argument("--out", required=True, help="output folder, e.g. runs/pendulum_norm_s31")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    torch.set_num_threads(1)  # tiny network; leave the cores to the env processes

    agent, rewards, *_ = train_agent(
        total_steps=args.steps, update_freq=args.update_freq, env_id=args.env, lr=args.lr,
        gamma=args.gamma, gae_lambda=args.gae_lambda, entropy_beta=args.entropy_beta,
        clip_epsilon=args.clip_epsilon, ppo_epochs=args.ppo_epochs, batch_size=args.batch_size,
        seed=args.seed, norm_reward=args.norm_reward, n_envs=args.n_envs,
        vectorization=args.vectorization, out_dir=out)
    print(f"Final avg (last 100 episodes): {np.mean(rewards[-100:]):.2f} | saved to {out}/")
