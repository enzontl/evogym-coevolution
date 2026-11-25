# EvoGym co-evolution

Deep Reinforcement Learning course project.

**Question:** does a soft robot whose body evolves walk better than a hand-designed robot, with the same brain (PPO)?

- Environment: [Evolution Gym](https://github.com/EvolutionGym/evogym) (MIT, NeurIPS 2021), task `Walker-v0`
- Controller: PPO with continuous actions
- Body: genetic algorithm over 5×5 voxel robots

## Setup

EvoGym 2.0 requires **Python 3.10**.

```bash
uv venv -p 3.10 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Check the install:

```bash
python -c "import evogym.envs; print('ok')"
```

## Plan

| # | Step | Content |
|---|---|---|
| 1 | Continuous PPO | Adapt the course PPO to continuous actions (Gaussian policy), sanity-check on Pendulum |
| 2 | Baseline | PPO with course hyperparameters on a hand-designed robot; learning curves + GIF |
| 3 | Analysis | Identify what limits the baseline (slow, unstable, robot not moving…) |
| 4 | PPO optimisation | One change at a time, 2–3 seeds each: observation/reward normalisation, separate actor and critic, learning rate, entropy |
| 5 | Body evolution | Genetic algorithm over 5×5 bodies (~8 robots × 8–10 generations) trained with the best PPO; compare the best body to the step-2 robot |
| 6 | Presentation | Evolution GIFs, comparison table, what worked and what did not |
