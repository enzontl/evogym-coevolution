# EvoGym co-evolution

Deep Reinforcement Learning course project.

**Question:** starting from random shapeless bodies ("blobs"), does evolution find a soft robot that walks better than a hand-designed one, with the same brain (PPO)? And does the optimal shape depend on the terrain?

- Environment: [Evolution Gym](https://github.com/EvolutionGym/evogym) (MIT, NeurIPS 2021), main task `Walker-v0` (flat ground)
- Controller: PPO with continuous actions
- Body: genetic algorithm over 5×5 voxel robots, starting from random bodies

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
| 5 | Body evolution | Genetic algorithm over 5×5 bodies starting from random blobs (~8 robots × 8–10 generations), trained with the best PPO; compared to random bodies (generation 0) and to the hand-designed robot |
| 6 | Terrains (extension) | Same evolution on other EvoGym terrains: stairs (`UpStepper-v0`), obstacles (`ObstacleTraverser-v0`), possibly `DownStepper-v0` or `CaveCrawler-v0`; specialisation matrix (each terrain's champion tested on the other terrains) |
| 7 | Visuals | Custom renderer: chronophotographs of the champions, evolution frieze (best body per generation, from blob to final body), side-by-side race, GIFs, family tree |
| 8 | Presentation | Comparison table, what worked and what did not |

The hand-designed robot is only used to tune PPO on a fixed problem and as a human reference to beat; it is never part of the evolution.
