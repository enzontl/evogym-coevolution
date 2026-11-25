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
