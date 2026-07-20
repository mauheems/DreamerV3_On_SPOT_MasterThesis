# Autonomous Navigation for Boston Dynamics SPOT using DreamerV3

> Master's thesis — Maurits Heemskerk, 2026  
> [📄 Read the thesis](Master-thesis-Maurits-Heemskerk.pdf)

---

![Policy rollout in imagination](dreamer_SPOT_implementation/Video_1.gif)

*Imagined trajectory rollout from the trained DreamerV3 actor-critic policy.*

---

## Overview

This thesis applies **model-based reinforcement learning** to autonomous navigation on the [Boston Dynamics SPOT](https://www.bostondynamics.com/products/spot) quadruped robot. The agent learns a world model of the robot's dynamics from offline data and uses it to train a navigation policy entirely in imagination — without interacting with the physical environment during training.

The work builds on **[Informed Dreamer](https://github.com/gaspardlambrechts/informed-dreamer)** (Lambrechts et al., 2024), an extension of DreamerV3 for partially observable environments. On top of this framework, I contributed:

- A **SPOT robot environment wrapper** for online and offline training
- **Offline training pipeline** from real robot data (rosbag → HDF5 → training)
- **Live deployment interface** for closed-loop policy execution on SPOT
- **NoObs variant** — camera-free navigation using proprioceptive state only
- Custom **reward functions** and **training configurations** for goal-directed walking

---

## Key Results

| Variant | Description | Result |
|---------|-------------|--------|
| **NoObs** | Proprioception only (velocity, IMU, joint angles) | ✅ Converges to goal-directed navigation |
| **Visual** | + RGB camera + terrain map | 🔬 Challenged by partial observability |

The NoObs variant successfully learns goal-directed walking behaviour from offline demonstrations, demonstrating that meaningful locomotion policies can be acquired without camera input. See the [paper](Master-thesis-Maurits-Heemskerk.pdf) for full quantitative results.

---

## Repository Structure

```
.
├── Master-thesis-Maurits-Heemskerk.pdf   # The thesis paper
├── dreamer_SPOT_implementation/
│   ├── informed-dreamer/                 # Base codebase (see Attribution below) + my SPOT modifications
│   ├── Video_1.gif                       # Policy rollout demo
│   ├── notebooks/                        # My analysis & evaluation notebooks
│   ├── scripts/                          # My standalone analysis scripts
│   ├── configs/                          # My training configuration files
│   ├── requirements.txt                  # Dependencies
│   └── README.md                         # Technical overview of my changes
├── docs/
│   └── SETUP.md                          # Installation & training instructions
└── LICENSE.md                             # Full attribution for all code
```

---

## Attribution

> **Important:** This repository contains third-party code that is **not my own work**. The attribution chain is:

### 1. DreamerV3 — Hafner et al. (2023)
The core world model framework. MIT License, Copyright © 2023 Danijar Hafner.
- Paper: [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104)
- Repo: [danijar/dreamerv3](https://github.com/danijar/dreamerv3)

### 2. Informed Dreamer — Lambrechts, Bolland & Ernst (2024)
Extends DreamerV3 for partially observable environments (POMDPs) using auxiliary information. **Includes the full DreamerV3 codebase** with modifications. MIT License.
- Paper: [Informed POMDP: Leveraging Additional Information in Model-Based RL](https://rlj.cs.umass.edu/2024/papers/Paper68.html)
- Repo: [gaspardlambrechts/informed-dreamer](https://github.com/gaspardlambrechts/informed-dreamer)

### 3. My SPOT Adaptations — Maurits Heemskerk (2026)
Extends Informed Dreamer with a physical robot integration for the Boston Dynamics SPOT. MIT License.
- All new files and modifications are listed below and in [dreamer_SPOT_implementation/README.md](dreamer_SPOT_implementation/README.md)

See [LICENSE.md](LICENSE.md) for the full license details.
|------|--------|-------------|
| `dreamerv3/embodied/envs/spot.py` | Modified | SPOT gym environment wrapper — observation/action space, reward, reset |
| `dreamerv3/embodied/envs/spot_live.py` | **New** | Live deployment interface for closed-loop control on the physical robot |
| `dreamerv3/embodied/run/train_offline.py` | **New** | Offline training loop — trains world model from pre-collected HDF5 episodes |
| `dreamerv3/train_online.py` | **New** | Online training entrypoint with SPOT-specific config and logging |
| `dreamerv3/agent.py` | Modified | Reward function integration, SPOT-specific observation processing |
| `dreamerv3/jaxagent.py` | Modified | Deployment-mode inference support |
| `dreamerv3/configs.yaml` | Modified | SPOT training configs: `spot_noobs`, `spot_simple`, `spot_informed` variants |
| `dreamerv3/train.py` | Modified | Entrypoint edits for SPOT datasets |
| `validate_episodes.py` | **New** | Script to validate and inspect collected HDF5 episodes before training |

---

## NoObs vs Visual Observations

A key design choice explored in this thesis is whether camera input is necessary for navigation:

| | **NoObs** | **Visual** |
|-|-----------|------------|
| **Observations** | Velocity (vx, vy, yaw) + IMU + joint angles | + RGB camera + terrain map |
| **State dimension** | ~11 scalars | + image tensors |
| **Training speed** | Fast (CPU/small GPU) | Slower (requires GPU) |
| **Deployment** | Lightweight, no vision pipeline | Requires camera stream |
| **Result** | Converges reliably | More challenging due to partial observability |

The NoObs variant isolates whether the world model can learn useful dynamics without visual input. It succeeded and is the main result of the thesis.

---

## Notebooks

All notebooks are in `dreamer_SPOT_implementation/notebooks/`. Outputs have been stripped — run them by pointing to your checkpoint/data directories.

| Notebook | Purpose |
|----------|---------|
| `exploration_data_collected.ipynb` | Inspect raw episode data: sensor readings, trajectory shape, data quality |
| `reward_computation.ipynb` | Visualise reward signal per timestep; sensor observations alongside computed rewards |
| `training_results_analysis.ipynb` | Compare training runs: config diffs, loss curves, metric grids across experiments |
| `world_model_reconstruction.ipynb` | Evaluate posterior reconstruction — does the world model reconstruct what it observed? (Visual) |
| `results_latentspace.ipynb` | t-SNE of posterior latent states coloured by velocity/distance/reward; cross-checkpoint interpretability |
| `agent_critic_evaluation.ipynb` | Policy evaluation in imagination: does the actor control the agent meaningfully? (Visual) |
| `agent_critic_evaluation_noobs.ipynb` | Same as above, but for the **NoObs** variant — no camera input |
| `deployment_evaluation.ipynb` | Load `.npz` deployment recordings from the physical robot and compute evaluation metrics |

> **NoObs notebooks** (`*_noobs.ipynb`) run the policy using only proprioceptive state, without camera or terrain observations.

---

## Quick Start

See [docs/SETUP.md](docs/SETUP.md) for full installation instructions.

```bash
# 1. Clone this repo
git clone https://github.com/mauheems/Master-Thesis.git
cd Master-Thesis

# 2. Install dependencies
cd dreamer_SPOT_implementation/informed-dreamer
pip install -e ./dreamerv3

# 3. Train (offline, NoObs)
python dreamerv3/train.py \
    --configs spot_noobs \
    --logdir /path/to/logdir \
    --offline_dir /path/to/hdf5_episodes
```

---

## About Informed Dreamer Fork

The original Informed Dreamer repository is maintained at: [gaspardlambrechts/informed-dreamer](https://github.com/gaspardlambrechts/informed-dreamer)

I also maintain a fork at [mauheems/Master_thesis_DAIC_code](https://github.com/mauheems/Master_thesis_DAIC_code) (branch: `noobs-dataset`) with my SPOT modifications for reference.

---

## Contributions Summary

**See [dreamer_SPOT_implementation/README.md](dreamer_SPOT_implementation/README.md) for a detailed technical breakdown of which files in informed-dreamer were modified.**

---

## Citation & Attribution

This work extends:

```bibtex
@article{lambrechts2024informed,
    title={Informed {POMDP}: {L}everaging Additional Information in Model-Based {RL}},
    author={Lambrechts, Gaspard and Bolland, Adrien and Ernst, Damien},
    journal={Reinforcement Learning Journal},
    volume={1}, issue={1}, year={2024}
}

@article{hafner2023dreamerv3,
    title={Mastering Diverse Domains through World Models},
    author={Hafner, Danijar and Pasukonis, Jurgis and Ba, Jimmy and Lillicrap, Timothy},
    journal={arXiv preprint arXiv:2301.04104}, year={2023}
}
```

---