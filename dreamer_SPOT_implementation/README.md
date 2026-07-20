# DreamerV3 SPOT Implementation — Technical Overview

This folder contains the complete training codebase, analysis notebooks, configurations, and helper scripts for the SPOT robot navigation thesis.

**Base Framework:** [DreamerV3](https://github.com/danijar/dreamerv3) (Hafner et al., 2023) → extended by [Informed Dreamer](https://github.com/gaspardlambrechts/informed-dreamer) (Lambrechts et al., 2024) → extended by me for SPOT.  
**Note:** This repo includes the full Informed Dreamer codebase (which itself includes DreamerV3 code) with my modifications on top. See [LICENSE.md](../LICENSE.md) for the full attribution chain.

---

## Structure

```
dreamer_SPOT_implementation/
├── informed-dreamer/         # Base framework (Lambrechts et al.) + my SPOT modifications
├── Video_1.gif               # Policy rollout demo GIF
├── notebooks/                # My analysis & evaluation notebooks
├── scripts/                  # My standalone analysis scripts
├── configs/                  # My training configurations
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## What I Modified in Informed Dreamer

All my changes are in `informed-dreamer/` and described below. This integrated version is included in this repo for complete portability.

### New Files

| File | Lines | Purpose |
|------|-------|---------|
| `dreamerv3/embodied/envs/spot_live.py` | ~425 | Live closed-loop deployment on the physical SPOT robot via Boston Dynamics SDK |
| `dreamerv3/embodied/run/train_offline.py` | ~185 | Offline training loop — iterates over pre-collected HDF5 episodes, no env interaction |
| `dreamerv3/train_online.py` | ~342 | Online training entrypoint with SPOT environment, W&B logging, SLURM support |
| `validate_episodes.py` | ~45 | Episode validation tool for checking HDF5 data integrity before training |

### Modified Files

| File | Description |
|------|-------------|
| `dreamerv3/embodied/envs/spot.py` | SPOT gym environment: observation space (11 scalars for NoObs, + images for visual), action space (velocity commands), reward function, reset logic |
| `dreamerv3/agent.py` | SPOT-specific observation preprocessing, reward normalisation, NoObs observation routing |
| `dreamerv3/jaxagent.py` | Deployment-mode inference support (pure-function wrapping for offline rollout) |
| `dreamerv3/configs.yaml` | Added `spot_noobs`, `spot_simple`, `spot_informed` config blocks with tuned hyperparameters |
| `dreamerv3/train.py` | Offline data path, HDF5 episode loading, SPOT dataset integration |
| `dreamerv3/embodied/run/train.py` | Minor edits for SPOT compatibility |

---

## NoObs vs Visual Architecture

```
NoObs:
  obs = [vx, vy, yaw_rate, imu×6, joint_angles×12]  →  Encoder (MLP)  →  RSSM  →  Actor/Critic

Visual:
  obs = [state] + camera_rgb + terrain_map  →  Encoder (CNN + MLP)  →  RSSM  →  Actor/Critic
```

The NoObs encoder is a small MLP (no CNN), making it fast to train on CPU. The visual encoder uses a CNN for image/terrain inputs combined with an MLP for the scalar observations.

---

## Notebooks

See [notebooks/README.md](notebooks/README.md) for individual notebook descriptions.

To run: point each notebook's `checkpoint_dir` and `data_dir` variables to your trained checkpoints and HDF5 data.
