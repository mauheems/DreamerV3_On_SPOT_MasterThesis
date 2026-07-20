# DreamerV3 SPOT Implementation — Technical Overview

This folder contains the analysis notebooks, configuration files, and helper scripts for the SPOT robot navigation thesis.

The core **training code** lives in the informed-dreamer fork: [mauheems/Master_thesis_DAIC_code](https://github.com/mauheems/Master_thesis_DAIC_code) (branch `noobs-dataset`).

---

## Structure

```
dreamer_SPOT_implementation/
├── informed-dreamer/         # Fork of Informed Dreamer with SPOT modifications
├── notebooks/                # Jupyter notebooks for analysis and evaluation
├── scripts/                  # Standalone analysis scripts
├── configs/                  # Auxiliary training configs (latent dim, horizon, etc.)
├── results/                  # Video_1.gif and generated figures
└── requirements.txt          # Python dependencies
```

---

## What Was Modified in Informed Dreamer

All changes are on the `noobs-dataset` branch of [mauheems/Master_thesis_DAIC_code](https://github.com/mauheems/Master_thesis_DAIC_code), on top of the original `a4ac0e5` import commit.

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

## Scripts

### `analyze_replay_imagine.py`

Analyses replay sampling statistics relative to episode boundaries and imagined rollout horizons.

```bash
python scripts/analyze_replay_imagine.py \
    --data_dir /path/to/hdf5_episodes \
    --batch_length 12 \
    --samples 2000
```

Reports what fraction of sampled windows contain a real terminal within the imagined horizon — useful for understanding whether the agent's imagination reaches meaningful episode boundaries.

---

## Notebooks

See [notebooks/README.md](notebooks/README.md) for individual notebook descriptions.

To run: point each notebook's `checkpoint_dir` and `data_dir` variables to your trained checkpoints and HDF5 data.
