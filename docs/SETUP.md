# Setup Instructions

## Prerequisites

- Python 3.10 (matching the training container)
- JAX-compatible hardware (NVIDIA GPU recommended for visual variant; CPU works for NoObs)
- ROS2 Humble (only needed for live deployment on SPOT — not required for training)

## 1. Clone the Repository

```bash
git clone https://github.com/mauheems/DreamerV3_On_SPOT_MasterThesis.git
cd DreamerV3_On_SPOT_MasterThesis
```

The repository includes the complete informed-dreamer codebase with all SPOT modifications integrated.

## 2. Python Environment

Create a Python 3.10 environment (the training code is pinned to 3.10):

```bash
# With conda
conda create -n dreamer python=3.10
conda activate dreamer

# Or with pyenv
pyenv install 3.10.13
pyenv virtualenv 3.10.13 dreamer-env
pyenv activate dreamer-env
```

## 3. Install Dependencies

```bash
cd dreamer_SPOT_implementation/informed-dreamer

# Install the dreamerv3 package and dependencies
pip install -e ./dreamerv3

# Install JAX (GPU version — adjust cuda version as needed)
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# For CPU-only (NoObs variant, faster to get started)
pip install --upgrade "jax[cpu]"
```

## 4. Verify Installation

```bash
python -c "import dreamerv3; import jax; print('JAX devices:', jax.devices())"
```

## Running Training

### NoObs (Recommended starting point — no GPU required)

```bash
cd dreamer_SPOT_implementation/informed-dreamer
python dreamerv3/train_offline.py \
    --configs spot_noobs \
    --logdir /path/to/logdir \
    --offline_dir /path/to/hdf5_episodes
```

### Visual variant (requires GPU)

```bash
python dreamerv3/train_offline.py \
    --configs spot_informed \
    --logdir /path/to/logdir \
    --offline_dir /path/to/hdf5_episodes
```

Available configs (in `dreamerv3/configs.yaml`):
- `spot_noobs` — proprioceptive state only, no camera
- `spot_simple` — state-only baseline (11 scalars)
- `spot_informed` — full visual + terrain map variant

## Data

Pre-collected episodes are not included in this repo (too large). For information on data collection, see the thesis paper (Chapter 3). To collect your own robot data or request access to processed HDF5 files, contact the author.

Validate episodes before training:

```bash
python validate_episodes.py --data_dir /path/to/hdf5_episodes
```

## Notebooks

```bash
pip install jupyter
cd dreamer_SPOT_implementation
jupyter notebook notebooks/
```

Update the `checkpoint_dir` and `data_dir` variables at the top of each notebook to point to your trained checkpoints and data.

## Docker (Optional)

A Dockerfile for the training environment is available at `dreamer_SPOT_implementation/informed-dreamer/dreamerv3/Dockerfile`. Useful for running on a cluster (e.g., SLURM/DAIC).

```bash
cd dreamer_SPOT_implementation/informed-dreamer/dreamerv3
docker build -t dreamer-spot .
docker run --gpus all -v /path/to/data:/data dreamer-spot \
    python train.py --configs spot_noobs --offline_dir /data
```
