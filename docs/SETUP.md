# Setup Instructions

## Prerequisites

- Python 3.10 (matching the training container)
- JAX-compatible hardware (NVIDIA GPU recommended for visual variant; CPU works for NoObs)
- ROS2 Humble (only needed for live deployment on SPOT — not required for training)

## 1. Clone the Repository

```bash
git clone https://github.com/mauheems/Master-Thesis.git
cd Master-Thesis
```

## 2. Clone the Informed Dreamer Fork

The training code lives in a separate repo (fork of Informed Dreamer with SPOT modifications):

```bash
git clone -b noobs-dataset https://github.com/mauheems/Master_thesis_DAIC_code.git \
    dreamer_SPOT_implementation/informed-dreamer
```

## 3. Python Environment

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

## 4. Install Dependencies

```bash
cd dreamer_SPOT_implementation/informed-dreamer

# Install the dreamerv3 package and dependencies
pip install -e ./dreamerv3

# Install JAX (GPU version — adjust cuda version as needed)
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# For CPU-only (NoObs variant, faster to get started)
pip install --upgrade "jax[cpu]"
```

## 5. Verify Installation

```bash
python -c "import dreamerv3; import jax; print('JAX devices:', jax.devices())"
```

## Running Training

### NoObs (Recommended starting point — no GPU required)

```bash
cd dreamer_SPOT_implementation/informed-dreamer
python dreamerv3/train.py \
    --configs spot_noobs \
    --logdir /path/to/logdir \
    --offline_dir /path/to/hdf5_episodes
```

### Visual variant (requires GPU)

```bash
python dreamerv3/train.py \
    --configs spot_informed \
    --logdir /path/to/logdir \
    --offline_dir /path/to/hdf5_episodes
```

Available configs (in `dreamerv3/configs.yaml`):
- `spot_noobs` — proprioceptive state only, no camera
- `spot_simple` — state-only baseline (11 scalars)
- `spot_informed` — full visual + terrain map variant

## Data

Pre-collected episodes are not included in this repo (too large). See [DATA_COLLECTION.md](DATA_COLLECTION.md) for how to collect your own, or contact the author for access to the processed HDF5 files.

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
