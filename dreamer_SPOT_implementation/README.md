# Dreamer-4: Transformer-Based World Model for Robotics

## Overview
Dreamer-4 is a transformer-based world model implementation for robot learning and control. It combines:
- A causal tokenizer for efficient image compression
- A transformer world model for predicting future states
- Actor-critic policy optimization in imagined rollouts
- Support for offline and synthetic data training

## Project Structure
- `dreamer4/`: Core model implementations
- `dataset/`: Data loading and preprocessing
- `training/`: Training scripts and pipelines
- `configs/`: Configuration files
- `notebooks/`: Experiments and analysis

## Installation
```bash
pip install -r requirements.txt
```

## Usage
See individual training scripts in the `training/` directory for usage examples.
