#!/usr/bin/env python3
"""Validate SPOT HDF5 episodes for training compatibility."""

import h5py
import glob
import sys
from pathlib import Path

data_dir = "/tudelft.net/staff-umbrella/openbots/mjheemskerk/spot_data/processed_data_with_rewards"
files = sorted(glob.glob(f"{data_dir}/*.hdf5"))

print(f"Checking {len(files)} episodes...\n")

bad_episodes = []
expected_action_dim = 4

for i, filepath in enumerate(files):
    try:
        with h5py.File(filepath, 'r') as hf:
            action_shape = hf['actions'].shape
            action_dim = action_shape[-1] if len(action_shape) > 1 else action_shape[0]
            
            if action_dim != expected_action_dim:
                name = Path(filepath).name
                bad_episodes.append({
                    'file': name,
                    'action_dim': action_dim,
                    'num_steps': action_shape[0] if len(action_shape) > 0 else 0
                })
                print(f"[{i+1}/{len(files)}] ❌ {name}: {action_dim}D actions (expected {expected_action_dim}D), {action_shape[0]} steps")
            else:
                if (i + 1) % 50 == 0:
                    print(f"[{i+1}/{len(files)}] ✓ OK", end="\r")
    except Exception as e:
        print(f"[{i+1}/{len(files)}] ⚠️  Error reading {Path(filepath).name}: {e}")

print(f"\n\n{'='*60}")
if bad_episodes:
    print(f"Found {len(bad_episodes)} BAD episodes:")
    for ep in bad_episodes:
        print(f"  - {ep['file']}: {ep['action_dim']}D actions")
    print(f"\nThese episodes should be removed from the dataset.")
else:
    print("✓ All episodes have correct action dimensions!")
print(f"{'='*60}")
