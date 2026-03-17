#!/usr/bin/env python3
"""
Check consistency of all HDF5 files in processed_data_with_rewards directory.
Verifies that all files have proper 4D actions and consistent dimensions.
"""

import h5py
import numpy as np
from pathlib import Path
import sys
from collections import defaultdict

def check_h5_file(h5_path):
    """Check a single H5 file for consistency issues."""
    issues = []
    data_info = {}
    
    try:
        with h5py.File(h5_path, 'r') as f:
            # Check required datasets
            required = ['observations/state', 'observations/velocities', 'actions']
            for key in required:
                if key not in f:
                    issues.append(f"Missing {key}")
            
            if not issues:
                # Get shapes
                states = f['observations/state'][:]
                velocities = f['observations/velocities'][:]
                actions = f['actions'][:]
                
                data_info = {
                    'states_shape': states.shape,
                    'velocities_shape': velocities.shape,
                    'actions_shape': actions.shape,
                    'states_dtype': str(states.dtype),
                    'velocities_dtype': str(velocities.dtype),
                    'actions_dtype': str(actions.dtype),
                }
                
                # Check for consistency
                n_steps = states.shape[0]
                
                if velocities.shape[0] != n_steps:
                    issues.append(f"Velocities length mismatch: {velocities.shape[0]} != {n_steps}")
                
                if actions.shape[0] != n_steps:
                    issues.append(f"Actions length mismatch: {actions.shape[0]} != {n_steps}")
                
                # Check actions dimension (should be 4)
                if len(actions.shape) != 2 or actions.shape[1] != 4:
                    issues.append(f"Actions should be 4D (nx4), got {actions.shape}")
                
                # Check velocities dimension (should be 6)
                if len(velocities.shape) != 2 or velocities.shape[1] != 6:
                    issues.append(f"Velocities should be 6D (nx6), got {velocities.shape}")
                
                # Check states dimension (should be 7)
                if len(states.shape) != 2 or states.shape[1] != 7:
                    issues.append(f"States should be 7D (nx7), got {states.shape}")
                
                # Check for NaN values
                if np.any(np.isnan(states)):
                    issues.append(f"States contain NaN values")
                if np.any(np.isnan(velocities)):
                    issues.append(f"Velocities contain NaN values")
                if np.any(np.isnan(actions)):
                    issues.append(f"Actions contain NaN values")
                
                # Check if rewards exist
                if 'rewards' in f:
                    rewards = f['rewards'][:]
                    if rewards.shape[0] != n_steps:
                        issues.append(f"Rewards length mismatch: {rewards.shape[0]} != {n_steps}")
    
    except Exception as e:
        issues.append(f"Error reading file: {e}")
    
    return issues, data_info

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Check H5 file consistency")
    parser.add_argument('--dir', type=str, default='/media/external_drive/processed_data_with_rewards',
                        help="Directory containing H5 files")
    args = parser.parse_args()
    
    h5_dir = Path(args.dir)
    
    if not h5_dir.exists():
        print(f"Error: Directory not found: {h5_dir}")
        sys.exit(1)
    
    # Find all .h5 files
    h5_files = sorted(h5_dir.glob('*.h5'))
    
    if not h5_files:
        print(f"No .h5 files found in {h5_dir}")
        sys.exit(1)
    
    print(f"Found {len(h5_files)} HDF5 files in {h5_dir}\n")
    print("=" * 80)
    
    # Track statistics
    passed = 0
    failed = 0
    shape_stats = defaultdict(int)
    action_issues = 0
    
    # Check each file
    for i, h5_file in enumerate(h5_files, 1):
        issues, data_info = check_h5_file(h5_file)
        
        status = "✓ PASS" if not issues else "✗ FAIL"
        print(f"\n[{i}/{len(h5_files)}] {status}: {h5_file.name}")
        
        if data_info:
            print(f"  States:     {data_info['states_shape']} ({data_info['states_dtype']})")
            print(f"  Velocities: {data_info['velocities_shape']} ({data_info['velocities_dtype']})")
            print(f"  Actions:    {data_info['actions_shape']} ({data_info['actions_dtype']})")
            
            # Track shape stats
            action_shape_key = data_info['actions_shape']
            shape_stats[action_shape_key] += 1
            
            # Check if actions are 4D
            if len(data_info['actions_shape']) == 2 and data_info['actions_shape'][1] != 4:
                action_issues += 1
        
        if issues:
            print(f"  Issues:")
            for issue in issues:
                print(f"    - {issue}")
            failed += 1
        else:
            passed += 1
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print(f"  ✓ Passed: {passed}/{len(h5_files)}")
    print(f"  ✗ Failed: {failed}/{len(h5_files)}")
    
    print(f"\nAction shape distribution:")
    for shape, count in sorted(shape_stats.items()):
        status = "✓" if (len(shape) == 2 and shape[1] == 4) else "✗"
        print(f"  {status} {shape}: {count} files")
    
    if action_issues > 0:
        print(f"\n⚠ WARNING: {action_issues} files have non-4D actions!")
    
    if failed > 0:
        print(f"\n⚠ WARNING: {failed} files have issues!")
        sys.exit(1)
    else:
        print(f"\n✓ All files are consistent!")
        sys.exit(0)

if __name__ == '__main__':
    main()
