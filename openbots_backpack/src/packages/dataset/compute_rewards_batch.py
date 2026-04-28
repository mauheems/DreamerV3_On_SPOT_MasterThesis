#!/usr/bin/env python3
"""
Pre-compute and save rewards for all HDF5 episodes in the dataset.
This processes all episodes offline before training.
"""

import h5py
import numpy as np
from pathlib import Path
import sys

# Centralized reward parameters (keep in sync with notebook REWARD_PARAMS)
REWARD_PARAMS = {
    'goal_threshold': 3.0,  # Gaussian sigma: shaped reward instead of one-shot spike
    'goal_reward_base': 0.5,  # Per-step max reward at goal (lower since accumulated every step)
    'goal_distance_reference': 5.0,
    'distance_reward_scale': 5.0,
    'jerk_scale': 0.05,  # Increased from 0.01 to penalize jerky commands more heavily
    'instability_scale': 0.02,
    'time_penalty': 0.01,
    'velocity_error_scale': 0.04,
}

def compute_rewards_for_episode(states, actions, velocities, dt=0.1):
    """
    Compute rewards for an episode using multiple components.
    Matches the notebook RewardComputer parameters.
    Returns: reward array of shape (T,)
    """
    T = len(states)
    
    # Reward parameters (centralized)
    goal_threshold = REWARD_PARAMS['goal_threshold']
    goal_reward_base = REWARD_PARAMS['goal_reward_base']
    goal_distance_reference = REWARD_PARAMS['goal_distance_reference']
    distance_reward_scale = REWARD_PARAMS['distance_reward_scale']
    jerk_scale = REWARD_PARAMS['jerk_scale']
    instability_scale = REWARD_PARAMS['instability_scale']
    time_penalty = REWARD_PARAMS['time_penalty']
    velocity_error_scale = REWARD_PARAMS['velocity_error_scale']
    
    # Goal is final position
    goal_position = states[-1, :3]
    
    # === Component 1: Distance Decrease Reward ===
    positions = states[:, :3]
    distances = np.linalg.norm(positions - goal_position, axis=1)
    initial_distance = distances[0]
    if initial_distance < 1e-6:
        initial_distance = 1.0
    
    distance_changes = np.diff(distances)
    distance_rewards = -distance_reward_scale * (distance_changes / initial_distance)
    distance_rewards = np.pad(distance_rewards, (1, 0), mode='constant', constant_values=0)
    
    # === Component 2: Goal Reaching Reward (Gaussian-shaped per-step) ===
    # Replaces one-time spike with continuous dense signal:
    #   reward(t) = goal_reward_base * exp(-d(t)^2 / (2 * goal_threshold^2))
    # Benefits: every timestep near goal yields reward (robot incentivized to stay),
    # short training sequences still get useful signal when passing through goal vicinity
    sigma = goal_threshold
    goal_rewards = goal_reward_base * np.exp(-(distances ** 2) / (2 * sigma ** 2))
    
    # === Component 3: Action Smoothness Penalty ===
    if actions is not None and len(actions) > 1:
        action_diffs = np.diff(actions, axis=0)
        action_magnitudes = np.linalg.norm(action_diffs, axis=1)
        # Linear penalty: large jerks cost proportionally more
        jerk_penalties = -jerk_scale * action_magnitudes
        jerk_penalties = np.pad(jerk_penalties, (1, 0), mode='constant', constant_values=0)
    else:
        jerk_penalties = np.zeros(T)
    
    # === Component 4: Instability Penalty (from quaternion changes) ===
    if states.shape[1] >= 7:
        quats = states[:, 3:7]
        quat_diffs = np.diff(quats, axis=0)
        quat_change_magnitude = np.linalg.norm(quat_diffs, axis=1)
        instability_penalties = -instability_scale * quat_change_magnitude
        instability_penalties = np.pad(instability_penalties, (1, 0), mode='constant', constant_values=0)
    else:
        instability_penalties = np.zeros(T)
    
    # === Component 5: Velocity Alignment Penalty ===
    if velocities is not None and actions is not None:
        # Compensate for known cmd->odom lag (1 sample) by shifting actions forward
        lag = 1
        # Actions (4D): [vx_cmd, vy_cmd, vyaw_cmd, gait] → indices 0, 1, 2
        # Velocities (6D): [vx, vy, vz, wx, wy, wz]       → indices 0, 1, 5
        # We align vx, vy, and yaw rate (wz). Gait is not a velocity comparison.
        act_sel_idx = [0, 1, 2]  # vx_cmd, vy_cmd, vyaw_cmd
        vel_sel_idx = [0, 1, 5]  # vx_obs, vy_obs, wz_obs

        # Create shifted actions with NaN padding for edges
        shifted_actions = np.full_like(actions, np.nan)
        if lag == 0:
            shifted_actions = actions
        else:
            shifted_actions[lag:] = actions[:-lag]

        # Compute sum of squared errors only on selected axes where shifted action exists
        act_sel = shifted_actions[:, act_sel_idx]
        vel_sel = velocities[:, vel_sel_idx]
        valid_mask = ~np.isnan(act_sel).any(axis=1)

        velocity_alignment_penalties = np.zeros(T, dtype=float)
        if np.any(valid_mask):
            diffs = vel_sel[valid_mask] - act_sel[valid_mask]
            sum_sq = np.sum(diffs ** 2, axis=1)
            velocity_alignment_penalties[valid_mask] = -velocity_error_scale * sum_sq
    else:
        velocity_alignment_penalties = np.zeros(T)
    
    # === Component 6: Time Penalty ===
    time_penalties = np.full(T, -time_penalty)
    
    # === Combine all components ===
    rewards = (goal_rewards + distance_rewards + jerk_penalties + 
               instability_penalties + velocity_alignment_penalties + time_penalties)
    
    return rewards.astype(np.float32)


def process_episode_file(input_path, output_path):
    """
    Compute rewards for an episode and save to new HDF5 file with rewards.
    Copies input file to output and adds rewards dataset.
    """
    try:
        # Read input file and copy to output
        with h5py.File(input_path, 'r') as f_in:
            # Load required data
            if 'observations/state' not in f_in or 'observations/velocities' not in f_in:
                print(f"  ✗ Missing state or velocities in {input_path.name}")
                return False
            
            states = f_in['observations/state'][:]
            velocities = f_in['observations/velocities'][:]
            actions = f_in['actions'][:] if 'actions' in f_in else None
            
            # Compute rewards
            rewards = compute_rewards_for_episode(states, actions, velocities)
            
            # Copy all data from input to output and add rewards
            with h5py.File(output_path, 'w') as f_out:
                # Copy all datasets from input
                def copy_dataset(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        f_out.create_dataset(name, data=obj[:], compression='gzip')
                    elif isinstance(obj, h5py.Group):
                        f_out.create_group(name)
                
                f_in.visititems(copy_dataset)
                
                # Copy attributes
                for attr_name, attr_val in f_in.attrs.items():
                    f_out.attrs[attr_name] = attr_val
                
                # Add computed rewards
                f_out.create_dataset('rewards', data=rewards, dtype=np.float32)
                f_out.attrs['rewards_computed'] = True
                f_out.attrs['reward_components'] = 'goal_scaled, distance, jerk, instability, velocity_alignment, time'
                # Record applied command->odom lag (samples) used when computing rewards
                f_out.attrs['cmd_odom_lag_samples'] = 1
            
            print(f"  ✓ Computed {len(rewards)} rewards: range=[{rewards.min():.4f}, {rewards.max():.4f}], sum={rewards.sum():.4f}")
            return True
            
    except Exception as e:
        print(f"  ✗ Error processing {input_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Pre-compute rewards for HDF5 episodes")
    parser.add_argument('--harddrive', type=str, default=None, 
                        help="Path to external harddrive processed_data directory")
    parser.add_argument('--input-dir', type=str, default=None,
                        help="Input directory containing HDF5 files")
    parser.add_argument('--output-dir', type=str, default=None,
                        help="Output directory for files with rewards (default: processed_data_with_rewards)")
    args = parser.parse_args()
    
    # Determine input directory
    if args.harddrive:
        # Check for processed_data_NoObs first (no-obstacle dataset), then fall back to processed_data
        noobs_dir = Path(args.harddrive) / 'processed_data_NoObs'
        default_dir = Path(args.harddrive) / 'processed_data'
        
        if noobs_dir.exists():
            input_dir = noobs_dir
            output_suffix = '_NoObs_with_rewards'
            print(f"Using external harddrive (NoObs): {input_dir}")
        elif default_dir.exists():
            input_dir = default_dir
            output_suffix = '_with_rewards'
            print(f"Using external harddrive: {input_dir}")
        else:
            print(f"Error: Neither {noobs_dir} nor {default_dir} found")
            sys.exit(1)
    elif args.input_dir:
        input_dir = Path(args.input_dir)
        output_suffix = '_with_rewards'
    else:
        # Default: look in parent of recorded_data
        input_dir = Path("/home/maurits-heemskerk/Documents/Uni/Master_Thesis/openbots_backpack/src/packages/dataset/processed_data")
        output_suffix = '_with_rewards'
    
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Create output with appropriate suffix (e.g., processed_data_NoObs_with_rewards or processed_data_with_rewards)
        output_dir = input_dir.parent / f'processed_data{output_suffix}'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all HDF5 files in input directory
    episode_files = sorted(input_dir.glob("**/*.h5"), key=lambda p: p.stat().st_mtime)
    
    if not episode_files:
        print(f"No .h5 files found in {input_dir}")
        sys.exit(1)
    
    print(f"Found {len(episode_files)} episodes in: {input_dir}")
    print(f"Output directory: {output_dir}")
    print("="*60)
    
    # Process each file
    processed = 0
    skipped = 0
    failed = 0
    
    for idx, input_path in enumerate(episode_files, 1):
        output_path = output_dir / input_path.name
        
        # Skip if already processed
        if output_path.exists():
            print(f"[{idx}/{len(episode_files)}] ⊘ {input_path.parent.name}/{input_path.name} (already processed)")
            skipped += 1
            continue
        
        print(f"[{idx}/{len(episode_files)}] {input_path.parent.name}/{input_path.name}")
        
        if process_episode_file(input_path, output_path):
            processed += 1
        else:
            failed += 1
    
    print("="*60)
    print(f"Summary:")
    print(f"  ✓ Processed: {processed}")
    print(f"  ⊘ Skipped (already have rewards): {skipped}")
    print(f"  ✗ Failed: {failed}")
    print(f"\nRewards saved to: {output_dir}")
    print(f"Ready for DreamerV3 training!")
    print("="*60)


if __name__ == "__main__":
    main()
