#!/usr/bin/env python3
"""
Pre-compute and save rewards for all HDF5 episodes in the dataset.
This processes all episodes offline before training.

Noobs reward design (v2):
  + Goal reward only  — Gaussian-shaped, no distance-decrease signal
  + Action smoothness — higher weight to encourage smooth commands
  + Orientation penalty — penalises robot for not facing the goal direction
  + Time penalty      — constant per-step cost to encourage speed

  Removed: distance-decrease reward, instability penalty, velocity-alignment penalty
"""

import h5py
import numpy as np
from pathlib import Path
import sys

# Centralized reward parameters (keep in sync with notebook Section 6 NOOBS_PARAMS)
REWARD_PARAMS = {
    # Goal reward: distance-decrease reward (primary gradient signal)
    # r = +delta_dist (negative when moving away from goal, positive when moving closer)
    'distance_scale':       1.0,   # scale of distance-decrease reward
    'goal_reach_threshold': 0.5,   # distance in metres below which goal is considered reached
    'goal_reach_bonus':     1,   # one-time reward bonus when entering goal_reach_threshold
    # Out-of-radius penalty: when robot moves away from goal AFTER reaching it (within 1m)
    'goal_radius':          0.5,   # metres
    'out_of_radius_penalty': 0.5,  # penalty per step when moving away from goal
    # Action smoothness: only applied to continuous command dims (vx, vy, vyaw), not gait
    'smoothness_scale':     0.1,
    # Orientation-to-goal: penalty = -scale * (1 - cos(heading, goal_dir)) / 2
    # Range: [0, -scale/2] per step.  0.2 → max -0.1/step, comparable to smoothness
    'orientation_scale':    0.1,
    # Time penalty: constant per-step cost
    'time_penalty':         0.01,
}


def compute_rewards_for_episode(states, actions, velocities, goal_xy=None, dt=0.1):
    """
    Compute noobs rewards for one episode.
    states:     (T, 7)  [x, y, z, qx, qy, qz, qw]
    actions:    (T, 4)  [vx_cmd, vy_cmd, vyaw_cmd, gait]
    velocities: (T, 6)  — not used in this reward version, kept for API compatibility
    goal_xy:    (2,)    goal position [x, y], defaults to final position if None
    Returns: reward array of shape (T,)
    """
    T = len(states)

    distance_scale       = REWARD_PARAMS['distance_scale']
    goal_threshold       = REWARD_PARAMS['goal_reach_threshold']
    goal_bonus           = REWARD_PARAMS['goal_reach_bonus']
    goal_radius          = REWARD_PARAMS['goal_radius']
    out_of_radius_pen    = REWARD_PARAMS['out_of_radius_penalty']
    smoothness_scale     = REWARD_PARAMS['smoothness_scale']
    orientation_scale    = REWARD_PARAMS['orientation_scale']
    time_penalty         = REWARD_PARAMS['time_penalty']

    # Goal is the provided position, or final position if not specified
    if goal_xy is None:
        goal_xy = states[-1, :2]
    goal_xy = goal_xy.astype(np.float32)


    # === Component 1: Distance-Decrease Reward (primary gradient) ===
    # reward(t) = -distance_scale * delta_dist
    # where delta_dist = dist(t-1) - dist(t)
    # Positive when moving closer (main RL signal), negative when moving away
    dist_to_goal = np.linalg.norm(states[:, :2] - goal_xy, axis=1)
    dist_changes = np.diff(dist_to_goal, prepend=dist_to_goal[0])  # Replicate first value
    distance_rewards = -distance_scale * dist_changes

    # === Component 1b: One-time Goal Reach Bonus ===
    # +goal_bonus when entering the goal_threshold radius (transition from outside to inside)
    is_in_goal = dist_to_goal < goal_threshold  # (T,) boolean
    crossed_into_goal = np.diff(is_in_goal.astype(int), prepend=0) > 0  # 0->1 transitions
    goal_reach_bonus = np.where(crossed_into_goal, goal_bonus, 0.0)

    # === Mask out distance and bonus rewards after first goal entry ===
    # Find first timestep where goal is reached
    first_goal_idx = np.argmax(is_in_goal)
    if not np.any(is_in_goal):
        # Never reached goal, so no masking
        mask = np.ones_like(distance_rewards, dtype=bool)
    else:
        # Only allow rewards up to and including first entry
        mask = np.arange(len(distance_rewards)) <= first_goal_idx
    distance_rewards = distance_rewards * mask
    goal_reach_bonus = goal_reach_bonus * mask

    # === Component 1c: Persistent Out-of-Radius Penalty ===
    # After first goal entry, penalize every step outside goal_radius
    if np.any(is_in_goal):
        # Only penalize after first goal entry
        after_goal_entry = np.arange(len(is_in_goal)) > first_goal_idx
        outside_radius = dist_to_goal > goal_radius
        out_of_radius_penalty = np.where(after_goal_entry & outside_radius, -out_of_radius_pen, 0.0)
    else:
        out_of_radius_penalty = np.zeros_like(dist_to_goal)

    # === Component 2: Action Smoothness Penalty ===
    # Only on continuous dims (vx, vy, vyaw) — index 3 is discrete gait selection
    if actions is not None and len(actions) > 1:
        action_diffs = np.diff(actions[:, :3], axis=0)   # (T-1, 3)
        action_magnitudes = np.linalg.norm(action_diffs, axis=1)
        jerk_penalties = -smoothness_scale * action_magnitudes
        jerk_penalties = np.pad(jerk_penalties, (1, 0), mode='constant', constant_values=0.0)
    else:
        jerk_penalties = np.zeros(T)

    # === Component 3: Orientation-to-Goal Penalty ===
    # Robot heading from quaternion yaw: [cos(yaw), sin(yaw)]
    # Direction to goal normalised: (goal_xy - pos_xy) / dist
    # penalty = -scale * (1 - cos_angle) / 2   range [0, -scale/2]
    qx, qy, qz, qw = states[:, 3], states[:, 4], states[:, 5], states[:, 6]
    yaw = np.arctan2(2.0 * (qw * qz + qx * qy),
                     1.0 - 2.0 * (qy ** 2 + qz ** 2))
    heading = np.stack([np.cos(yaw), np.sin(yaw)], axis=1)       # (T, 2)

    goal_dir = goal_xy - states[:, :2]                            # (T, 2)
    goal_dist_2d = np.linalg.norm(goal_dir, axis=1, keepdims=True)
    goal_dir_norm = goal_dir / np.maximum(goal_dist_2d, 1e-6)     # (T, 2)

    cos_angle = np.sum(heading * goal_dir_norm, axis=1)           # (T,)
    orientation_penalties = -orientation_scale * (1.0 - cos_angle) / 2.0

    # === Component 4: Time Penalty ===
    time_penalties = np.full(T, -time_penalty)

    # === Combine ===
    rewards = distance_rewards + goal_reach_bonus + out_of_radius_penalty + jerk_penalties + orientation_penalties + time_penalties

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
            
            # Read goal position from metadata (set by converter)
            # Default to final position if not specified
            if 'goal_position_x' in f_in.attrs and 'goal_position_y' in f_in.attrs:
                goal_xy = np.array([
                    f_in.attrs['goal_position_x'],
                    f_in.attrs['goal_position_y']
                ], dtype=np.float32)
                goal_type = f_in.attrs.get('goal_type', 'end_position')
            else:
                # Fallback for episodes without explicit goal metadata
                goal_xy = states[-1, :2].astype(np.float32)
                goal_type = 'end_position'
            
            # Compute rewards
            rewards = compute_rewards_for_episode(states, actions, velocities, goal_xy)
            
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
                f_out.attrs['reward_components'] = 'distance_decrease, goal_reach_bonus, out_of_radius_penalty, smoothness, orientation, time'
                f_out.attrs['goal_used_for_reward'] = goal_type
            
            print(f"  ✓ Computed {len(rewards)} rewards: range=[{rewards.min():.4f}, {rewards.max():.4f}], sum={rewards.sum():.4f}")
            print(f"    Goal type: {goal_type}, position: ({goal_xy[0]:.2f}, {goal_xy[1]:.2f})")
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
                        help="Output directory for files with rewards (default: processed_data_NoObs_with_rewards)")
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
