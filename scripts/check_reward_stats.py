import h5py
import numpy as np
import os
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/media/maurits-heemskerk/69987a47-b840-4db7-9f8b-7cc05f14d09e10/processed_data_NoObs_with_rewardsv8')
parser.add_argument('--max_files', type=int, default=None, help='Limit number of files (default: all)')
args = parser.parse_args()

files = sorted(glob.glob(os.path.join(args.data_dir, '*.h5')))
if args.max_files:
    files = files[:args.max_files]
print(f'Found {len(files)} HDF5 files in {args.data_dir}')

all_rewards = []
for f in files:
    try:
        with h5py.File(f, 'r') as hf:
            key = 'rewards' if 'rewards' in hf else 'reward'
            all_rewards.append(hf[key][:])
    except Exception as e:
        print(f'  Skipping {os.path.basename(f)}: {e}')

rewards = np.concatenate(all_rewards)
print(f'\nReward statistics over {len(rewards):,} steps ({len(files)} files):')
print(f'  mean:         {rewards.mean():.4f}')
print(f'  std:          {rewards.std():.4f}')
print(f'  min:          {rewards.min():.4f}')
print(f'  max:          {rewards.max():.4f}')
print(f'  median:       {np.median(rewards):.4f}')
print(f'  p25 / p75:    {np.percentile(rewards, 25):.4f} / {np.percentile(rewards, 75):.4f}')
print(f'  pct positive: {(rewards > 0).mean()*100:.1f}%')
print(f'  pct negative: {(rewards < 0).mean()*100:.1f}%')
print(f'  pct zero:     {(rewards == 0).mean()*100:.1f}%')

# Per-episode mean (one mean per episode, then average those)
episode_means = [r.mean() for r in all_rewards]
print(f'\nPer-episode mean reward:')
print(f'  mean of episode means: {np.mean(episode_means):.4f}')
print(f'  std of episode means:  {np.std(episode_means):.4f}')
print(f'  episodes with positive mean: {(np.array(episode_means) > 0).mean()*100:.1f}%')

# Show keys in first file
with h5py.File(files[0], 'r') as hf:
    print(f'\nHDF5 keys: {list(hf.keys())}')
    key = 'rewards' if 'rewards' in hf else 'reward'
    print(f'Episode length (first file): {len(hf[key])}')
