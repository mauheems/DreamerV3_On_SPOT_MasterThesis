# Data Collection

This document describes how robot episode data was collected, converted, and preprocessed for offline training.

For full details, including motivation, sensor setup, and statistics, refer to Chapter 3 of the [thesis paper](../Master-thesis-Maurits-Heemskerk.pdf).

---

## Overview

Data collection consisted of:
1. **Teleoperation** — manually driving SPOT through target environments
2. **Recording** — saving sensor observations as ROS2 bag files
3. **Conversion** — converting bags to HDF5 episode files
4. **Validation** — checking episode quality before training

---

## 1. Teleoperation Recording

Episodes were collected by teleoprating SPOT using a joystick controller. A ROS2 node recorded the following observations synchronously at ~5 Hz:

| Observation | Description |
|-------------|-------------|
| `velocity` | Body-frame linear (vx, vy) and angular (yaw rate) velocity |
| `imu` | IMU readings (orientation, angular velocity) |
| `joint_angles` | All 12 joint positions |
| `camera` | RGB image from front camera (visual variant only) |
| `terrain_map` | Local terrain elevation grid (visual variant only) |
| `goal_position` | Ego-centric goal vector (distance + bearing) |
| `action` | Commanded velocity sent to SPOT |

Episodes were stored as ROS2 `.bag` files.

---

## 2. ROS2 Bag → HDF5 Conversion

Bag files were converted to HDF5 episodes using a preprocessing script from the `openbots_backpack` package (not included in this repo — contact the author or university for details). Each HDF5 file corresponds to one episode with arrays of shape `(T, ...)` where `T` is the number of timesteps.

Expected HDF5 structure per episode:

```
episode.h5
├── obs/
│   ├── velocity       (T, 3)
│   ├── joint_angles   (T, 12)
│   ├── imu            (T, 6)
│   ├── camera         (T, H, W, 3)   # visual variant only
│   └── terrain_map    (T, H, W)      # visual variant only
├── action             (T, 3)
├── reward             (T,)
├── is_first           (T,)
└── is_terminal        (T,)
```

---

## 3. Reward Computation

Rewards were computed post-hoc using the goal position and robot pose logged during teleoperation. The reward function is a shaped combination of:

- **Goal proximity**: negative distance to goal
- **Velocity alignment**: dot product of velocity with goal direction
- **Smoothness penalty**: penalise large velocity changes

See `dreamerv3/embodied/envs/spot.py` in the [informed-dreamer fork](https://github.com/mauheems/Master_thesis_DAIC_code) for the reward implementation, and `notebooks/reward_computation.ipynb` for visualisation.

---

## 4. Validating Episodes

Before training, validate episodes:

```bash
cd dreamer_SPOT_implementation/informed-dreamer
python validate_episodes.py --data_dir /path/to/hdf5_episodes
```

This checks:
- Correct array shapes
- No NaN/Inf values
- Reasonable reward magnitudes
- Episode length distribution

See also: `notebooks/exploration_data_collected.ipynb` for interactive data inspection.

---

## Data Access

Raw and processed episode data is not included in this repository (file sizes are large). Contact the author if you need access to the processed HDF5 files for reproducing results.

Approximate dataset stats used in the thesis:

| Split | Episodes | Avg length |
|-------|----------|------------|
| Training | ~15 | ~120 steps |
| Validation | ~3 | ~120 steps |
