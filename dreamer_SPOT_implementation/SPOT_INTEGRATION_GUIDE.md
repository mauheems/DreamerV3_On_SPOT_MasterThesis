# SPOT Environment Integration Guide

## Overview

This directory now contains a complete DreamerV3 setup for training SPOT robot policies. The SPOT environment wrapper is located in `embodied/envs/spot.py`.

## Files Created/Modified

### New Files
- **`embodied/envs/spot.py`** - Main SPOT environment wrapper (450+ lines with detailed comments)
- **`embodied/tests/test_spot.py`** - Comprehensive test suite for SPOT environment
- **`conftest.py`** - Pytest configuration for correct import paths

### Modified Files
- **`dreamerv3/configs.yaml`** - Added SPOT-specific configuration blocks

## What You Need to Implement

The `spot.py` wrapper is a template with TODO markers. You need to fill in:

### 1. Robot Connection (`__init__`)
- Connect to real SPOT or simulator
- Handle initialization parameters
- Set up communication interfaces

**Example:**
```python
if use_sim:
    from spot_sim import SpotSimulator
    self.robot = SpotSimulator(**kwargs)
else:
    import bosdyn.client
    self.robot = self._connect_to_real_spot(**kwargs)
```

### 2. Observation Collection (`_get_observations`)
- Capture front camera image (480×640×3 RGB)
- Get stereo obstacle/depth map (64×64)
- Read joint angles, IMU, contact states
- Return all values as numpy arrays

**Required observations:**
- `image`: RGB camera (uint8)
- `obstacle_map`: Distance or height map (float32)
- `joint_angles`: 12 joint angles in radians (float32)
- `body_orientation`: Roll, pitch, yaw (float32)
- `body_velocity`: vx, vy, vz (float32)
- `contact_state`: 4 feet contact flags (bool)

### 3. Action Execution (in `step`)
- Parse gait selection (int: 0=trot, 1=pace, 2=bound, etc.)
- Parse velocity commands (vx, vy, angular_vel)
- Send to robot controller
- Handle timing/synchronization

**Action structure:**
```python
gait_id = int(action['gait'])
vel_x = float(action['velocity_x'])
vel_y = float(action['velocity_y'])
angular_vel = float(action['angular_velocity'])
```

### 4. Reward Function (`_compute_reward`)
- Define what constitutes good behavior
- Reward forward progress, obstacle traversal, stability, efficiency
- Return scalar reward

**Example for obstacle traversal:**
```python
reward = 0.0
reward += max(0, vel_x) * 0.1  # Forward motion bonus
reward -= collision_penalty
reward += traversal_bonus
return reward
```

### 5. Termination Conditions (`_check_termination`)
- Detect robot falls (pitch/roll > threshold)
- Detect collisions (force feedback, obstacle hits)
- Detect out-of-bounds conditions
- Return boolean

**Example:**
```python
roll = self.observation['body_orientation'][0]
pitch = self.observation['body_orientation'][1]
if abs(roll) > 0.5 or abs(pitch) > 0.5:
    return True  # Robot fell
```

## Testing Your Implementation

All the code is tested and working with dummy data:

```bash
cd /path/to/dreamerv3
python3 -m pytest embodied/tests/test_spot.py -v
```

Or run tests directly:
```bash
python3 << 'EOF'
import sys
sys.path.insert(0, '.')
from embodied.tests.test_spot import TestSpotEnv
test = TestSpotEnv()
test.test_reset()  # etc.
EOF
```

## Training Configurations

Three pre-configured training scenarios are available:

### 1. Simulated SPOT Training
```bash
python dreamerv3/main.py \
  --configs spot_sim debug \
  --logdir ~/logdir/spot_sim
```
- Fast iteration on simulated robot
- 1M training steps, 1 parallel env
- train_ratio: 256 (lots of training per collection step)

### 2. Real SPOT Data Collection + Offline Training
```bash
python dreamerv3/main.py \
  --configs spot_real debug \
  --logdir ~/logdir/spot_real
```
- Designed for real robot operation
- Collects ~100k steps of real data
- Lower train_ratio for online learning
- Can switch to pure offline after collection

### 3. Obstacle Traversal Task
```bash
python dreamerv3/main.py \
  --configs spot_obstacle_sim debug \
  --logdir ~/logdir/spot_obstacles
```
- Specialized for obstacle navigation
- 2 parallel environments
- 5M total training steps

## File Structure

```
dreamerv3/
├── embodied/
│   ├── envs/
│   │   ├── spot.py           ← Your SPOT wrapper (EDIT THIS)
│   │   ├── dummy.py          ← Reference: dummy environment
│   │   ├── from_gym.py       ← Reference: Gym adapter
│   │   └── from_dm.py        ← Reference: DeepMind adapter
│   ├── core/                 ← Training infrastructure (don't touch)
│   └── tests/
│       ├── test_spot.py      ← SPOT environment tests
│       └── test_driver.py    ← Other infrastructure tests
├── dreamerv3/
│   ├── main.py               ← Entry point (don't modify)
│   ├── agent.py              ← DreamerV3 agent (don't modify)
│   ├── configs.yaml          ← Configuration (edit for new configs)
│   └── rssm.py               ← World model (don't modify)
└── conftest.py               ← Pytest configuration
```

## Next Steps

1. **Understand your SPOT API** - Check Boston Dynamics SDK documentation
2. **Fill in robot connection** - Implement `__init__` method
3. **Implement observations** - Fill in `_get_observations()`
4. **Test with dummy data** - Run the test suite
5. **Implement actions** - Fill in action execution in `step()`
6. **Tune reward** - Implement `_compute_reward()`
7. **Add termination** - Implement `_check_termination()`
8. **Collect real data** - Run actual robot collection
9. **Train world model** - Use offline training mode
10. **Deploy policy** - Test policy on real robot

## Debugging Tips

- **View TensorBoard logs:**
  ```bash
  tensorboard --logdir ~/logdir/spot_sim
  ```

- **Test environment in isolation:**
  ```python
  from embodied.envs.spot import Spot
  env = Spot(use_sim=True)
  obs = env.reset()
  for _ in range(100):
      action = {'reset': False, 'gait': 0, 'velocity_x': 0.5, ...}
      obs = env.step(action)
      print(f"Reward: {obs['reward']}")
  ```

- **Check observation shapes:**
  ```python
  env = Spot()
  obs = env.reset()
  for key, val in obs.items():
      print(f"{key}: {val.shape if hasattr(val, 'shape') else type(val)}")
  ```

## Common Issues

**Import errors?** Make sure you're in the dreamerv3 root and have run:
```bash
python -m pip install -r requirements.txt
```

**Robot doesn't connect?** Check that SPOT SDK is installed and reachable:
```bash
python -c "import bosdyn; print(bosdyn.__file__)"
```

**Tests fail?** Run individual tests to debug:
```python
from embodied.tests.test_spot import TestSpotEnv
TestSpotEnv().test_observation_ranges()
```

## Thesis Integration

When writing your thesis, you can reference:
- The architecture: "We adapted the DreamerV3 framework with a custom SPOT environment wrapper"
- The implementation: "The wrapper translates SPOT sensor data (camera, stereo depth, proprioception) to observations and executes gait+velocity actions"
- The configuration: "We configured three training scenarios: simulation-based, real robot online, and offline training on collected data"

Good luck! 🤖
