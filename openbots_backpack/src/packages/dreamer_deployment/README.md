# Dreamer Policy Deployment for SPOT

This package contains the inference and deployment code for running a trained Dreamer policy on the SPOT robot. It's integrated into the openbots_backpack ecosystem for seamless deployment.

## Package Structure

```
openbots_backpack/src/packages/dreamer_deployment/
├── dreamer_deployment/
│   ├── __init__.py
│   ├── dreamer_policy_node.py      # Main ROS 2 node for policy inference
│   └── goal_command_client.py      # Helper script to send goal commands
├── launch/
│   └── dreamer_deployment.launch.py   # Launch file
├── config/
│   └── params.yaml                 # Configuration file
├── srv/
│   └── SetGoalWaypoint.srv         # Service definition for setting goals
├── package.xml                     # ROS 2 package manifest
├── setup.py                        # Python package setup
├── setup.cfg                       # Python package config
└── README.md
```

## Components

### dreamer_policy_node.py
- Loads trained Dreamer checkpoint
- Subscribes to SPOT's joint states and IMU
- Maintains state history buffer (sequence of observations)
- Runs policy inference at 50 Hz
- Publishes joint commands to SPOT's controller
- Exposes service to set goal waypoints

### SetGoalWaypoint.srv
Custom ROS service for setting goal states:
```
Request:
  float32 x           # Goal X position
  float32 y           # Goal Y position  
  float32 theta       # Goal orientation

Response:
  bool success
  string message
```

## Setup

### 1. Add to openbots_backpack Workspace

The package is already in: `openbots_backpack/src/packages/dreamer_deployment/`

Build it with the rest of openbots_backpack:

```bash
cd ~/Documents/Uni/Master_Thesis/openbots_backpack
colcon build --packages-select dreamer_deployment
```

Or build everything:
```bash
colcon build
```

### 2. Update Checkpoint Path

Edit `config/params.yaml` or pass as launch argument:

```yaml
checkpoint_path: "/path/to/your/checkpoint.ckpt"
```

### 3. Verify Observation Format

Ensure your policy's expected observation dimensions match SPOT's sensors:
- **obs_dim: 48** (default in params.yaml)
- **Breakdown**: joint_positions(12) + joint_velocities(12) + imu_accel(3) + imu_ang_vel(3) + goal_state(3) + padding(6)

Modify `_build_observation()` in `dreamer_policy_node.py` if different.

## Usage

### Source Setup

```bash
cd ~/Documents/Uni/Master_Thesis/openbots_backpack
source install/setup.bash
```

### Option 1: Mock Mode (Testing)

```bash
# Start the node with mock hardware (no real robot needed)
ros2 launch dreamer_deployment dreamer_deployment.launch.py \
    hardware_interface:=mock \
    checkpoint_path:=/path/to/checkpoint.ckpt
```

### Option 2: Real Robot

```bash
# Start on real SPOT
ros2 launch dreamer_deployment dreamer_deployment.launch.py \
    hardware_interface:=robot \
    checkpoint_path:=/path/to/checkpoint.ckpt \
    robot_name:=spot
```

### Set Goal Waypoint

In another terminal:

```bash
# Use helper script
ros2 run dreamer_deployment goal_command_client --x 1.0 --y 0.5 --theta 0.0

# Or use ros2 service CLI
ros2 service call /spot/policy/set_goal dreamer_deployment/SetGoalWaypoint \
    "x: 1.0; y: 0.5; theta: 0.0"
```

## Configuration

Key parameters in `config/params.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `checkpoint_path` | - | Path to `.ckpt` file |
| `control_rate_hz` | 50 | Inference loop frequency |
| `history_length` | 32 | State buffer length (must match training) |
| `obs_dim` | 48 | Observation vector size |
| `action_dim` | 12 | Number of joints (4 legs × 3 joints) |
| `robot_name` | spot | Robot namespace |

## Topics and Services

### Subscriptions
- `/<robot_name>/joint_states` (sensor_msgs/JointState)
- `/<robot_name>/imu` (sensor_msgs/Imu)

### Publications
- `/<robot_name>/policy_action` (std_msgs/Float32MultiArray) - Debug action vector
- `/<robot_name>/spot_joint_controller/joint_commands` (spot_msgs/JointCommand) - Actual robot commands

### Services
- `/<robot_name>/policy/set_goal` (SetGoalWaypoint)

## Policy Integration

To integrate your trained model, modify `_load_policy()` in `dreamer_deployment/dreamer_policy_node.py`:

```python
def _load_policy(self, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=self.device)
    
    # TODO: Initialize and load your model
    self.policy = YourDreamerModel(...)
    self.policy.load_state_dict(checkpoint['model_state_dict'])
    self.policy.to(self.device)
    self.policy.eval()
```

And update the forward pass in `_control_loop()`:

```python
action = self.policy(state_tensor, goal_tensor)
```

## Debugging

### Check node status
```bash
ros2 node list
ros2 node info /spot/dreamer_policy_node
```

### Monitor topics
```bash
ros2 topic echo /spot/policy_action
ros2 topic echo /spot/joint_states
```

### View logs
```bash
ros2 launch dreamer_deployment dreamer_deployment.launch.py --verbose
```

## Safety Notes

1. **Always test in mock mode first** before running on real robot
2. **Set action bounds** to prevent joint damage
3. **Emergency stop**: Use SPOT's E-stop or kill the node
4. **Rate limiting**: Policy runs at 50 Hz, ensure this matches robot control frequency

## Integration with openbots_backpack

This package leverages the openbots_backpack infrastructure:
- Uses existing SPOT hardware abstraction layer
- Integrates with colcon build system
- Shares same Docker environment and drivers
- Can use other openbots_backpack tools and utilities

## Next Steps

1. [ ] Load your trained Dreamer checkpoint
2. [ ] Create proper JointCommand message publishing (currently placeholder)
3. [ ] Test observation building with real SPOT data
4. [ ] Integrate goal service callback with SetGoalWaypoint
5. [ ] Add safety checks and action bounds
6. [ ] Test in mock mode
7. [ ] Deploy on real SPOT
