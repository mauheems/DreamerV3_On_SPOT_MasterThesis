# SPOT Setup Guide

## 5. Building the Docker Container

Navigate to the workspace and run the build script. The building process can take **more than 30 minutes** on the RPi:

```bash
./docker_build.sh
```

This will:
- Build a Ubuntu 22.04 image (ROS2 Humble)
- Copy the spot_driver inside the container
- Place it in the `externals` directory (accessible within the container only)

### Running the Container

After building, execute the run script, then build and source the workspace:

```bash
./docker_run.sh
# This automatically starts the container with USB device access (/dev/input mounted)
# The --privileged flag is already included in the script

colcon build
source install/setup.bash
```

**Note:** The `docker_run.sh` script already includes USB device mapping (`-v /dev/input:/dev/input`), so you don't need to add `--privileged` separately.

This will:
- Start the container
- Mount the `openbots/src/packages` directory in it
- The `packages` directory contains a custom UDP to ROS2 bridge

---

## 6. Launching Spot Driver

### Environment Variables

First, set the following environment variables:

```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export BOSDYN_CLIENT_USERNAME=user
export BOSDYN_CLIENT_PASSWORD=corspotuser1
export SPOT_IP=192.168.10.102
```

> **Note:** `SPOT_IP` should be configured for the robot during Teltonika setup.

### Launch Commands

Launch the Spot driver and associated nodes:

```bash
ros2 launch spot_nav spot_driver_nmea.launch.py \
  cameras_used:=frontleft,frontright \
  stitch_front_images:=true \
  config_file:=/home/ob/openbots_ws/src/packages/spot_recorder_config.yaml
```

This launches:
- `joy_node` (joystick input)
- `nmea_udp_to_ros` node
- `nmea_topic_driver` node
- `spot_driver` node
- `spot_local_grid_node`

### Joystick Teleop (2nd terminal)

To enable joystick control, run in a separate terminal (after the spot driver is running):

```bash
ros2 run dataset spot_teleop_joy --ros-args \
  -p recording_mode:=expert \
  -p recording_difficulty:=medium
```

This will start accepting PS4 controller input for controlling the robot. Recordings will be saved directly to the external 1TB drive mounted at `/media/external_drive/recorded_data`.

**Joystick Controls:**
- **Left Stick**: Forward/Backward movement
- **Right Stick**: Rotation (Yaw)
- **X button**: Stand
- **Circle button**: Sit
- **L1**: Auto gait
- **L2**: SpeedSelectTrot gait
- **R1**: Crawl gait
- **R2**: Jog gait
- **L3**: SpeedSelectAmble gait
- **Square button**: Start/Stop recording
- **Triangle button**: Discard current recording (without saving)
- **D-Pad Down**: Mark collision event (flagged in rosbag)


### Data Recording (3rd terminal)

In another terminal, access the container:

```bash
docker exec -it openbots_container_new bash
```

Record sensor data:

```bash
python3 -m dataset.spot_data_recorder

cd src/packages/dataset/
python3 episode_bag_recorder.py
```

Record ROS2 bag:

```bash
ros2 bag record -a
```

# 3. After recording, convert the bag to HDF5
python compute_rewards_batch.py --harddrive /media/external_drive

cd ~/openbots_ws/src/packages/dataset/
python3 convert_bag_to_hdf5.py

python convert_bag_to_hdf5.py --batch --harddrive /media/external_drive/recorded_data

python compute_rewards_batch.py --harddrive /media/external_drive
---

## 7. Spot WiFi Network Configuration

On the **Spot admin panel**:

1. Go to **Admin Panel**
2. Change the network name and password:

| Setting | Value |
|---------|-------|
| **SSID** | RUT_1CF1_5G |
| **Password** | OBrutc%)19 |

3. Connect both the Spot and your PC to the new WiFi network

