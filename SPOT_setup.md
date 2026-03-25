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


mount harddrive:
sudo mkdir -p /media/maurits/recorded_data
sudo chown -R $(id -u):$(id -g) /media/external_drive/recorded_data
sudo chmod -R 775 /media/external_drive/recorded_data
# quick test as the same user:
touch /media/external_drive/recorded_data/test && echo OK || echo FAIL


# 3. After recording, convert the bag to HDF5
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



# DAIC 
## upload data to DAIC from harddrive:

rsync -avz --progress --partial --append-verify --no-perms -e "ssh -J mjheemskerk@student-linux.tudelft.nl" /media/maurits-heemskerk/69987a47-b840-4db7-9f8b-7cc05f14d09e3/processed_data_with_rewards mjheemskerk@login.daic.tudelft.nl:/tudelft.net/staff-umbrella/openbots/mjheemskerk/spot_data


## build apptainer: 
apptainer build   --tmpdir /media/maurits-heemskerk/69987a47-b840-4db7-9f8b-7cc05f14d09e2/tmp   /media/maurits-heemskerk/69987a47-b840-4db7-9f8b-7cc05f14d09e2/dreamer_spot.sif   docker-daemon://dreamer_spot_cuda12:latest

## copy apptainer to DAIC
rsync -avz --progress   /media/maurits-heemskerk/69987a47-b840-4db7-9f8b-7cc05f14d09e2/dreamer_spot.sif   mjheemskerk@login.daic.tudelft.nl:/home/nfs/mjheemskerk/


## sinteractive:
sinteractive --mem=96G --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --time=4:00:00 

## run training in sinteractive: 
apptainer exec --cleanenv --nv \
  -B /home/nfs/mjheemskerk:/mnt/home \
  -B /tudelft.net:/tudelft.net \
  --env PYTHONNOUSERSITE=1,PYTHONPATH= \
  /tudelft.net/staff-umbrella/openbots/mjheemskerk/dreamer_spotv5.sif \
  python3 /mnt/home/Master_thesis_DAIC_code/dreamerv3/train.py \
    --logdir /mnt/home/logdir/test_run \
    --configs spot \
    --task spot_nav \
    --jax.platform gpu \
    --jax.prealloc False \
    --data_loaders 1 \
    --batch_size 2 \
    --batch_length 8 \
    --replay_size 2e4 \
    --envs.amount 1 \
    --run.train_ratio 16 \
    --env.spot.data_dir=/tudelft.net/staff-umbrella/openbots/mjheemskerk/spot_data/processed_data_with_rewards


## pull logdir files from training to local: 
rsync -avz --progress --exclude='replay' \
  daic:/home/nfs/mjheemskerk/logdir/dreamer-results-20260310-160927 \
  /home/maurits-heemskerk/Documents/Uni/Master_Thesis/dreamer_results_local/

## push results directory back to DAIC logdir:
rsync -avz --progress --exclude='replay' \
  /home/maurits-heemskerk/Documents/Uni/Master_Thesis/dreamer_results_local/dreamer-results-YYYYMMDD-HHMMSS-label \
  daic:/home/nfs/mjheemskerk/logdir/


## run policy:

ros2 launch dreamer_deployment dreamer_deployment.launch.py   checkpoint_path:=/home/ob/dreamer_results_local/dreamer-results-20260316-104505-informedlargeRSSM/checkpoint.ckpt

# with target goal (x/y are RELATIVE to robot's current position)
ros2 topic pub --once /spot/policy/goal   geometry_msgs/msg/PointStamped   "{header: {frame_id: 'odom'}, point: {x: 2.0, y: 0.0, z: 0.0}}"