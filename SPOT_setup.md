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
./docker_run
colcon build
source install/setup.bash
```

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
```bash
ros2 launch spot_driver spot_driver.launch.py \
  cameras_used:=frontleft,frontright \
  stitch_front_images:=true \
  config_file:=/home/ob/openbots_ws/src/packages/spot_recorder_config.yaml
```

This launches:
- `nmea_udp_to_ros` node
- `nmea_topic_driver` node
- `spot_driver` node

### Spot Navigation

Send waypoints to the Spot robot using `spot_nav_node`. For example, to walk forward 1 meter:

```bash
ros2 run spot_nav spot_nav_node
```

### Data Recording

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

