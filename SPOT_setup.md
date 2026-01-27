Navigate to the workspace and run the build script. The building process can take more than 30 minutes on the RPi:

./docker_build.sh

This will build a Ubuntu 22.04 image (ROS2 Humble) and copy the spot_driver inside the container. It will be placed in the externals directory, which is accessible within the container only.

After building the container, execute the run script, then build and source the workspace:

./docker_run

colcon build
source install/setup.bash

This will start the container and mount the openbots/src/packages directory in it. packages contains a custom UDP to ROS2 bridge.

6. Launching spot_driver

To start the Spot driver, the following environment variables must be defined:

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export BOSDYN_CLIENT_USERNAME=user
export BOSDYN_CLIENT_PASSWORD=corspotuser1
export SPOT_IP=192.168.10.102

SPOT_IP should be configured for the robot during Teltonika setup.

The following launch file will launch the nmea_udp_to_ros and nmea_topic_driver nodes, as well as spot_driver:

ros2 launch spot_nav spot_driver_nmea.launch.py

spot_nav_node is a node that allows sending the Spot robot relative waypoints. Running the following command will send a walk forward command to the Spot robot, causing it to walk forward 1 meter:

ros2 run spot_nav spot_nav_node

copy in another terminal:  
docker exec -it openbots_container_new bash

record data: python3 -m dataset.spot_data_recorder

record rosbag 


ON SPOT:


- go to admin panel
- change name and password of network:
    SSID 	Password
RUT_1CF1_5G 	OBrutc%)19
- connect to new wifi with SPOT and PC

