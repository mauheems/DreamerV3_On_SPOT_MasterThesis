# openbots_backpack
## Set up the Raspberry Pi 5
#### 1. Flash SD card with RPi OS (64-bit).
To flash the SD card, you can use [Raspberry Pi Imager](https://www.raspberrypi.com/software/). For the current setup, RPi OS Debian Bookworm (64-bit) was used.

#### 2. Enable SSH service
Run this command to enable the SSH service, then reboot:
```
sudo systemctl enable ssh.service
```
#### 3. Create and link SSH key for GitHub
*TODO: Use a public account instead of private*

- Add new SSH key: [https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent]

- Link the SSH key to GitHub account: [https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account]

#### 4. Docker for ROS2
Install Docker for RPi: [Docker Engine on Debian](https://docs.docker.com/engine/install/debian/)

Add the current user to the Docker group:
```
sudo usermod -aG docker $USER
```
Log out and log back in for the group membership changes to take effect.

#### 5. Building Docker image
Clone the repository:
```
git clone --recurse-submodules git@github.com:OpenBots/openbots_backpack.git
cd openbots_backpack
git submodule update --init --recursive --remote
```
Navigate to the workspace and run the build script. The building process can take more than 30 minutes on the RPi:
```
./docker_build.sh
```
This will build a Ubuntu 22.04 image (ROS2 Humble) and copy the `spot_driver` inside the container. It will be placed in the `externals` directory, which is accessible within the container only.

After building the container, execute the `run` script, then build and source the workspace:
```
./docker_run
```
```
colcon build
source install/setup.bash
```
This will start the container and mount the `openbots/src/packages` directory in it. `packages` contains a custom UDP to ROS2 bridge.

#### 6. Launching `spot_driver`
To start the Spot driver, the following environment variables must be defined:
```
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export BOSDYN_CLIENT_USERNAME=user
export BOSDYN_CLIENT_PASSWORD=corspotuser1
export SPOT_IP=192.168.10.102
```
`SPOT_IP` should be configured for the robot during Teltonika setup.

The following launch file will launch the `nmea_udp_to_ros` and `nmea_topic_driver` nodes, as well as `spot_driver`:
```
ros2 launch spot_nav spot_driver_nmea.launch.py
```
`spot_nav_node` is a node that allows sending the Spot robot relative waypoints. Running the following command will send a `walk forward` command to the Spot robot, causing it to walk forward 1 meter:
```
ros2 run spot_nav spot_nav_node
```

## Overview
The image uses a Ubuntu 22.04 base image with ROS2 Humble installed. The `nmea_navsat_driver` package is used to parse the forwarded messages and publish them to the appropriate topics.


### Teltonika setup
The RUTC50 is used to establish a connection between Spot, the installed RPi, and the user. The router needs to be set up before using the robot.
For the existing configuration, the RUTC50 has the following credentials:
| Username | Password | 
| --- | --- |
admin | P[rmNpyd!$9 | 

**Wireless interfaces:**
 | SSID | Password |
 | --- | --- |
 | RUT_1CF1_5G | OBrutc%)19
 |RUT_1CF0_2G | OBrutc%)19
 
 **CLI:**
 | Username | Password | 
| --- | --- |
root | P[rmNpyd!$9 | 

The router's IP address is set to `192.168.10.1`.

#### Static IP addresses
Static IP address can be assigned using `Network > DHCP > Static Leases` page in the RUTC50's WebUI.
##### Raspberry Pi
The user needs to assign static IP addresses for both the RPi and Spot. The RPi's IP is used in Teltonika's NMEA sentence forwarding (`Services > GPS > NMEA`). For the current configuration, the UDP protocol is used. The RPi's port selected for NMEA forwarding needs to match the port used with `nmea_udp_bridge`, which is currently set to 10110.
For the existing configuration, RPi has the following credentials:
| Username | Password | 
| --- | --- |
openbots | openbots |

The RPI's IP address is set to `192.168.10.101`.

##### Spot
Spot's assigned static IP needs to match the one used with the `spot_driver`; it is set to `192.168.10.102`.
> [!NOTE]
>To avoid routing conflicts, ensure that Spot is not assigned an IP address in any of the following ranges and that the network Spot connects to does not include these ranges in any of its subnets:
>- 192.168.0.x/24
>- 192.168.1.x/24
>- 192.168.50.x/24
>- 192.168.80.x/24
