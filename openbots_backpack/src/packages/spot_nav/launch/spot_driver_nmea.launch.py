import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    launch_description = LaunchDescription()

    spot_driver_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('spot_driver'),'launch', 'spot_driver.launch.py')
        )
    )
    nmea_udp_bridge_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('nmea_udp_bridge'), 'launch', 'nmea_udp_bridge.launch.py')
        )
    )

    spot_local_grid_node = Node(
        package='dataset',
        executable='spot_local_grid_node',
        name='spot_local_grid_node',
        output='screen'
    )

    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        output='screen'
    )

    launch_description.add_action(spot_driver_launch)
    launch_description.add_action(nmea_udp_bridge_launch)
    launch_description.add_action(spot_local_grid_node)
    launch_description.add_action(joy_node)
    return launch_description