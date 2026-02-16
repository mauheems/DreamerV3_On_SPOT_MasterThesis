from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    launch_description = LaunchDescription()
    nmea_udp_to_ros = Node (
        package='nmea_udp_bridge',
        executable='nmea_udp_to_ros',
        output='screen'
    )
    nmea_topic_driver = Node (
        package='nmea_navsat_driver',
        executable='nmea_topic_driver',
        output='screen'
    )
    vtg_heading_parser = Node (
        package='nmea_udp_bridge',
        executable='vtg_heading_parser',
        output='screen'
    )
    launch_description.add_action(nmea_udp_to_ros)
    launch_description.add_action(nmea_topic_driver)
    launch_description.add_action(vtg_heading_parser)
    return launch_description
