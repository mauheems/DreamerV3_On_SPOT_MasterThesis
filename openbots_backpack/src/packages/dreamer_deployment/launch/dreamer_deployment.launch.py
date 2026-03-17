#!/usr/bin/env python3
"""
Launch file for Dreamer Policy deployment on SPOT.

Prerequisite: spot_driver_nmea.launch.py must already be running.
This file starts only the policy inference node.

Usage:
  ros2 launch dreamer_deployment dreamer_deployment.launch.py \
    checkpoint_path:=/home/ob/dreamer_results/checkpoint.ckpt \
    dreamerv3_root:=/home/ob/openbots_ws/src/dreamer_SPOT_implementation/informed-dreamer
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from pathlib import Path


def generate_launch_description() -> LaunchDescription:

    launch_args = [
        DeclareLaunchArgument(
            'checkpoint_path',
            default_value=str(Path.home() / 'dreamer_results/checkpoint.ckpt'),
            description='Path to trained Dreamer checkpoint (.ckpt). Change this to swap policies.'
        ),
        DeclareLaunchArgument(
            'dreamerv3_root',
            default_value='',
            description='Path to informed-dreamer root containing dreamerv3/ package.'
        ),
    ]

    param_file = PathJoinSubstitution([
        FindPackageShare('dreamer_deployment'),
        'config',
        'params.yaml'
    ])

    dreamer_node = Node(
        package='dreamer_deployment',
        executable='dreamer_policy_node',
        name='dreamer_policy_node',
        output='both',
        parameters=[
            param_file,
            {
                'checkpoint_path': LaunchConfiguration('checkpoint_path'),
                'dreamerv3_root':  LaunchConfiguration('dreamerv3_root'),
            }
        ],
    )

    ld = LaunchDescription(launch_args)
    ld.add_action(dreamer_node)
    return ld
