#!/usr/bin/env python3
"""
Launch file for Dreamer Policy deployment on SPOT — NoObs (state-only) variant.

For models trained on the `noobs-dataset` branch.  No camera image or terrain
grid is required — only odometry is needed.

Prerequisite: spot_driver_nmea.launch.py must already be running.

Usage:
  ros2 launch dreamer_deployment dreamer_deployment_noobs.launch.py \
    checkpoint_path:=/home/ob/dreamer_results_noobs/rep0.2/checkpoint.ckpt \
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
            default_value=str(Path.home() / 'dreamer_results_noobs/rep0.2/checkpoint.ckpt'),
            description='Path to trained NoObs Dreamer checkpoint (.ckpt).'
        ),
        DeclareLaunchArgument(
            'dreamerv3_root',
            default_value='',
            description='Path to informed-dreamer root containing dreamerv3/ package.'
        ),
        DeclareLaunchArgument(
            'disable_gait_selection',
            default_value='false',
            description='If true, ignore action[3] and use fixed_gait_mode for all gaits.'
        ),
        DeclareLaunchArgument(
            'fixed_gait_mode',
            default_value='trot',
            description='Gait to use when disable_gait_selection=true. Either "trot" or "crawl".'
        ),
        DeclareLaunchArgument(
            'stop_at_goal',
            default_value='true',
            description='If true, the robot will stop at the target goal.'
        ),
        DeclareLaunchArgument(
            'record_rosbag',
            default_value='false',
            description='If true, record a rosbag with velocity, goal, orientation, and actions.'
        ),
        DeclareLaunchArgument(
            'rosbag_output_dir',
            default_value='',
            description='Optional rosbag output directory. Defaults to <checkpoint_dir>/rosbag.'
        ),
    ]

    param_file = PathJoinSubstitution([
        FindPackageShare('dreamer_deployment'),
        'config',
        'params.yaml'
    ])

    dreamer_node = Node(
        package='dreamer_deployment',
        executable='dreamer_policy_node_noobs',
        name='dreamer_policy_node_noobs',
        output='both',
        parameters=[
            param_file,
            {
                'checkpoint_path': LaunchConfiguration('checkpoint_path'),
                'dreamerv3_root':  LaunchConfiguration('dreamerv3_root'),
                'disable_gait_selection': LaunchConfiguration('disable_gait_selection'),
                'fixed_gait_mode': LaunchConfiguration('fixed_gait_mode'),
                'stop_at_goal': LaunchConfiguration('stop_at_goal'),
                'record_rosbag': LaunchConfiguration('record_rosbag'),
                'rosbag_output_dir': LaunchConfiguration('rosbag_output_dir'),
            }
        ],
        additional_env={
            'XLA_FLAGS': '--xla_gpu_strict_conv_algorithm_picker=false',
            'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
        },
    )

    ld = LaunchDescription(launch_args)
    ld.add_action(dreamer_node)
    return ld
