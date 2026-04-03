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
                'disable_gait_selection': LaunchConfiguration('disable_gait_selection'),
                'fixed_gait_mode': LaunchConfiguration('fixed_gait_mode'),
            }
        ],
        additional_env={
            # Disable strict cuDNN autotuning — falls back to default algorithm
            # when the GPU/cuDNN version doesn't support all benchmark engines.
            'XLA_FLAGS': '--xla_gpu_strict_conv_algorithm_picker=false',
            # Disable JAX's 75% GPU memory pre-allocation so XLA autotuning
            # scratch buffers don't compete with the reserved pool (prevents OOM).
            'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
        },
    )

    ld = LaunchDescription(launch_args)
    ld.add_action(dreamer_node)
    return ld
