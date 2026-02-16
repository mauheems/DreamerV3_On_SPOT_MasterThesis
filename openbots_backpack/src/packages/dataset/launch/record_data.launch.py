from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'output_dir',
            default_value='./spot_data',
            description='Directory to save recorded data'
        ),
        
        DeclareLaunchArgument(
            'record_frequency',
            default_value='10.0',
            description='Recording frequency in Hz'
        ),
        
        DeclareLaunchArgument(
            'episode_length',
            default_value='1000',
            description='Maximum steps per episode'
        ),

        DeclareLaunchArgument(
            'terrain_grid_source',
            default_value='topic',
            description='Source of terrain grid: topic or sdk'
        ),
        
        Node(
            package='dataset',
            executable='spot_data_recorder',
            name='spot_data_recorder',
            output='screen',
            parameters=[{
                'output_dir': LaunchConfiguration('output_dir'),
                'record_frequency': LaunchConfiguration('record_frequency'),
                'episode_length': LaunchConfiguration('episode_length'),
                'terrain_grid_source': LaunchConfiguration('terrain_grid_source')
            }]
        )
    ])