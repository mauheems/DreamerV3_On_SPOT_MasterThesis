from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'dataset'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='SPOT data recording package for DreamerV3 training',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'spot_data_recorder = dataset.spot_data_recorder:main',
            'spot_local_grid_node = dataset.spot_local_grid_node:main',
            'spot_teleop_custom = dataset.spot_teleop_custom:main',
            'spot_teleop_joy = dataset.spot_teleop_joy:main',
        ],
    },
)