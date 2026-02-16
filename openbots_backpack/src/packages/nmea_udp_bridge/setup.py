import os
from glob import glob
from setuptools import setup

package_name = 'nmea_udp_bridge'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Kseniia Khomenko',
    maintainer_email='khomenko.ks99@gmail.com',
    description='Package to receive NMEA over UDP and republish them as ROS 2 messages',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'nmea_udp_to_ros = nmea_udp_bridge.nmea_udp_to_ros:main',
            'vtg_heading_parser = nmea_udp_bridge.vtg_heading_parser:main'
        ],
    },
)



# from setuptools import find_packages, setup

# package_name = 'nmea_udp_bridge'

# setup(
#     name=package_name,
#     version='0.0.0',
#     packages=find_packages(exclude=['test']),
#     data_files=[
#         ('share/ament_index/resource_index/packages',
#             ['resource/' + package_name]),
#         ('share/' + package_name, ['package.xml']),
#     ],
#     install_requires=['setuptools'],
#     zip_safe=True,
#     maintainer='ob',
#     maintainer_email='ob@todo.todo',
#     description='TODO: Package description',
#     license='TODO: License declaration',
#     tests_require=['pytest'],
#     entry_points={
#         'console_scripts': [
#         ],
#     },
# )
