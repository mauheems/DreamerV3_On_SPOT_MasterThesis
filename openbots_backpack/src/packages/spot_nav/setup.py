import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'spot_nav'

setup(
    name=package_name,
    version='0.0.0',
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
    description='Package to send relative movement commands to Spot',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'spot_nav_node = spot_nav.spot_nav_node:main',
        ],
    },
)

