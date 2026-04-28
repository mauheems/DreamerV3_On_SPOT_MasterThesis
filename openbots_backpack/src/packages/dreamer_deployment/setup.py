from setuptools import find_packages, setup

package_name = 'dreamer_deployment'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/dreamer_deployment.launch.py',
            'launch/dreamer_deployment_noobs.launch.py',
        ]),
        ('share/' + package_name + '/config', ['config/params.yaml']),
        ('share/' + package_name + '/srv', ['srv/SetGoalWaypoint.srv']),
    ],
    install_requires=['setuptools', 'readchar'],
    zip_safe=True,
    maintainer='Maurits',
    maintainer_email='maurits@example.com',
    description='Dreamer policy deployment node for SPOT robot',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'dreamer_policy_node = dreamer_deployment.dreamer_policy_node:main',
            'dreamer_policy_node_noobs = dreamer_deployment.dreamer_policy_node_noobs:main',
            'goal_command_client = dreamer_deployment.goal_command_client:main',
        ],
    },
)
