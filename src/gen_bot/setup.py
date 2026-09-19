import os
from glob import glob
from setuptools import setup

package_name = 'gen_bot'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.world')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='charu',
    maintainer_email='charu@todo.todo',
    description='ROS2 Gazebo DRL Navigation Package',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gazebo_gym_env = scripts.gazebo_gym_env:main',
            'hybrid_astar = scripts.hybrid_astar:main',
            'gazebo_hybrid_drl_navigator = scripts.gazebo_hybrid_drl_navigator:main',
        ],
    },
)