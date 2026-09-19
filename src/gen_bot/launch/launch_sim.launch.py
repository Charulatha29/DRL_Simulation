import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():

    package_name = 'gen_bot'
    package_dir = get_package_share_directory(package_name)

    # 1. Set Gazebo Model Path explicitly to prevent scanning root/parent folders
    gazebo_model_path = SetEnvironmentVariable(
        'GAZEBO_MODEL_PATH',
        os.path.join(package_dir, 'models')
    )

    # 2. Robot State Publisher (RSP)
    rsp = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [os.path.join(package_dir, 'launch', 'rsp.launch.py')]
        ),
        launch_arguments={'use_sim_time': 'true'}.items()
    )

    # 3. Path to custom rc.world environment
    world_file = os.path.join(package_dir, 'worlds', 'rc.world')

    # 4. Gazebo Simulation Server & GUI Launch
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [os.path.join(get_package_share_directory('gazebo_ros'),
            'launch', 'gazebo.launch.py')]
        ),
        launch_arguments={
            'world': world_file,
            'verbose': 'false'  # Suppresses verbose plugin warnings
        }.items()
    )

    # 5. Spawn Robot Entity at Start Pose (x=0.25, y=0.25)
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'gen_bot',
            '-x', '0.25',
            '-y', '0.25',
            '-z', '0.05',
            '-Y', '0.0'
        ],
        output='screen'
    )

    return LaunchDescription([
        gazebo_model_path,
        rsp,
        gazebo,
        spawn_entity,
    ])
