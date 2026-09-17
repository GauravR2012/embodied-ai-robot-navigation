import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription

from launch.launch_description_sources import (
    PythonLaunchDescriptionSource
)

from launch.substitutions import LaunchConfiguration


def generate_launch_description():

    # --------------------------------------------------
    # Package directories
    # --------------------------------------------------

    navigation_config_dir = get_package_share_directory(
        'navigation_config'
    )

    local_launch_dir = os.path.join(
        navigation_config_dir,
        'launch'
    )

    default_params_file = os.path.join(
        navigation_config_dir,
        'config',
        'nav2_params.yaml'
    )

    # --------------------------------------------------
    # Launch configuration variables
    # --------------------------------------------------

    namespace = LaunchConfiguration('namespace')
    use_sim_time = LaunchConfiguration('use_sim_time')
    autostart = LaunchConfiguration('autostart')
    params_file = LaunchConfiguration('params_file')

    # --------------------------------------------------
    # Launch arguments
    # --------------------------------------------------

    declare_namespace_cmd = DeclareLaunchArgument(
        'namespace',
        default_value='',
        description='Top-level namespace'
    )

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use Gazebo simulation clock'
    )

    declare_autostart_cmd = DeclareLaunchArgument(
        'autostart',
        default_value='true',
        description='Automatically start the Nav2 stack'
    )

    declare_params_file_cmd = DeclareLaunchArgument(
        'params_file',
        default_value=default_params_file,
        description='Path to the Nav2 parameter file'
    )

    # --------------------------------------------------
    # Include local Nav2 launch file without docking
    # --------------------------------------------------

    navigation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                local_launch_dir,
                'nav2_navigation_no_docking.launch.py'
            )
        ),
        launch_arguments={
            'namespace': namespace,
            'use_sim_time': use_sim_time,
            'autostart': autostart,
            'params_file': params_file,
        }.items(),
    )

    # --------------------------------------------------
    # Launch description
    # --------------------------------------------------

    ld = LaunchDescription()

    ld.add_action(declare_namespace_cmd)
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_autostart_cmd)
    ld.add_action(declare_params_file_cmd)
    ld.add_action(navigation_launch)

    return ld