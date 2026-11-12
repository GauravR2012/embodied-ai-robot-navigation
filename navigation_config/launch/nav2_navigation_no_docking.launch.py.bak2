import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node, SetParameter
from nav2_common.launch import RewrittenYaml


def generate_launch_description():

    bringup_dir = get_package_share_directory("nav2_bringup")

    namespace = LaunchConfiguration("namespace")
    use_sim_time = LaunchConfiguration("use_sim_time")
    autostart = LaunchConfiguration("autostart")
    params_file = LaunchConfiguration("params_file")
    use_respawn = LaunchConfiguration("use_respawn")
    log_level = LaunchConfiguration("log_level")

    lifecycle_nodes = [
        "controller_server",
        "smoother_server",
        "planner_server",
        "route_server",
        "behavior_server",
        "velocity_smoother",
        "collision_monitor",
        "bt_navigator",
        "waypoint_follower",
    ]

    remappings = [
        ("/tf", "tf"),
        ("/tf_static", "tf_static"),
    ]

    param_substitutions = {
        "autostart": autostart,
    }

    configured_params = RewrittenYaml(
        source_file=params_file,
        root_key=namespace,
        param_rewrites=param_substitutions,
        convert_types=True,
    )

    stdout_linebuf_envvar = SetEnvironmentVariable(
        "RCUTILS_LOGGING_BUFFERED_STREAM",
        "1",
    )

    declare_namespace_cmd = DeclareLaunchArgument(
        "namespace",
        default_value="",
        description="Top-level namespace",
    )

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
        description="Use simulation time if true",
    )

    declare_params_file_cmd = DeclareLaunchArgument(
        "params_file",
        default_value=os.path.join(
            bringup_dir,
            "params",
            "nav2_params.yaml",
        ),
        description="Full path to the Nav2 parameters file",
    )

    declare_autostart_cmd = DeclareLaunchArgument(
        "autostart",
        default_value="true",
        description="Automatically start the Nav2 stack",
    )

    declare_use_respawn_cmd = DeclareLaunchArgument(
        "use_respawn",
        default_value="False",
        description="Respawn nodes if they crash",
    )

    declare_log_level_cmd = DeclareLaunchArgument(
        "log_level",
        default_value="info",
        description="Log level",
    )

    load_nodes = GroupAction(
        actions=[

            SetParameter(
                "use_sim_time",
                use_sim_time,
            ),

            # ==========================================================
            # Controller Server
            # Publishes: /cmd_vel_nav
            # Message type: geometry_msgs/msg/Twist
            # ==========================================================
            Node(
                package="nav2_controller",
                executable="controller_server",
                name="controller_server",
                output="screen",
                respawn=use_respawn,
                respawn_delay=2.0,
                parameters=[configured_params],
                arguments=[
                    "--ros-args",
                    "--log-level",
                    log_level,
                ],
                remappings=remappings + [
                    ("cmd_vel", "cmd_vel_nav"),
                ],
            ),

            # ==========================================================
            # Smoother Server
            # ==========================================================
            Node(
                package="nav2_smoother",
                executable="smoother_server",
                name="smoother_server",
                output="screen",
                respawn=use_respawn,
                respawn_delay=2.0,
                parameters=[configured_params],
                arguments=[
                    "--ros-args",
                    "--log-level",
                    log_level,
                ],
                remappings=remappings,
            ),

            # ==========================================================
            # Planner Server
            # ==========================================================
            Node(
                package="nav2_planner",
                executable="planner_server",
                name="planner_server",
                output="screen",
                respawn=use_respawn,
                respawn_delay=2.0,
                parameters=[configured_params],
                arguments=[
                    "--ros-args",
                    "--log-level",
                    log_level,
                ],
                remappings=remappings,
            ),

            # ==========================================================
            # Route Server
            # ==========================================================
            Node(
                package="nav2_route",
                executable="route_server",
                name="route_server",
                output="screen",
                respawn=use_respawn,
                respawn_delay=2.0,
                parameters=[configured_params],
                arguments=[
                    "--ros-args",
                    "--log-level",
                    log_level,
                ],
                remappings=remappings,
            ),

            # ==========================================================
            # Behavior Server
            # Publishes: /cmd_vel_nav
            # Message type: geometry_msgs/msg/Twist
            # ==========================================================
            Node(
                package="nav2_behaviors",
                executable="behavior_server",
                name="behavior_server",
                output="screen",
                respawn=use_respawn,
                respawn_delay=2.0,
                parameters=[configured_params],
                arguments=[
                    "--ros-args",
                    "--log-level",
                    log_level,
                ],
                remappings=remappings + [
                    ("cmd_vel", "cmd_vel_nav"),
                ],
            ),

            # ==========================================================
            # Behavior Tree Navigator
            # ==========================================================
            Node(
                package="nav2_bt_navigator",
                executable="bt_navigator",
                name="bt_navigator",
                output="screen",
                respawn=use_respawn,
                respawn_delay=2.0,
                parameters=[configured_params],
                arguments=[
                    "--ros-args",
                    "--log-level",
                    log_level,
                ],
                remappings=remappings,
            ),

            # ==========================================================
            # Waypoint Follower
            # ==========================================================
            Node(
                package="nav2_waypoint_follower",
                executable="waypoint_follower",
                name="waypoint_follower",
                output="screen",
                respawn=use_respawn,
                respawn_delay=2.0,
                parameters=[configured_params],
                arguments=[
                    "--ros-args",
                    "--log-level",
                    log_level,
                ],
                remappings=remappings,
            ),

            # ==========================================================
            # Velocity Smoother
            # Input:  /cmd_vel_nav
            # Output: /cmd_vel_smoothed
            # ==========================================================
            Node(
                package="nav2_velocity_smoother",
                executable="velocity_smoother",
                name="velocity_smoother",
                output="screen",
                respawn=use_respawn,
                respawn_delay=2.0,
                parameters=[configured_params],
                arguments=[
                    "--ros-args",
                    "--log-level",
                    log_level,
                ],
                remappings=remappings + [
                    ("cmd_vel", "cmd_vel_nav"),
                ],
            ),

            # ==========================================================
            # Collision Monitor
            #
            # Input:  /cmd_vel_smoothed
            # Output: /cmd_vel
            #
            # The topics are configured in nav2_params.yaml.
            # ==========================================================
            Node(
                package="nav2_collision_monitor",
                executable="collision_monitor",
                name="collision_monitor",
                output="screen",
                respawn=use_respawn,
                respawn_delay=2.0,
                parameters=[configured_params],
                arguments=[
                    "--ros-args",
                    "--log-level",
                    log_level,
                ],
                remappings=remappings,
            ),

            # ==========================================================
            # Lifecycle Manager
            # ==========================================================
            Node(
                package="nav2_lifecycle_manager",
                executable="lifecycle_manager",
                name="lifecycle_manager_navigation",
                output="screen",
                arguments=[
                    "--ros-args",
                    "--log-level",
                    log_level,
                ],
                parameters=[
                    {
                        "autostart": autostart,
                        "node_names": lifecycle_nodes,
                    }
                ],
            ),
        ]
    )

    ld = LaunchDescription()

    ld.add_action(stdout_linebuf_envvar)

    ld.add_action(declare_namespace_cmd)
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_params_file_cmd)
    ld.add_action(declare_autostart_cmd)
    ld.add_action(declare_use_respawn_cmd)
    ld.add_action(declare_log_level_cmd)

    ld.add_action(load_nodes)

    return ld