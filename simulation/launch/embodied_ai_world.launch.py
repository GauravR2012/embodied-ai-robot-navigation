#!/usr/bin/env python3

from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """
    Complete embodied-AI TurtleBot simulation.

    The startup sequence is deliberately deterministic.

    1. Start Gazebo.
    2. Wait until the Gazebo world service responds.
    3. Spawn the TurtleBot.
    4. Wait until the TurtleBot exists in Gazebo.
    5. Start the Gazebo <-> ROS bridge.
    6. Publish the ROS robot TF tree from the official TurtleBot URDF.
    7. Start SLAM Toolbox.
    8. Start Nav2.
    9. Start RViz.

    Gazebo uses the project's SDF/Xacro.

    robot_state_publisher uses the official TurtleBot URDF.

    This is intentional:
        SDF  -> Gazebo simulation
        URDF -> ROS TF / RViz RobotModel
    """

    # ================================================================
    # PROJECT PATHS
    # ================================================================

    project_root = Path(__file__).resolve().parents[2]

    world_file = (
        project_root
        / "simulation"
        / "worlds"
        / "embodied_ai_boxes.sdf"
    )

    robot_sdf_xacro = (
        project_root
        / "simulation"
        / "robot"
        / "gz_waffle.sdf.xacro"
    )

    if not world_file.exists():
        raise FileNotFoundError(
            f"Gazebo world does not exist:\n{world_file}"
        )

    if not robot_sdf_xacro.exists():
        raise FileNotFoundError(
            f"Robot SDF Xacro does not exist:\n{robot_sdf_xacro}"
        )

    # ================================================================
    # ROS PACKAGE PATHS
    # ================================================================

    nav2_bringup_share = Path(
        subprocess.check_output(
            [
                "ros2",
                "pkg",
                "prefix",
                "nav2_bringup",
            ],
            text=True,
        ).strip()
    ) / "share" / "nav2_bringup"

    tb3_sim_share = Path(
        subprocess.check_output(
            [
                "ros2",
                "pkg",
                "prefix",
                "nav2_minimal_tb3_sim",
            ],
            text=True,
        ).strip()
    ) / "share" / "nav2_minimal_tb3_sim"

    slam_toolbox_share = Path(
        subprocess.check_output(
            [
                "ros2",
                "pkg",
                "prefix",
                "slam_toolbox",
            ],
            text=True,
        ).strip()
    ) / "share" / "slam_toolbox"

    # ================================================================
    # OFFICIAL TURTLEBOT URDF
    # ================================================================

    robot_urdf = (
        tb3_sim_share
        / "urdf"
        / "turtlebot3_waffle.urdf"
    )

    if not robot_urdf.exists():
        raise FileNotFoundError(
            f"Official TurtleBot URDF does not exist:\n{robot_urdf}"
        )

    with robot_urdf.open(
        "r",
        encoding="utf-8",
    ) as urdf_file:
        robot_description = urdf_file.read()

    # ================================================================
    # EXPAND PROJECT ROBOT SDF
    # ================================================================

    temp_sdf = (
        Path(tempfile.gettempdir())
        / "embodied_ai_gz_waffle.sdf"
    )

    with temp_sdf.open(
        "w",
        encoding="utf-8",
    ) as sdf_file:
        subprocess.run(
            [
                "xacro",
                str(robot_sdf_xacro),
            ],
            check=True,
            stdout=sdf_file,
        )

    # ================================================================
    # LAUNCH ARGUMENT
    # ================================================================

    use_rviz = DeclareLaunchArgument(
        "use_rviz",
        default_value="true",
        description="Start RViz.",
    )

    # ================================================================
    # GAZEBO RESOURCE PATH
    # ================================================================

    gazebo_resource_path = SetEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH",
        value=(
            "/opt/ros/jazzy/share:"
            + os.environ.get(
                "GZ_SIM_RESOURCE_PATH",
                "",
            )
        ),
    )

    # ================================================================
    # GAZEBO + ROBOT SPAWN
    #
    # IMPORTANT:
    #
    # Do not use arbitrary TimerAction delays here.
    #
    # We start Gazebo and explicitly wait for the world service.
    # Then we spawn the robot.
    # Then we explicitly wait until the robot exists.
    #
    # This eliminates the race condition we have been hitting.
    # ================================================================

    world_quoted = shlex.quote(str(world_file))
    sdf_quoted = shlex.quote(str(temp_sdf))

    gazebo_and_spawn_script = f"""
set -u

WORLD={world_quoted}
ROBOT_SDF={sdf_quoted}

echo ""
echo "============================================================"
echo " EMBODIED AI SIMULATION"
echo "============================================================"
echo ""
echo "Starting Gazebo:"
echo "  $WORLD"
echo ""

gz sim -r "$WORLD" &
GZ_PID=$!

cleanup()
{{
    echo ""
    echo "[simulation] Shutting down Gazebo..."
    kill "$GZ_PID" 2>/dev/null || true
}}

trap cleanup EXIT INT TERM

echo "[simulation] Waiting for Gazebo world..."

WORLD_READY=0

for i in $(seq 1 60); do

    if ! kill -0 "$GZ_PID" 2>/dev/null; then
        echo "[simulation] ERROR: Gazebo exited before the world became ready."
        exit 1
    fi

    if gz model --list >/dev/null 2>&1; then
        WORLD_READY=1
        break
    fi

    echo "[simulation] Waiting for Gazebo... ($i/60)"
    sleep 1
done

if [ "$WORLD_READY" -ne 1 ]; then
    echo ""
    echo "[simulation] ERROR: Gazebo world did not become available."
    echo ""
    exit 1
fi

echo ""
echo "[simulation] Gazebo world is READY."
echo ""
echo "[simulation] Spawning TurtleBot3 Waffle..."
echo ""

ros2 run ros_gz_sim create \
    -name turtlebot3_waffle \
    -file "$ROBOT_SDF" \
    -x 0.0 \
    -y -2.3 \
    -z 0.01 \
    -Y 1.5708

SPAWN_RESULT=$?

if [ "$SPAWN_RESULT" -ne 0 ]; then
    echo ""
    echo "[simulation] ERROR: TurtleBot spawn failed."
    echo ""
    exit "$SPAWN_RESULT"
fi

echo ""
echo "[simulation] Spawn command completed."
echo ""
echo "[simulation] Waiting for TurtleBot to appear in Gazebo..."
echo ""

ROBOT_READY=0

for i in $(seq 1 30); do

    if gz model --list 2>/dev/null | grep -q "turtlebot3_waffle"; then
        ROBOT_READY=1
        break
    fi

    echo "[simulation] Waiting for turtlebot3_waffle... ($i/30)"
    sleep 1
done

if [ "$ROBOT_READY" -ne 1 ]; then
    echo ""
    echo "[simulation] ERROR: Robot was not found in Gazebo after spawning."
    echo ""
    echo "Current Gazebo models:"
    gz model --list || true
    echo ""
    exit 1
fi

echo ""
echo "============================================================"
echo " TURTLEBOT3 Waffle is READY in Gazebo"
echo "============================================================"
echo ""

wait "$GZ_PID"
"""

    gazebo_and_spawn = ExecuteProcess(
        cmd=[
            "bash",
            "-c",
            gazebo_and_spawn_script,
        ],
        output="screen",
    )

    # ================================================================
    # ROBOT STATE PUBLISHER
    #
    # Official URDF -> ROS TF.
    # ================================================================

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[
            {
                "use_sim_time": True,
                "robot_description": robot_description,
            }
        ],
        remappings=[
            ("/tf", "tf"),
            ("/tf_static", "tf_static"),
        ],
    )

    # ================================================================
    # GAZEBO -> ROS 2 BRIDGE
    #
    # Start after Gazebo has had a chance to initialize.
    #
    # Unlike robot spawning, the bridge does not need to race against
    # a specific Gazebo entity.
    # ================================================================

    bridge = TimerAction(
        period=4.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    "ros2",
                    "run",
                    "ros_gz_bridge",
                    "parameter_bridge",

                    "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",

                    "/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan",

                    "/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry",

                    "/tf@tf2_msgs/msg/TFMessage@gz.msgs.Pose_V",

                    "/camera/image_raw@sensor_msgs/msg/Image@gz.msgs.Image",
                    
                    "/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",
                ],
                output="screen",
            )
        ],
    )

    # ================================================================
    # SLAM TOOLBOX
    # ================================================================

    slam = TimerAction(
        period=15.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    str(
                        slam_toolbox_share
                        / "launch"
                        / "online_async_launch.py"
                    )
                ),
                launch_arguments={
                    "use_sim_time": "true",
                }.items(),
            )
        ],
    )

    # ================================================================
    # NAV2
    #
    # We use the normal Nav2 navigation launch rather than trying to
    # make the Gazebo launcher also own Nav2.
    # ================================================================

    nav2 = TimerAction(
        period=20.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    str(
                        nav2_bringup_share
                        / "launch"
                        / "navigation_launch.py"
                    )
                ),
                launch_arguments={
                    "use_sim_time": "true",
                    "autostart": "true",
                }.items(),
            )
        ],
    )

    # ================================================================
    # RVIZ
    # ================================================================

    rviz = TimerAction(
        period=23.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    str(
                        nav2_bringup_share
                        / "launch"
                        / "rviz_launch.py"
                    )
                ),
                condition=IfCondition(
                    LaunchConfiguration("use_rviz")
                ),
                launch_arguments={
                    "use_sim_time": "true",
                }.items(),
            )
        ],
    )

    # ================================================================
    # LAUNCH DESCRIPTION
    # ================================================================

    return LaunchDescription(
        [
            use_rviz,

            gazebo_resource_path,

            # Gazebo + deterministic robot spawning
            gazebo_and_spawn,

            # ROS TF
            robot_state_publisher,

            # Gazebo sensors / odometry
            bridge,

            # Mapping
            slam,

            # Navigation
            nav2,

            # Visualization
            rviz,
        ]
    )


if __name__ == "__main__":
    generate_launch_description()