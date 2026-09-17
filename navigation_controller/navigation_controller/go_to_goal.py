
#!/usr/bin/env python3

import math
import time
from typing import Optional

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan


def clamp(value: float, minimum: float, maximum: float) -> float:
    """Limit a value to a specified range."""
    return max(minimum, min(maximum, value))


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def quaternion_to_yaw(
    x: float,
    y: float,
    z: float,
    w: float
) -> float:
    """Convert quaternion orientation to planar yaw."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)

    return math.atan2(siny_cosp, cosy_cosp)


class GoToGoal(Node):

    def __init__(self):
        super().__init__("go_to_goal")

        # --------------------------------------------------
        # ROS 2 PARAMETERS
        # --------------------------------------------------

        self.declare_parameter("goal_x", 3.0)
        self.declare_parameter("goal_y", 0.0)
        self.declare_parameter("goal_yaw", 0.0)

        self.declare_parameter("position_tolerance", 0.08)
        self.declare_parameter("yaw_tolerance_degrees", 5.0)

        self.declare_parameter("linear_kp", 0.5)
        self.declare_parameter("angular_kp", 1.5)

        self.declare_parameter("max_linear_velocity", 0.20)
        self.declare_parameter("max_angular_velocity", 0.50)

        self.declare_parameter("control_frequency", 20.0)
        self.declare_parameter("goal_timeout", 120.0)

        # --------------------------------------------------
        # LOAD PARAMETERS
        # --------------------------------------------------

        self.goal_x = float(
            self.get_parameter("goal_x").value
        )

        self.goal_y = float(
            self.get_parameter("goal_y").value
        )

        self.goal_yaw = math.radians(
            float(self.get_parameter("goal_yaw").value)
        )

        self.position_tolerance = float(
            self.get_parameter("position_tolerance").value
        )

        self.yaw_tolerance = math.radians(
            float(self.get_parameter("yaw_tolerance_degrees").value)
        )

        self.linear_kp = float(
            self.get_parameter("linear_kp").value
        )

        self.angular_kp = float(
            self.get_parameter("angular_kp").value
        )

        self.max_linear_velocity = float(
            self.get_parameter("max_linear_velocity").value
        )

        self.max_angular_velocity = float(
            self.get_parameter("max_angular_velocity").value
        )

        control_frequency = float(
            self.get_parameter("control_frequency").value
        )

        self.goal_timeout = float(
            self.get_parameter("goal_timeout").value
        )

        # --------------------------------------------------
        # ROBOT STATE
        # --------------------------------------------------

        self.x: Optional[float] = None
        self.y: Optional[float] = None
        self.yaw: Optional[float] = None

        self.previous_x: Optional[float] = None
        self.previous_y: Optional[float] = None

        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

        self.path_length = 0.0
        self.minimum_obstacle_distance = float("inf")

        self.goal_reached = False
        self.timeout_reached = False
        self.shutdown_requested = False

        # --------------------------------------------------
        # ROS 2 PUBLISHERS
        # --------------------------------------------------

        self.cmd_pub = self.create_publisher(
            TwistStamped,
            "/cmd_vel",
            10
        )

        # --------------------------------------------------
        # ROS 2 SUBSCRIBERS
        # --------------------------------------------------

        self.odom_sub = self.create_subscription(
            Odometry,
            "/odom",
            self.odom_callback,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            "/scan",
            self.scan_callback,
            10
        )

        # --------------------------------------------------
        # CONTROL TIMER
        # --------------------------------------------------

        timer_period = 1.0 / control_frequency

        self.timer = self.create_timer(
            timer_period,
            self.control_loop
        )

        self.get_logger().info(
            "Go-to-goal controller initialized."
        )

        self.get_logger().info(
            f"Goal: x={self.goal_x:.3f}, "
            f"y={self.goal_y:.3f}, "
            f"yaw={math.degrees(self.goal_yaw):.2f} degrees"
        )

    # ======================================================
    # ODOMETRY CALLBACK
    # ======================================================

    def odom_callback(self, msg: Odometry):

        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation

        current_x = position.x
        current_y = position.y

        current_yaw = quaternion_to_yaw(
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w
        )

        # Initialize state.
        if self.start_time is None:
            self.start_time = time.monotonic()

            self.previous_x = current_x
            self.previous_y = current_y

        # Calculate incremental path length.
        if self.previous_x is not None and self.previous_y is not None:

            delta_x = current_x - self.previous_x
            delta_y = current_y - self.previous_y

            displacement = math.sqrt(
                delta_x * delta_x +
                delta_y * delta_y
            )

            # Ignore unreasonable odometry jumps.
            if displacement < 1.0:
                self.path_length += displacement

        self.x = current_x
        self.y = current_y
        self.yaw = current_yaw

        self.previous_x = current_x
        self.previous_y = current_y

    # ======================================================
    # LASER SCAN CALLBACK
    # ======================================================

    def scan_callback(self, msg: LaserScan):

        valid_ranges = [
            distance
            for distance in msg.ranges
            if math.isfinite(distance)
            and msg.range_min <= distance <= msg.range_max
        ]

        if valid_ranges:
            current_minimum = min(valid_ranges)

            self.minimum_obstacle_distance = min(
                self.minimum_obstacle_distance,
                current_minimum
            )

    # ======================================================
    # VELOCITY COMMAND
    # ======================================================

    def publish_velocity(
        self,
        linear_x: float,
        angular_z: float
    ):

        msg = TwistStamped()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"

        msg.twist.linear.x = linear_x
        msg.twist.linear.y = 0.0
        msg.twist.linear.z = 0.0

        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = angular_z

        self.cmd_pub.publish(msg)

    # ======================================================
    # STOP ROBOT
    # ======================================================

    def stop_robot(self):

        self.publish_velocity(
            linear_x=0.0,
            angular_z=0.0
        )

    # ======================================================
    # METRICS
    # ======================================================

    def calculate_metrics(
        self,
        position_error: float,
        yaw_error: float
    ):

        if self.start_time is None:
            elapsed_time = 0.0
        else:
            end_time = self.end_time or time.monotonic()
            elapsed_time = end_time - self.start_time

        minimum_obstacle_distance = (
            self.minimum_obstacle_distance
            if math.isfinite(self.minimum_obstacle_distance)
            else None
        )

        self.get_logger().info(
            "========== NAVIGATION METRICS =========="
        )

        self.get_logger().info(
            f"Success: {self.goal_reached}"
        )

        self.get_logger().info(
            f"Timeout: {self.timeout_reached}"
        )

        self.get_logger().info(
            f"Elapsed time: {elapsed_time:.3f} seconds"
        )

        self.get_logger().info(
            f"Path length: {self.path_length:.3f} meters"
        )

        self.get_logger().info(
            f"Final position: "
            f"x={self.x:.3f}, y={self.y:.3f}"
        )

        self.get_logger().info(
            f"Final position error: "
            f"{position_error:.3f} meters"
        )

        self.get_logger().info(
            f"Final yaw error: "
            f"{math.degrees(abs(yaw_error)):.3f} degrees"
        )

        if minimum_obstacle_distance is not None:
            self.get_logger().info(
                f"Minimum obstacle distance: "
                f"{minimum_obstacle_distance:.3f} meters"
            )
        else:
            self.get_logger().info(
                "Minimum obstacle distance: unavailable"
            )

        self.get_logger().info(
            "========================================="
        )

    # ======================================================
    # GOAL COMPLETION
    # ======================================================

    def finish_navigation(
        self,
        position_error: float,
        yaw_error: float
    ):

        if self.goal_reached or self.timeout_reached:
            return

        self.goal_reached = True
        self.end_time = time.monotonic()

        self.stop_robot()

        self.get_logger().info(
            "Goal reached successfully."
        )

        self.calculate_metrics(
            position_error,
            yaw_error
        )

    # ======================================================
    # TIMEOUT
    # ======================================================

    def check_timeout(self) -> bool:

        if self.start_time is None:
            return False

        elapsed_time = time.monotonic() - self.start_time

        if elapsed_time >= self.goal_timeout:

            self.timeout_reached = True
            self.end_time = time.monotonic()

            self.stop_robot()

            self.get_logger().warning(
                f"Navigation timeout after "
                f"{elapsed_time:.3f} seconds."
            )

            if self.x is not None and self.y is not None:

                dx = self.goal_x - self.x
                dy = self.goal_y - self.y

                position_error = math.sqrt(
                    dx * dx + dy * dy
                )

                yaw_error = normalize_angle(
                    self.goal_yaw - self.yaw
                )

                self.calculate_metrics(
                    position_error,
                    yaw_error
                )

            return True

        return False

    # ======================================================
    # CONTROL LOOP
    # ======================================================

    def control_loop(self):

        if self.x is None or self.yaw is None:
            return

        if self.goal_reached or self.timeout_reached:
            self.stop_robot()
            return

        if self.check_timeout():
            return

        # --------------------------------------------------
        # POSITION ERROR
        # --------------------------------------------------

        dx = self.goal_x - self.x
        dy = self.goal_y - self.y

        distance_error = math.sqrt(
            dx * dx +
            dy * dy
        )

        # --------------------------------------------------
        # HEADING ERROR
        # --------------------------------------------------

        target_heading = math.atan2(dy, dx)

        heading_error = normalize_angle(
            target_heading - self.yaw
        )

        final_yaw_error = normalize_angle(
            self.goal_yaw - self.yaw
        )

        # --------------------------------------------------
        # STATE 1: POSITION REACHED
        # --------------------------------------------------

        if distance_error <= self.position_tolerance:

            # Align with desired final orientation.
            if abs(final_yaw_error) > self.yaw_tolerance:

                angular_velocity = (
                    self.angular_kp * final_yaw_error
                )

                angular_velocity = clamp(
                    angular_velocity,
                    -self.max_angular_velocity,
                    self.max_angular_velocity
                )

                self.publish_velocity(
                    linear_x=0.0,
                    angular_z=angular_velocity
                )

                return

            # Position and orientation reached.
            self.finish_navigation(
                position_error=distance_error,
                yaw_error=final_yaw_error
            )

            return

        # --------------------------------------------------
        # STATE 2: ROTATE TOWARD GOAL
        # --------------------------------------------------

        rotate_threshold = math.radians(20.0)

        if abs(heading_error) > rotate_threshold:

            angular_velocity = (
                self.angular_kp * heading_error
            )

            angular_velocity = clamp(
                angular_velocity,
                -self.max_angular_velocity,
                self.max_angular_velocity
            )

            self.publish_velocity(
                linear_x=0.0,
                angular_z=angular_velocity
            )

            return

        # --------------------------------------------------
        # STATE 3: MOVE TOWARD GOAL
        # --------------------------------------------------

        linear_velocity = (
            self.linear_kp * distance_error
        )

        linear_velocity = clamp(
            linear_velocity,
            0.0,
            self.max_linear_velocity
        )

        # Reduce linear speed when heading error increases.
        heading_factor = max(
            0.0,
            math.cos(heading_error)
        )

        linear_velocity *= heading_factor

        angular_velocity = (
            self.angular_kp * heading_error
        )

        angular_velocity = clamp(
            angular_velocity,
            -self.max_angular_velocity,
            self.max_angular_velocity
        )

        self.publish_velocity(
            linear_x=linear_velocity,
            angular_z=angular_velocity
        )

    # ======================================================
    # SHUTDOWN
    # ======================================================

    def destroy_node(self):

        self.stop_robot()

        self.get_logger().info(
            "Robot stopped. Shutting down controller."
        )

        super().destroy_node()


def main(args=None):

    rclpy.init(args=args)

    node = GoToGoal()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.get_logger().info(
            "Keyboard interrupt received."
        )

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()