#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener


def pose6(text):
    values = [float(v) for v in (text or "0 0 0 0 0 0").split()]
    if len(values) != 6:
        raise ValueError(f"Invalid SDF pose: {text}")
    return values


def find_model_pose(world_file: Path, requested: str):
    root = ET.parse(world_file).getroot()
    matches = []
    for model in root.findall(".//model"):
        name = model.attrib.get("name", "")
        if requested.lower() in name.lower():
            pose_node = model.find("pose")
            matches.append((name, pose6(pose_node.text if pose_node is not None else None)))
    if not matches:
        names = [m.attrib.get("name", "") for m in root.findall(".//model")]
        raise RuntimeError(f"No model matched '{requested}'. Models: {names}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple models matched '{requested}': {[m[0] for m in matches]}")
    return matches[0]


def world_to_map(x, y, start_x, start_y, start_yaw):
    dx, dy = x - start_x, y - start_y
    c, s = math.cos(start_yaw), math.sin(start_yaw)
    return c * dx + s * dy, -s * dx + c * dy


def normalize(a):
    return math.atan2(math.sin(a), math.cos(a))


class BoxGoalTest(Node):
    def __init__(self, args):
        super().__init__("box_to_nav2_goal_test")
        self.args = args
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.client = ActionClient(self, NavigateToPose, "/navigate_to_pose")

    def robot_pose(self):
        tf = self.tf_buffer.lookup_transform(
            self.args.map_frame, self.args.base_frame, rclpy.time.Time()
        )
        t, q = tf.transform.translation, tf.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        return t.x, t.y, yaw

    def make_goal(self, box_x, box_y):
        robot_x, robot_y, _ = self.robot_pose()
        dx, dy = robot_x - box_x, robot_y - box_y
        distance = math.hypot(dx, dy)
        if distance < 1e-6:
            raise RuntimeError("Robot and box positions are identical.")

        # Goal lies standoff_distance away from the box, on the robot-facing side.
        ux, uy = dx / distance, dy / distance
        gx = box_x + self.args.standoff_distance * ux
        gy = box_y + self.args.standoff_distance * uy
        gyaw = normalize(math.atan2(box_y - gy, box_x - gx))
        return gx, gy, gyaw

    def send(self, x, y, yaw):
        pose = PoseStamped()
        pose.header.frame_id = self.args.map_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x, pose.pose.position.y = x, y
        pose.pose.orientation.z = math.sin(yaw / 2)
        pose.pose.orientation.w = math.cos(yaw / 2)

        goal = NavigateToPose.Goal()
        goal.pose = pose

        self.get_logger().info("Waiting for /navigate_to_pose...")
        if not self.client.wait_for_server(timeout_sec=self.args.timeout):
            raise RuntimeError("/navigate_to_pose action server unavailable")

        future = self.client.send_goal_async(goal, feedback_callback=self.feedback)
        rclpy.spin_until_future_complete(self, future)
        handle = future.result()
        if handle is None or not handle.accepted:
            raise RuntimeError("Nav2 rejected the goal")

        self.get_logger().info("Nav2 accepted goal")
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        status = result_future.result().status
        if status == 4:
            self.get_logger().info("SUCCESS: robot reached box standoff goal")
            return 0
        self.get_logger().error(f"Navigation failed; action status={status}")
        return 1

    def feedback(self, message):
        d = getattr(message.feedback, "distance_remaining", None)
        if d is not None:
            self.get_logger().info(f"Distance remaining: {d:.2f} m")

    def run(self):
        world_file = Path(self.args.world_file).expanduser().resolve()
        name, pose = find_model_pose(world_file, self.args.box_name)
        wx, wy, _, _, _, _ = pose
        bx, by = world_to_map(
            wx, wy,
            self.args.initial_robot_x,
            self.args.initial_robot_y,
            self.args.initial_robot_yaw,
        )

        self.get_logger().info(f"Selected model: {name}")
        self.get_logger().info(f"Box world position: ({wx:.3f}, {wy:.3f})")
        self.get_logger().info(f"Assumed map position: ({bx:.3f}, {by:.3f})")

        deadline = self.get_clock().now().nanoseconds + int(self.args.timeout * 1e9)
        while rclpy.ok():
            try:
                gx, gy, gyaw = self.make_goal(bx, by)
                break
            except Exception as exc:
                if self.get_clock().now().nanoseconds >= deadline:
                    raise RuntimeError(f"TF lookup failed: {exc}") from exc
                rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().info(f"Goal: x={gx:.3f}, y={gy:.3f}, yaw={gyaw:.3f}")
        if self.args.dry_run:
            return 0
        return self.send(gx, gy, gyaw)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--world-file", required=True)
    p.add_argument("--box-name", required=True, help="Unique SDF model name or substring")
    p.add_argument("--map-frame", default="map")
    p.add_argument("--base-frame", default="base_link")
    p.add_argument("--initial-robot-x", type=float, default=0.0)
    p.add_argument("--initial-robot-y", type=float, default=-2.3)
    p.add_argument("--initial-robot-yaw", type=float, default=math.pi / 2)
    p.add_argument("--standoff-distance", type=float, default=0.75)
    p.add_argument("--timeout", type=float, default=20.0)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    rclpy.init()
    node = BoxGoalTest(args)
    try:
        code = node.run()
    except Exception as exc:
        node.get_logger().error(f"TEST FAILED: {exc}")
        code = 1
    finally:
        node.destroy_node()
        rclpy.shutdown()
    sys.exit(code)


if __name__ == "__main__":
    main()
