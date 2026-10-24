#!/usr/bin/env python3

"""
End-to-end RGB-D object navigation test.

Pipeline:

    Gazebo RGB camera
            |
            v
    Remote VLM on Vast.ai
            |
            v
    Object bounding box
            |
            v
    RGB-D depth grounding
            |
            v
    3D point in camera frame
            |
            v
    TF2 -> map
            |
            v
    Standoff navigation pose
            |
            v
    Nav2 NavigateToPose

This test intentionally does NOT use the LLM/task planner yet.

Target:
    "red box"
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import cv2
import numpy as np

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from cv_bridge import CvBridge

from geometry_msgs.msg import PointStamped
from nav2_msgs.action import NavigateToPose
from sensor_msgs.msg import CameraInfo, Image

import tf2_geometry_msgs
import tf2_ros

from ai_client.gateway_client import AIGatewayClient
from ai_client.vlm_client import RemoteVLMClient
from perception.remote_vlm import RemoteVLMPerception


# =============================================================================
# Configuration
# =============================================================================

RGB_TOPIC = "/camera/image_raw"

DEPTH_TOPIC = (
    "/world/embodied_ai_boxes/model/turtlebot3_waffle/"
    "link/camera_link/sensor/intel_realsense_r200_depth/"
    "depth_image"
)

CAMERA_INFO_TOPIC = (
    "/world/embodied_ai_boxes/model/turtlebot3_waffle/"
    "link/camera_link/sensor/intel_realsense_r200_depth/"
    "camera_info"
)

DEPTH_FRAME = "camera_depth_frame"
MAP_FRAME = "map"
BASE_FRAME = "base_link"

TARGET_LABEL = "red box"

# Distance at which the robot should stop from the object.
STANDOFF_DISTANCE = 0.8

# Existing local AI gateway.
#
# This is the same gateway used by run_camera_remote.py.
VLM_GATEWAY_URL = "http://127.0.0.1:8000"

VLM_TIMEOUT = 60.0

# Where the RGB frame sent to the VLM is saved.
FRAME_DIRECTORY = Path("/tmp/embodied_ai_frames")

# Sensor wait timeout.
SENSOR_TIMEOUT = 30.0

# TF lookup timeout.
TF_TIMEOUT = 5.0


# =============================================================================
# VLM instruction
# =============================================================================

VLM_PROMPT = """
You are an object localization system.

Find the object described below in the image.

Target:
red box

Return ONLY valid JSON.

Required format:

{
  "scene_description": "string",
  "objects": [
    {
      "label": "red box",
      "confidence": 0.0,
      "bbox": {
        "x_min": 0,
        "y_min": 0,
        "x_max": 0,
        "y_max": 0
      },
      "attributes": {}
    }
  ]
}

Bounding box coordinates must be pixel coordinates in the original image.

The original image resolution is 640x480.

The bounding box must satisfy:

x_min < x_max
y_min < y_max

If the target is visible, include its bounding box.

If the target is not visible, return:

{
  "scene_description": "target not visible",
  "objects": []
}

Do not include markdown.
Do not include explanations outside the JSON.
"""


# =============================================================================
# ROS 2 Node
# =============================================================================

class VLMRGBDNavigationTest(Node):

    def __init__(self) -> None:
        super().__init__("vlm_rgbd_navigation_test")

        self.bridge = CvBridge()

        # ---------------------------------------------------------------------
        # Latest sensor messages
        # ---------------------------------------------------------------------

        self.latest_rgb: Image | None = None
        self.latest_depth: Image | None = None
        self.camera_info: CameraInfo | None = None

        # ---------------------------------------------------------------------
        # Existing project VLM architecture
        #
        # Local:
        #     AIGatewayClient
        #
        #         ↓
        #
        #     RemoteVLMClient
        #
        #         ↓
        #
        #     RemoteVLMPerception
        #
        #         ↓
        #
        #     Vast.ai VLM
        # ---------------------------------------------------------------------

        self.gateway = AIGatewayClient(
            base_url=VLM_GATEWAY_URL,
            timeout=VLM_TIMEOUT,
        )

        self.vlm_client = RemoteVLMClient(
            self.gateway,
        )

        self.perception = RemoteVLMPerception(
            self.vlm_client,
        )

        # ---------------------------------------------------------------------
        # ROS subscriptions
        # ---------------------------------------------------------------------

        self.create_subscription(
            Image,
            RGB_TOPIC,
            self.rgb_callback,
            10,
        )

        self.create_subscription(
            Image,
            DEPTH_TOPIC,
            self.depth_callback,
            10,
        )

        self.create_subscription(
            CameraInfo,
            CAMERA_INFO_TOPIC,
            self.camera_info_callback,
            10,
        )

        # ---------------------------------------------------------------------
        # TF2
        # ---------------------------------------------------------------------

        self.tf_buffer = tf2_ros.Buffer()

        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer,
            self,
        )

        # ---------------------------------------------------------------------
        # Nav2
        # ---------------------------------------------------------------------

        self.nav_client = ActionClient(
            self,
            NavigateToPose,
            "/navigate_to_pose",
        )

        # ---------------------------------------------------------------------
        # Frame storage
        # ---------------------------------------------------------------------

        FRAME_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.get_logger().info(
            "VLM RGB-D navigation test initialized."
        )

    # =========================================================================
    # ROS callbacks
    # =========================================================================

    def rgb_callback(
        self,
        msg: Image,
    ) -> None:
        self.latest_rgb = msg

    def depth_callback(
        self,
        msg: Image,
    ) -> None:
        self.latest_depth = msg

    def camera_info_callback(
        self,
        msg: CameraInfo,
    ) -> None:
        self.camera_info = msg

    # =========================================================================
    # Sensor synchronization / availability
    # =========================================================================

    def wait_for_sensor_data(
        self,
        timeout: float = SENSOR_TIMEOUT,
    ) -> bool:

        self.get_logger().info(
            "Waiting for RGB, depth and CameraInfo..."
        )

        start = time.monotonic()

        while rclpy.ok():

            rclpy.spin_once(
                self,
                timeout_sec=0.1,
            )

            if (
                self.latest_rgb is not None
                and self.latest_depth is not None
                and self.camera_info is not None
            ):
                self.get_logger().info(
                    "RGB-D sensor data available."
                )

                return True

            if time.monotonic() - start > timeout:
                return False

        return False

    # =========================================================================
    # Gateway health
    # =========================================================================

    def check_gateway(self) -> None:

        self.get_logger().info(
            "Checking AI gateway..."
        )

        health = self.gateway.health()

        self.get_logger().info(
            f"Gateway health: {health}"
        )

    # =========================================================================
    # Save RGB frame
    # =========================================================================

    def save_rgb_frame(
        self,
        image_msg: Image,
    ) -> Path:

        image = self.bridge.imgmsg_to_cv2(
            image_msg,
            desired_encoding="bgr8",
        )

        timestamp = int(
            time.time() * 1000
        )

        frame_path = (
            FRAME_DIRECTORY
            / f"vlm_e2e_{timestamp}.jpg"
        )

        success = cv2.imwrite(
            str(frame_path),
            image,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                90,
            ],
        )

        if not success:
            raise RuntimeError(
                f"Failed to save RGB frame: {frame_path}"
            )

        self.get_logger().info(
            f"Saved RGB frame: {frame_path}"
        )

        return frame_path

    # =========================================================================
    # Remote VLM
    # =========================================================================

    def call_remote_vlm(
        self,
        image_msg: Image,
    ):
        """
        Send the RGB image through the project's existing
        RemoteVLMPerception interface.

        No manual HTTP request.
        No manual JSON parsing.
        """

        frame_path = self.save_rgb_frame(
            image_msg,
        )

        self.get_logger().info(
            "Sending RGB frame to remote VLM..."
        )

        result = self.perception.analyze(
            image_path=frame_path,
            instruction=VLM_PROMPT,
        )

        self.get_logger().info(
            f"VLM scene description: "
            f"{result.scene_description}"
        )

        self.get_logger().info(
            f"VLM detected {len(result.objects)} object(s)."
        )

        for obj in result.objects:

            self.get_logger().info(
                f"  Object: {obj.label}"
            )

            if obj.confidence is not None:
                self.get_logger().info(
                    f"  Confidence: {obj.confidence:.3f}"
                )

            if obj.bbox is not None:
                self.get_logger().info(
                    "  BBox: "
                    f"x_min={obj.bbox.x_min:.1f}, "
                    f"y_min={obj.bbox.y_min:.1f}, "
                    f"x_max={obj.bbox.x_max:.1f}, "
                    f"y_max={obj.bbox.y_max:.1f}"
                )

        return result

    # =========================================================================
    # Find target object
    # =========================================================================

    def find_target_object(
        self,
        perception_result,
    ):

        target = None

        for obj in perception_result.objects:

            label = obj.label.lower().strip()

            if (
                "red" in label
                and "box" in label
                and obj.bbox is not None
            ):
                target = obj
                break

        if target is None:

            # Fallback:
            # allow "box" if the model placed the color
            # inside the attributes.

            for obj in perception_result.objects:

                label = obj.label.lower().strip()

                attributes = {
                    str(k).lower(): str(v).lower()
                    for k, v in obj.attributes.items()
                }

                has_box = "box" in label
                has_red = (
                    "red" in label
                    or "red" in attributes.values()
                    or "red" in attributes.keys()
                )

                if (
                    has_box
                    and has_red
                    and obj.bbox is not None
                ):
                    target = obj
                    break

        if target is None:

            raise RuntimeError(
                f"Target '{TARGET_LABEL}' was not detected "
                "with a valid bounding box.\n"
                f"VLM result: {perception_result}"
            )

        self.get_logger().info(
            f"Target detected: {target.label}"
        )

        if target.confidence is not None:

            self.get_logger().info(
                f"Target confidence: "
                f"{target.confidence:.3f}"
            )

        self.get_logger().info(
            "Target bbox: "
            f"x_min={target.bbox.x_min:.1f}, "
            f"y_min={target.bbox.y_min:.1f}, "
            f"x_max={target.bbox.x_max:.1f}, "
            f"y_max={target.bbox.y_max:.1f}"
        )

        return target

    # =========================================================================
    # RGB-D grounding
    # =========================================================================

    def get_object_3d_point(
        self,
        bbox,
    ):

        if self.camera_info is None:
            raise RuntimeError(
                "CameraInfo unavailable."
            )

        if self.latest_depth is None:
            raise RuntimeError(
                "Depth image unavailable."
            )

        if self.latest_rgb is None:
            raise RuntimeError(
                "RGB image unavailable."
            )

        # ---------------------------------------------------------------------
        # Convert depth ROS image to NumPy
        # ---------------------------------------------------------------------

        depth = self.bridge.imgmsg_to_cv2(
            self.latest_depth,
            desired_encoding="passthrough",
        )

        depth = np.asarray(depth)

        if depth.ndim != 2:
            raise RuntimeError(
                f"Expected single-channel depth image, "
                f"got shape {depth.shape}."
            )

        # ---------------------------------------------------------------------
        # RGB bbox
        #
        # VLM bbox is in RGB pixel coordinates.
        # ---------------------------------------------------------------------

        xmin = float(bbox.x_min)
        ymin = float(bbox.y_min)
        xmax = float(bbox.x_max)
        ymax = float(bbox.y_max)

        # ---------------------------------------------------------------------
        # RGB → depth resolution scaling
        #
        # Current simulation:
        #
        # RGB   = 640 × 480
        # Depth = 320 × 240
        # ---------------------------------------------------------------------

        rgb_width = float(
            self.latest_rgb.width
        )

        rgb_height = float(
            self.latest_rgb.height
        )

        depth_height, depth_width = depth.shape[:2]

        sx = depth_width / rgb_width
        sy = depth_height / rgb_height

        dxmin = int(
            round(xmin * sx)
        )

        dymin = int(
            round(ymin * sy)
        )

        dxmax = int(
            round(xmax * sx)
        )

        dymax = int(
            round(ymax * sy)
        )

        # Clamp bbox to depth image.

        dxmin = max(
            0,
            min(depth_width - 1, dxmin),
        )

        dymin = max(
            0,
            min(depth_height - 1, dymin),
        )

        dxmax = max(
            0,
            min(depth_width, dxmax),
        )

        dymax = max(
            0,
            min(depth_height, dymax),
        )

        if dxmax <= dxmin or dymax <= dymin:
            raise RuntimeError(
                "Projected RGB bounding box is invalid "
                f"for depth image: "
                f"({dxmin}, {dymin}, {dxmax}, {dymax})"
            )

        # ---------------------------------------------------------------------
        # Center of bbox in depth coordinates
        # ---------------------------------------------------------------------

        depth_u = (
            dxmin + dxmax
        ) / 2.0

        depth_v = (
            dymin + dymax
        ) / 2.0

        # ---------------------------------------------------------------------
        # Use central bbox patch.
        #
        # This avoids depending on one potentially noisy depth pixel.
        # ---------------------------------------------------------------------

        radius_x = max(
            2,
            int(
                (dxmax - dxmin) * 0.15
            ),
        )

        radius_y = max(
            2,
            int(
                (dymax - dymin) * 0.15
            ),
        )

        center_x = int(
            round(depth_u)
        )

        center_y = int(
            round(depth_v)
        )

        x0 = max(
            0,
            center_x - radius_x,
        )

        x1 = min(
            depth_width,
            center_x + radius_x + 1,
        )

        y0 = max(
            0,
            center_y - radius_y,
        )

        y1 = min(
            depth_height,
            center_y + radius_y + 1,
        )

        patch = depth[
            y0:y1,
            x0:x1,
        ]

        # ---------------------------------------------------------------------
        # Filter invalid depth
        # ---------------------------------------------------------------------

        valid = patch[
            np.isfinite(patch)
            & (patch > 0.05)
            & (patch < 10.0)
        ]

        if valid.size == 0:

            raise RuntimeError(
                "No valid depth values inside "
                "target bounding-box center."
            )

        depth_value = float(
            np.median(valid)
        )

        # ---------------------------------------------------------------------
        # Convert depth to metres.
        #
        # Current Gazebo depth is 32FC1, therefore already metres.
        #
        # Also support 16UC1 / mono16 sensors where depth is millimetres.
        # ---------------------------------------------------------------------

        if self.latest_depth.encoding in (
            "16UC1",
            "mono16",
        ):
            depth_m = (
                depth_value / 1000.0
            )
        else:
            depth_m = depth_value

        # ---------------------------------------------------------------------
        # Camera intrinsics
        # ---------------------------------------------------------------------

        fx = float(
            self.camera_info.k[0]
        )

        fy = float(
            self.camera_info.k[4]
        )

        camera_cx = float(
            self.camera_info.k[2]
        )

        camera_cy = float(
            self.camera_info.k[5]
        )

        if fx <= 0.0 or fy <= 0.0:
            raise RuntimeError(
                "Invalid CameraInfo focal lengths."
            )

        # ---------------------------------------------------------------------
        # Convert depth pixel back to RGB-equivalent pixel coordinates.
        #
        # CameraInfo belongs to the depth camera, so we need the depth
        # pixel coordinates for the projection.
        # ---------------------------------------------------------------------

        u = depth_u
        v = depth_v

        # ---------------------------------------------------------------------
        # Pinhole back-projection
        #
        # X = (u - cx) * Z / fx
        # Y = (v - cy) * Z / fy
        # Z = depth
        # ---------------------------------------------------------------------

        X = (
            (u - camera_cx)
            * depth_m
            / fx
        )

        Y = (
            (v - camera_cy)
            * depth_m
            / fy
        )

        Z = depth_m

        self.get_logger().info(
            "RGB-D grounding:"
        )

        self.get_logger().info(
            f"  RGB bbox: "
            f"({xmin:.1f}, {ymin:.1f}) -> "
            f"({xmax:.1f}, {ymax:.1f})"
        )

        self.get_logger().info(
            f"  Depth pixel: "
            f"({u:.1f}, {v:.1f})"
        )

        self.get_logger().info(
            f"  Depth: {depth_m:.3f} m"
        )

        self.get_logger().info(
            "  Camera point: "
            f"X={X:.3f}, "
            f"Y={Y:.3f}, "
            f"Z={Z:.3f}"
        )

        return X, Y, Z

    # =========================================================================
    # Camera frame -> map frame
    # =========================================================================

    def transform_point_to_map(
        self,
        point_xyz,
        stamp,
    ):

        X, Y, Z = point_xyz

        point = PointStamped()

        point.header.frame_id = DEPTH_FRAME
        point.header.stamp = stamp

        point.point.x = X
        point.point.y = Y
        point.point.z = Z

        self.get_logger().info(
            f"Waiting for TF: "
            f"{DEPTH_FRAME} -> {MAP_FRAME}"
        )

        transform = self.tf_buffer.lookup_transform(
            MAP_FRAME,
            DEPTH_FRAME,
            rclpy.time.Time(),
            timeout=rclpy.duration.Duration(
                seconds=TF_TIMEOUT,
            ),
        )

        transformed = tf2_geometry_msgs.do_transform_point(
            point,
            transform,
        )

        map_x = transformed.point.x
        map_y = transformed.point.y
        map_z = transformed.point.z

        self.get_logger().info(
            "Object position in map:"
        )

        self.get_logger().info(
            f"  x = {map_x:.3f}"
        )

        self.get_logger().info(
            f"  y = {map_y:.3f}"
        )

        self.get_logger().info(
            f"  z = {map_z:.3f}"
        )

        return map_x, map_y

    # =========================================================================
    # Compute standoff pose
    # =========================================================================

    def compute_standoff_pose(
        self,
        object_x: float,
        object_y: float,
    ):

        # ---------------------------------------------------------------------
        # Current robot pose from TF.
        # ---------------------------------------------------------------------

        transform = self.tf_buffer.lookup_transform(
            MAP_FRAME,
            BASE_FRAME,
            rclpy.time.Time(),
            timeout=rclpy.duration.Duration(
                seconds=TF_TIMEOUT,
            ),
        )

        robot_x = (
            transform.transform.translation.x
        )

        robot_y = (
            transform.transform.translation.y
        )

        # ---------------------------------------------------------------------
        # Robot -> object vector
        # ---------------------------------------------------------------------

        dx = object_x - robot_x
        dy = object_y - robot_y

        distance = math.hypot(
            dx,
            dy,
        )

        if distance < 0.05:

            raise RuntimeError(
                "Object is too close to robot."
            )

        # ---------------------------------------------------------------------
        # Normalize direction.
        # ---------------------------------------------------------------------

        ux = dx / distance
        uy = dy / distance

        # ---------------------------------------------------------------------
        # Put goal STANDOFF_DISTANCE metres before object.
        # ---------------------------------------------------------------------

        goal_x = (
            object_x
            - STANDOFF_DISTANCE * ux
        )

        goal_y = (
            object_y
            - STANDOFF_DISTANCE * uy
        )

        # ---------------------------------------------------------------------
        # Robot should face object.
        # ---------------------------------------------------------------------

        yaw = math.atan2(
            object_y - goal_y,
            object_x - goal_x,
        )

        self.get_logger().info(
            "Standoff pose:"
        )

        self.get_logger().info(
            f"  robot = "
            f"({robot_x:.3f}, {robot_y:.3f})"
        )

        self.get_logger().info(
            f"  object = "
            f"({object_x:.3f}, {object_y:.3f})"
        )

        self.get_logger().info(
            f"  distance = "
            f"{distance:.3f} m"
        )

        self.get_logger().info(
            f"  goal = "
            f"({goal_x:.3f}, {goal_y:.3f})"
        )

        self.get_logger().info(
            f"  yaw = {yaw:.3f} rad"
        )

        return goal_x, goal_y, yaw

    # =========================================================================
    # Nav2
    # =========================================================================

    def navigate_to_pose(
        self,
        x: float,
        y: float,
        yaw: float,
    ):

        self.get_logger().info(
            "Waiting for Nav2 NavigateToPose..."
        )

        if not self.nav_client.wait_for_server(
            timeout_sec=10.0,
        ):
            raise RuntimeError(
                "Nav2 NavigateToPose action server unavailable."
            )

        goal = NavigateToPose.Goal()

        goal.pose.header.frame_id = MAP_FRAME

        goal.pose.header.stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )

        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.position.z = 0.0

        # Yaw -> quaternion.
        goal.pose.pose.orientation.x = 0.0
        goal.pose.pose.orientation.y = 0.0

        goal.pose.pose.orientation.z = math.sin(
            yaw / 2.0
        )

        goal.pose.pose.orientation.w = math.cos(
            yaw / 2.0
        )

        self.get_logger().info(
            "Sending Nav2 goal:"
        )

        self.get_logger().info(
            f"  x = {x:.3f}"
        )

        self.get_logger().info(
            f"  y = {y:.3f}"
        )

        self.get_logger().info(
            f"  yaw = {yaw:.3f}"
        )

        future = (
            self.nav_client
            .send_goal_async(goal)
        )

        rclpy.spin_until_future_complete(
            self,
            future,
        )

        goal_handle = future.result()

        if goal_handle is None:
            raise RuntimeError(
                "Nav2 goal submission failed."
            )

        if not goal_handle.accepted:
            raise RuntimeError(
                "Nav2 rejected the goal."
            )

        self.get_logger().info(
            "Nav2 goal accepted."
        )

        result_future = (
            goal_handle
            .get_result_async()
        )

        rclpy.spin_until_future_complete(
            self,
            result_future,
        )

        result = result_future.result()

        if result is None:
            raise RuntimeError(
                "No Nav2 result received."
            )

        status = result.status

        self.get_logger().info(
            f"Nav2 result status: {status}"
        )

        return status

    # =========================================================================
    # Main test
    # =========================================================================

    def run_test(self) -> None:

        self.get_logger().info(
            "============================================"
        )

        self.get_logger().info(
            " VLM → RGB-D → TF2 → STANDOFF → NAV2"
        )

        self.get_logger().info(
            f" Target: {TARGET_LABEL}"
        )

        self.get_logger().info(
            "============================================"
        )

        # ---------------------------------------------------------------------
        # 1. Check remote AI gateway
        # ---------------------------------------------------------------------

        self.check_gateway()

        # ---------------------------------------------------------------------
        # 2. Wait for sensors
        # ---------------------------------------------------------------------

        if not self.wait_for_sensor_data():
            raise RuntimeError(
                "RGB-D sensor data did not become available."
            )

        # ---------------------------------------------------------------------
        # 3. Snapshot latest sensor data
        # ---------------------------------------------------------------------

        rgb_msg = self.latest_rgb

        depth_msg = self.latest_depth

        if rgb_msg is None or depth_msg is None:
            raise RuntimeError(
                "Sensor data disappeared unexpectedly."
            )

        self.get_logger().info(
            f"RGB frame: "
            f"{rgb_msg.width}x{rgb_msg.height}"
        )

        self.get_logger().info(
            f"Depth frame: "
            f"{depth_msg.width}x{depth_msg.height}"
        )

        self.get_logger().info(
            f"Depth encoding: "
            f"{depth_msg.encoding}"
        )

        self.get_logger().info(
            f"Depth frame ID: "
            f"{depth_msg.header.frame_id}"
        )

        # ---------------------------------------------------------------------
        # 4. VLM
        # ---------------------------------------------------------------------

        perception_result = (
            self.call_remote_vlm(
                rgb_msg,
            )
        )

        # ---------------------------------------------------------------------
        # 5. Find target
        # ---------------------------------------------------------------------

        target = self.find_target_object(
            perception_result,
        )

        bbox = target.bbox

        if bbox is None:
            raise RuntimeError(
                "Target does not have a bounding box."
            )

        # ---------------------------------------------------------------------
        # 6. RGB-D grounding
        # ---------------------------------------------------------------------

        point_xyz = (
            self.get_object_3d_point(
                bbox,
            )
        )

        # ---------------------------------------------------------------------
        # 7. TF2 -> map
        #
        # Use depth timestamp because the 3D point comes from the depth
        # image.
        # ---------------------------------------------------------------------

        object_x, object_y = (
            self.transform_point_to_map(
                point_xyz,
                depth_msg.header.stamp,
            )
        )

        # ---------------------------------------------------------------------
        # 8. Compute standoff pose
        # ---------------------------------------------------------------------

        goal_x, goal_y, yaw = (
            self.compute_standoff_pose(
                object_x,
                object_y,
            )
        )

        # ---------------------------------------------------------------------
        # 9. Navigate
        # ---------------------------------------------------------------------

        status = self.navigate_to_pose(
            goal_x,
            goal_y,
            yaw,
        )

        # ---------------------------------------------------------------------
        # Complete
        # ---------------------------------------------------------------------

        self.get_logger().info(
            "============================================"
        )

        self.get_logger().info(
            " TEST COMPLETE"
        )

        self.get_logger().info(
            f" Target: {TARGET_LABEL}"
        )

        self.get_logger().info(
            " Object map position: "
            f"({object_x:.3f}, {object_y:.3f})"
        )

        self.get_logger().info(
            " Navigation goal: "
            f"({goal_x:.3f}, {goal_y:.3f})"
        )

        self.get_logger().info(
            f" Nav2 status: {status}"
        )

        self.get_logger().info(
            "============================================"
        )


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:

    rclpy.init()

    node = VLMRGBDNavigationTest()

    try:

        node.run_test()

    except KeyboardInterrupt:

        node.get_logger().info(
            "Interrupted."
        )

    except Exception as exc:

        node.get_logger().error(
            f"TEST FAILED: {exc}"
        )

    finally:

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()