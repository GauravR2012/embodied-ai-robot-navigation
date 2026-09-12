from __future__ import annotations

import time
from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

from perception.interface import PerceptionBackend


class CameraPerceptionNode(Node):
    """
    ROS 2 camera ingestion node.

    Subscribes to a ROS Image topic, samples frames at a configurable
    rate, and passes selected frames to a perception backend.
    """

    def __init__(
        self,
        perception_backend: PerceptionBackend,
    ) -> None:
        super().__init__("camera_perception_node")

        self.backend = perception_backend
        self.bridge = CvBridge()

        self.declare_parameter(
            "image_topic",
            "/camera/image_raw",
        )

        self.declare_parameter(
            "sample_interval",
            1.0,
        )

        self.declare_parameter(
            "output_directory",
            "/tmp/embodied_ai_frames",
        )

        self.declare_parameter(
            "instruction",
            "Identify relevant objects, obstacles, and possible navigation targets.",
        )

        self.image_topic = (
            self.get_parameter("image_topic")
            .get_parameter_value()
            .string_value
        )

        self.sample_interval = (
            self.get_parameter("sample_interval")
            .get_parameter_value()
            .double_value
        )

        self.output_directory = Path(
            self.get_parameter("output_directory")
            .get_parameter_value()
            .string_value
        )

        self.instruction = (
            self.get_parameter("instruction")
            .get_parameter_value()
            .string_value
        )

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._last_sample_time = 0.0
        self._frame_counter = 0

        self.subscription = self.create_subscription(
            Image,
            self.image_topic,
            self._image_callback,
            10,
        )

        self.get_logger().info(
            f"Subscribed to camera topic: {self.image_topic}"
        )

        self.get_logger().info(
            f"Frame sampling interval: {self.sample_interval:.2f}s"
        )

    def _image_callback(self, msg: Image) -> None:
        now = time.monotonic()

        if now - self._last_sample_time < self.sample_interval:
            return

        self._last_sample_time = now
        self._frame_counter += 1

        try:
            image = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8",
            )
        except Exception as exc:
            self.get_logger().error(
                f"Failed to convert ROS image: {exc}"
            )
            return

        frame_path = (
            self.output_directory
            / f"frame_{self._frame_counter:06d}.jpg"
        )

        try:
            success = cv2.imwrite(
                str(frame_path),
                image,
                [cv2.IMWRITE_JPEG_QUALITY, 90],
            )

            if not success:
                raise RuntimeError(
                    "cv2.imwrite returned False."
                )

        except Exception as exc:
            self.get_logger().error(
                f"Failed to save camera frame: {exc}"
            )
            return

        self.get_logger().info(
            f"Sampled frame: {frame_path}"
        )

        try:
            result = self.backend.analyze(
                image_path=frame_path,
                instruction=self.instruction,
            )
        except Exception as exc:
            self.get_logger().error(
                f"Perception inference failed: {exc}"
            )
            return

        self.get_logger().info(
            f"Perception result: {result}"
        )


def main() -> None:
    raise RuntimeError(
        "CameraPerceptionNode requires a PerceptionBackend. "
        "Construct it from an application entry point."
    )


if __name__ == "__main__":
    main()
