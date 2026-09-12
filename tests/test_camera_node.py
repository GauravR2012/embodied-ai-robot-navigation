from __future__ import annotations

from pathlib import Path

import rclpy
from sensor_msgs.msg import Image

from perception.models import DetectedObject, PerceptionResult
from perception.ros2.camera_node import CameraPerceptionNode


class FakePerceptionBackend:
    def __init__(self) -> None:
        self.calls = []

    def analyze(
        self,
        image_path: str | Path,
        instruction: str,
    ) -> PerceptionResult:
        self.calls.append(
            {
                "image_path": Path(image_path),
                "instruction": instruction,
            }
        )

        return PerceptionResult(
            objects=(
                DetectedObject(
                    label="test obstacle",
                    confidence=0.99,
                ),
            ),
            scene_description="Synthetic test scene.",
        )


def test_camera_node_initialization():
    rclpy.init()

    backend = FakePerceptionBackend()
    node = CameraPerceptionNode(backend)

    try:
        assert node.image_topic == "/camera/image_raw"
        assert node.sample_interval == 1.0
        assert node.instruction

    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_camera_node_processes_sampled_image(tmp_path):
    rclpy.init()

    backend = FakePerceptionBackend()
    node = CameraPerceptionNode(backend)

    node.output_directory = tmp_path
    node.sample_interval = 0.0

    try:
        msg = Image()
        msg.header.frame_id = "camera_link"
        msg.height = 2
        msg.width = 2
        msg.encoding = "bgr8"
        msg.step = 6
        msg.data = bytes(
            [
                0, 0, 255,
                0, 255, 0,
                255, 0, 0,
                255, 255, 255,
            ]
        )

        node._image_callback(msg)

        assert len(backend.calls) == 1

        frame_path = backend.calls[0]["image_path"]

        assert frame_path.exists()
        assert frame_path.suffix == ".jpg"
        assert backend.calls[0]["instruction"] == node.instruction

    finally:
        node.destroy_node()
        rclpy.shutdown()
