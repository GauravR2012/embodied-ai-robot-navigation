from __future__ import annotations

import rclpy

from perception.models import DetectedObject, PerceptionResult
from perception.ros2.camera_node import CameraPerceptionNode


class FakePerceptionBackend:
    def analyze(self, image_path, instruction):
        print(f"[FAKE VLM] Received frame: {image_path}")
        print(f"[FAKE VLM] Instruction: {instruction}")

        return PerceptionResult(
            objects=(
                DetectedObject(
                    label="test object",
                    confidence=0.99,
                ),
            ),
            scene_description="Fake perception result.",
        )


def main():
    rclpy.init()

    node = CameraPerceptionNode(
        perception_backend=FakePerceptionBackend()
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
