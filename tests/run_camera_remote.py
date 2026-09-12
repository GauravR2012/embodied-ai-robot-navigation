from __future__ import annotations

import rclpy

from ai_client.gateway_client import AIGatewayClient
from ai_client.vlm_client import RemoteVLMClient
from perception.remote_vlm import RemoteVLMPerception
from perception.ros2.camera_node import CameraPerceptionNode


def main() -> None:
    rclpy.init()

    gateway = AIGatewayClient(
        base_url="http://127.0.0.1:8000",
        timeout=60.0,
    )

    print("Checking AI gateway...")
    print("Gateway health:", gateway.health())

    vlm_client = RemoteVLMClient(gateway)
    perception_backend = RemoteVLMPerception(vlm_client)

    node = CameraPerceptionNode(
        perception_backend=perception_backend
    )

    print("Remote camera perception node started.")
    print("Waiting for /camera/image_raw ...")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nStopping remote camera perception...")
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
