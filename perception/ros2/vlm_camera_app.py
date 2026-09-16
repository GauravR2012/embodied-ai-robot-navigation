from __future__ import annotations

import rclpy

from ai_client.gateway_client import AIGatewayClient
from ai_client.vlm_client import RemoteVLMClient

from perception.remote_vlm import RemoteVLMPerception

from .camera_node import CameraPerceptionNode


def main() -> None:
    """
    Run the ROS 2 camera perception application.

    Composition:

        AIGatewayClient
            -> RemoteVLMClient
            -> RemoteVLMPerception
            -> CameraPerceptionNode

    ROS 2 remains local. The VLM inference request is sent through
    the locally exposed AI gateway, which may itself be connected
    to a remote GPU machine through the SSH tunnel.
    """

    rclpy.init()

    gateway = AIGatewayClient(
        base_url="http://127.0.0.1:8000",
        timeout=60.0,
    )

    # Fail early if the remote inference gateway is unavailable.
    try:
        health = gateway.health()
    except Exception as exc:
        print(
            "AI gateway health check failed. "
            "Make sure the SSH tunnel and remote gateway are running."
        )
        raise RuntimeError(
            f"Unable to connect to AI gateway: {exc}"
        ) from exc

    print(
        "AI gateway health:",
        health,
    )

    vlm_client = RemoteVLMClient(
        gateway=gateway,
    )

    perception_backend = RemoteVLMPerception(
        client=vlm_client,
    )

    node = CameraPerceptionNode(
        perception_backend=perception_backend,
    )

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
