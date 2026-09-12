import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from nav2_msgs.action import NavigateToPose


class ActionClientTest(Node):

    def __init__(self):
        super().__init__("action_client_test")

        self.client = ActionClient(
            self,
            NavigateToPose,
            "/navigate_to_pose",
        )


def main():
    rclpy.init()

    node = ActionClientTest()

    node.get_logger().info(
        "Waiting for /navigate_to_pose..."
    )

    available = node.client.wait_for_server(
        timeout_sec=10.0
    )

    node.get_logger().info(
        f"Action server available: {available}"
    )

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
