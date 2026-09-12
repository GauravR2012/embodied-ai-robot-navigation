import rclpy

from ros2_executor import ROS2SkillExecutor
from target_resolver import TargetResolver


def main():
    rclpy.init()

    resolver = TargetResolver("config/locations.yaml")
    executor = ROS2SkillExecutor(resolver)

    executor.get_logger().info(
        "Testing ActionClient from ROS2SkillExecutor..."
    )

    available = executor._navigate_client.wait_for_server(
        timeout_sec=10.0
    )

    executor.get_logger().info(
        f"Action server available: {available}"
    )

    executor.shutdown()

    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
