from __future__ import annotations

import rclpy

from command_pipeline import CommandPipelineError, execute_command
from ros2_executor import ROS2SkillExecutor
from target_resolver import TargetResolver


def main() -> int:
    rclpy.init()

    executor: ROS2SkillExecutor | None = None

    try:
        resolver = TargetResolver(
            "config/test_locations.yaml"
        )

        executor = ROS2SkillExecutor(resolver)

        command = {
            "action": "navigate",
            "target": {
                "type": "named_location",
                "value": "Executor Test",
            },
        }

        skills = execute_command(
            command,
            executor,
        )

        executor.get_logger().info(
            "Command pipeline completed successfully."
        )

        executor.get_logger().info(
            f"Executed skills: "
            f"{[skill.name for skill in skills]}"
        )

        return 0

    except CommandPipelineError as exc:
        if executor is not None:
            executor.get_logger().error(
                f"Command pipeline failed: {exc}"
            )
        else:
            print(f"Command pipeline failed: {exc}")

        return 1

    except Exception as exc:
        if executor is not None:
            executor.get_logger().error(
                f"Unexpected test failure: {exc}"
            )
        else:
            print(f"Unexpected test failure: {exc}")

        return 1

    finally:
        if executor is not None:
            executor.shutdown()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())