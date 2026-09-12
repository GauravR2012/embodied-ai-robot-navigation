from __future__ import annotations

import rclpy

from command_pipeline import CommandPipelineError, execute_command
from llm_adapter import LLMCommandAdapter, OllamaLLMBackend
from ros2_executor import ROS2SkillExecutor
from target_resolver import TargetResolver


def main() -> int:
    rclpy.init()

    executor: ROS2SkillExecutor | None = None

    try:
        # --------------------------------------------------------------
        # LLM
        # --------------------------------------------------------------

        backend = OllamaLLMBackend(
            model="qwen3:4b",
            temperature=0.0,
        )

        llm = LLMCommandAdapter(backend)

        user_text = "Go to Executor Test"

        command = llm.interpret_to_dict(
            user_text
        )

        # --------------------------------------------------------------
        # ROS 2 executor
        # --------------------------------------------------------------

        resolver = TargetResolver(
            "config/test_locations.yaml"
        )

        executor = ROS2SkillExecutor(
            resolver
        )

        executor.get_logger().info(
            f"LLM produced validated command: {command}"
        )

        # --------------------------------------------------------------
        # Command pipeline
        # --------------------------------------------------------------

        skills = execute_command(
            command,
            executor,
        )

        executor.get_logger().info(
            "LLM → command pipeline → Nav2 "
            "completed successfully."
        )

        executor.get_logger().info(
            f"Executed skills: "
            f"{[skill.name for skill in skills]}"
        )

        return 0

    except CommandPipelineError as exc:
        if executor is not None:
            executor.get_logger().error(
                f"LLM pipeline failed: {exc}"
            )
        else:
            print(
                f"LLM pipeline failed: {exc}"
            )

        return 1

    except Exception as exc:
        if executor is not None:
            executor.get_logger().error(
                f"Unexpected LLM pipeline failure: {exc}"
            )
        else:
            print(
                f"Unexpected LLM pipeline failure: {exc}"
            )

        return 1

    finally:
        if executor is not None:
            executor.shutdown()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())