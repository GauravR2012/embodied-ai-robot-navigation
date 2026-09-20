from pathlib import Path

import rclpy

from command_pipeline import execute_command
from llm_adapter import LLMCommandAdapter, MockLLMBackend
from ros2_executor import ROS2SkillExecutor
from target_resolver import TargetResolver


def main() -> int:
    rclpy.init()

    executor = None

    try:
        # 1. Load named navigation locations
        config_path = Path("config/locations.yaml")
        resolver = TargetResolver(config_path)

        # 2. Initialize ROS 2 executor
        executor = ROS2SkillExecutor(resolver)

        # 3. Initialize natural-language adapter
        backend = MockLLMBackend()
        adapter = LLMCommandAdapter(backend)

        print("\nNatural-language robot control")
        print("Available mock commands:")
        print("  stop")
        print("  wait five seconds")
        print("  go to executor test")
        print("Type 'exit' to quit.\n")

        while rclpy.ok():
            user_text = input("Command > ").strip()

            if user_text.lower() in {"exit", "quit"}:
                break

            if not user_text:
                continue

            try:
                # 4. Natural language → validated RobotCommand
                command = adapter.interpret_to_dict(user_text)

                print(f"Validated command: {command}")

                # 5. RobotCommand → skills → ROS 2 execution
                execute_command(command, executor)

                print("Command completed successfully.")

            except Exception as exc:
                print(f"Command failed: {exc}")

        return 0

    except Exception as exc:
        print(f"Controller failed: {exc}")
        return 1

    finally:
        if executor is not None:
            executor.shutdown()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
