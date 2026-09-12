
import rclpy

from command_pipeline import execute_command
from llm_adapter import LLMCommandAdapter, OllamaLLMBackend
from ros2_executor import ROS2SkillExecutor
from target_resolver import TargetResolver


rclpy.init()

executor = None

try:
    resolver = TargetResolver("config/locations.yaml")
    executor = ROS2SkillExecutor(resolver)

    backend = OllamaLLMBackend(
        model="qwen3:4b",
        temperature=0.0,
    )

    llm = LLMCommandAdapter(backend)

    user_instruction = "Go to the Executor Test location."

    executor.get_logger().info(
        f"User instruction: {user_instruction}"
    )

    command = llm.interpret_to_dict(user_instruction)

    executor.get_logger().info(
        f"LLM produced validated command: {command}"
    )

    execute_command(command, executor)

    executor.get_logger().info(
        "REAL LLM -> COMMAND -> PLANNER -> ROS 2 -> NAV2 SUCCESS"
    )

    raise SystemExit(0)

except Exception as exc:
    if executor is not None:
        executor.get_logger().error(
            f"LLM robot pipeline failed: {exc}"
        )
    else:
        print(f"LLM robot pipeline failed: {exc}")

    raise SystemExit(1)

finally:
    if executor is not None:
        executor.shutdown()

    if rclpy.ok():
        rclpy.shutdown()
