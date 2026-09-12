from __future__ import annotations

from typing import Any

from command_schema import parse_command
from ros2_executor import ROS2SkillExecutor
from task_planner import Skill, plan_command


class CommandPipelineError(RuntimeError):
    """Raised when command processing or execution fails."""


def execute_command(
    command_data: dict[str, Any],
    executor: ROS2SkillExecutor,
) -> list[Skill]:
    """Validate, plan, and execute a robot command.

    Pipeline:

        raw command
            ↓
        schema validation
            ↓
        task planning
            ↓
        skill execution
            ↓
        structured execution result
    """

    # --------------------------------------------------------------
    # Validate input type
    # --------------------------------------------------------------

    if not isinstance(command_data, dict):
        raise CommandPipelineError(
            "Command input must be a dictionary."
        )

    # --------------------------------------------------------------
    # Validate command against the Pydantic schema
    # --------------------------------------------------------------

    try:
        command = parse_command(command_data)
    except Exception as exc:
        raise CommandPipelineError(
            f"Command validation failed: {exc}"
        ) from exc

    # --------------------------------------------------------------
    # Convert command into executable skills
    # --------------------------------------------------------------

    try:
        skills = plan_command(command)
    except Exception as exc:
        raise CommandPipelineError(
            f"Task planning failed: {exc}"
        ) from exc

    if not skills:
        raise CommandPipelineError(
            "Task planner produced no skills."
        )

    # --------------------------------------------------------------
    # Execute skills through ROS 2
    # --------------------------------------------------------------

    result = executor.execute(skills)

    if not result.success:
        raise CommandPipelineError(
            f"{result.error_code}: {result.error_message}"
        )

    return skills