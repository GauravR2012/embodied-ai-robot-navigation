from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from command_schema import (
    NavigateCommand,
    MoveCommand,
    RotateCommand,
    StopCommand,
    WaitCommand,
    SequenceCommand,
    RobotCommand,
)


@dataclass(frozen=True)
class Skill:
    """
    A deterministic robot skill produced by the task planner.

    The planner decides WHAT needs to happen.
    The executor will later decide HOW to perform it on ROS 2.
    """

    name: str
    parameters: dict


def plan_command(command: RobotCommand) -> list[Skill]:
    """
    Convert a validated robot command into an ordered list of skills.
    """

    if isinstance(command, NavigateCommand):
        # Physical objects are located with the VLM (rotating in place
        # if necessary) instead of being looked up in the location file.
        # If the object is not found, find_object fails and the robot
        # never receives a navigation goal.
        if command.target.type == "object":
            return [
                Skill(
                    name="find_object",
                    parameters={
                        "target_type": command.target.type,
                        "target_value": command.target.value,
                    },
                ),
                Skill(
                    name="navigate_to_target",
                    parameters={
                        "target_type": command.target.type,
                        "target_value": command.target.value,
                    },
                ),
                Skill(
                    name="wait_for_navigation_result",
                    parameters={},
                ),
            ]

        return [
            Skill(
                name="resolve_target",
                parameters={
                    "target_type": command.target.type,
                    "target_value": command.target.value,
                },
            ),
            Skill(
                name="navigate_to_target",
                parameters={
                    "target_type": command.target.type,
                    "target_value": command.target.value,
                },
            ),
            Skill(
                name="wait_for_navigation_result",
                parameters={},
            ),
        ]

    if isinstance(command, MoveCommand):
        return [
            Skill(
                name="move",
                parameters={
                    "direction": command.direction,
                    "distance": command.distance,
                    "unit": command.unit,
                },
            )
        ]

    if isinstance(command, RotateCommand):
        return [
            Skill(
                name="rotate",
                parameters={
                    "direction": command.direction,
                    "angle": command.angle,
                    "unit": command.unit,
                },
            )
        ]

    if isinstance(command, StopCommand):
        return [
            Skill(
                name="stop",
                parameters={},
            )
        ]

    if isinstance(command, WaitCommand):
        return [
            Skill(
                name="wait",
                parameters={
                    "duration": command.duration,
                    "unit": command.unit,
                },
            )
        ]

    if isinstance(command, SequenceCommand):
        skills: list[Skill] = []

        for subcommand in command.commands:
            skills.extend(plan_command(subcommand))

        return skills

    raise TypeError(
        f"Unsupported command type: {type(command).__name__}"
    )
