from __future__ import annotations

from task_planner import Skill
from target_resolver import TargetResolver
from execution_context import ExecutionContext


class MockExecutor:
    """
    Simulated robot executor.

    This executor does not communicate with ROS.
    It exists to verify the complete command -> plan ->
    target resolution -> execution pipeline.
    """

    def __init__(
        self,
        target_resolver: TargetResolver,
        context: ExecutionContext | None = None,
    ):
        self.target_resolver = target_resolver
        self.context = context or ExecutionContext()

    def execute(self, skills: list[Skill]) -> None:
        for index, skill in enumerate(skills, start=1):
            print(
                f"[{index}/{len(skills)}] "
                f"Executing: {skill.name} "
                f"{skill.parameters}"
            )

            self._execute_skill(skill)

    def _execute_skill(self, skill: Skill) -> None:

        if skill.name == "resolve_target":
            self._resolve_target(skill)

        elif skill.name == "navigate_to_target":
            self._navigate_to_target(skill)

        elif skill.name == "wait_for_navigation_result":
            self._wait_for_navigation_result()

        elif skill.name == "move":
            self._move(skill)

        elif skill.name == "rotate":
            self._rotate(skill)

        elif skill.name == "stop":
            self._stop()

        elif skill.name == "wait":
            self._wait(skill)

        else:
            raise ValueError(
                f"Unknown skill: {skill.name}"
            )

    def _resolve_target(self, skill: Skill) -> None:
        target_type = skill.parameters["target_type"]
        target_value = skill.parameters["target_value"]

        pose = self.target_resolver.resolve(
            target_type,
            target_value,
        )

        self.context.set_target(target_value, pose)

        print(
            f"    [MOCK] Resolved '{target_value}' -> "
            f"frame={pose.frame_id}, "
            f"x={pose.x}, "
            f"y={pose.y}, "
            f"yaw={pose.yaw}"
        )

    def _navigate_to_target(self, skill: Skill) -> None:
        target_value = skill.parameters["target_value"]

        pose = self.context.get_target(target_value)

        print(
            f"    [MOCK] Would navigate to "
            f"x={pose.x}, "
            f"y={pose.y}, "
            f"yaw={pose.yaw}"
        )

    @staticmethod
    def _wait_for_navigation_result() -> None:
        print("    [MOCK] Waiting for navigation result...")
        print("    [MOCK] Navigation succeeded.")

    @staticmethod
    def _move(skill: Skill) -> None:
        print(
            f"    [MOCK] Moving "
            f"{skill.parameters['direction']} "
            f"{skill.parameters['distance']}"
            f"{skill.parameters['unit']}"
        )

    @staticmethod
    def _rotate(skill: Skill) -> None:
        print(
            f"    [MOCK] Rotating "
            f"{skill.parameters['direction']} "
            f"{skill.parameters['angle']}"
            f"{skill.parameters['unit']}"
        )

    @staticmethod
    def _stop() -> None:
        print("    [MOCK] Robot stopped.")

    @staticmethod
    def _wait(skill: Skill) -> None:
        print(
            f"    [MOCK] Waiting for "
            f"{skill.parameters['duration']}"
            f"{skill.parameters['unit']}"
        )