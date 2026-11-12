from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from target_resolver import Pose2D


@dataclass
class ExecutionContext:
    """
    Runtime state shared between skills during execution.

    Targets are stored under a key of the form ``"<type>:<value>"``
    (for example ``"named_location:Station 1"`` or ``"object:red box"``)
    so a named location and an object can never collide.
    """

    resolved_targets: dict[str, Pose2D] = field(default_factory=dict)

    # One summary dictionary per object search performed in this run.
    object_searches: list[dict[str, Any]] = field(default_factory=list)

    @staticmethod
    def target_key(target_type: str, target_value: str) -> str:
        return f"{target_type}:{target_value}"

    def set_target(self, name: str, pose: Pose2D) -> None:
        self.resolved_targets[name] = pose

    def get_target(self, name: str) -> Pose2D:
        try:
            return self.resolved_targets[name]
        except KeyError as exc:
            raise RuntimeError(
                f"Target '{name}' has not been resolved."
            ) from exc

    def record_object_search(self, summary: dict[str, Any]) -> None:
        self.object_searches.append(summary)
