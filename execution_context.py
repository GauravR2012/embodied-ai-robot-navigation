from __future__ import annotations

from dataclasses import dataclass, field

from target_resolver import Pose2D


@dataclass
class ExecutionContext:
    """
    Runtime state shared between skills during execution.
    """

    resolved_targets: dict[str, Pose2D] = field(default_factory=dict)

    def set_target(self, name: str, pose: Pose2D) -> None:
        self.resolved_targets[name] = pose

    def get_target(self, name: str) -> Pose2D:
        try:
            return self.resolved_targets[name]
        except KeyError as exc:
            raise RuntimeError(
                f"Target '{name}' has not been resolved."
            ) from exc

