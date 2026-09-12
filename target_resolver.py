from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Pose2D:
    """
    Robot navigation goal in a 2D map frame.
    """

    frame_id: str
    x: float
    y: float
    yaw: float


class TargetResolutionError(Exception):
    """Raised when a navigation target cannot be resolved."""


class TargetResolver:
    """
    Resolves logical navigation targets into environment-specific poses.

    The LLM never supplies coordinates.
    Coordinates come exclusively from configuration.
    """

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Location configuration not found: {self.config_path}"
            )

        self._load_config()

    def _load_config(self) -> None:
        with self.config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        if not isinstance(config, dict):
            raise TargetResolutionError(
                "Location configuration must be a YAML mapping."
            )

        self.frame_id = config.get("frame_id", "map")
        raw_locations = config.get("locations")

        if not isinstance(raw_locations, dict):
            raise TargetResolutionError(
                "'locations' must be a mapping."
            )

        self.locations: dict[str, Pose2D] = {}

        for name, data in raw_locations.items():
            if not isinstance(data, dict):
                raise TargetResolutionError(
                    f"Invalid configuration for location '{name}'."
                )

            try:
                x = float(data["x"])
                y = float(data["y"])
                yaw = float(data["yaw"])
            except (KeyError, TypeError, ValueError) as exc:
                raise TargetResolutionError(
                    f"Invalid pose for location '{name}'."
                ) from exc

            self.locations[name] = Pose2D(
                frame_id=self.frame_id,
                x=x,
                y=y,
                yaw=yaw,
            )

    def resolve_named_location(self, name: str) -> Pose2D:
        """
        Resolve a named location.

        Raises:
            TargetResolutionError:
                If the location is not configured.
        """

        try:
            return self.locations[name]
        except KeyError as exc:
            available = ", ".join(sorted(self.locations))

            raise TargetResolutionError(
                f"Unknown location '{name}'. "
                f"Available locations: {available}"
            ) from exc

    def resolve(self, target_type: str, target_value: str) -> Pose2D:
        if target_type != "named_location":
            raise TargetResolutionError(
                f"Unsupported target type: {target_type}"
            )

        return self.resolve_named_location(target_value)


def yaw_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
    """
    Convert planar yaw to a quaternion.

    Returns:
        (qx, qy, qz, qw)
    """

    half_yaw = yaw / 2.0

    return (
        0.0,
        0.0,
        math.sin(half_yaw),
        math.cos(half_yaw),
    )
