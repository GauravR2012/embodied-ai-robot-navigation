from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BoundingBox:
    """2D image-space bounding box in pixel coordinates."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass(frozen=True)
class DetectedObject:
    """Object detected by a perception backend."""

    label: str
    confidence: float | None = None
    bbox: BoundingBox | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PerceptionResult:
    """Backend-independent perception result."""

    objects: tuple[DetectedObject, ...] = ()
    scene_description: str | None = None
    raw_response: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
