from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ProjectedScanRay:
    """
    A single valid LaserScan measurement projected into image space.

    The LaserScan measurement originates in the LiDAR frame and is
    geometrically transformed into the camera optical frame before
    projection onto the image plane.
    """

    scan_index: int
    range_m: float

    # Original LaserScan polar measurement.
    angle_rad: float

    # Point expressed in the camera optical frame.
    camera_x_m: float
    camera_y_m: float
    camera_z_m: float

    # Pixel coordinates in the RGB image.
    pixel_u: float
    pixel_v: float


@dataclass(frozen=True)
class GroundedPoint:
    """
    Robust geometric estimate of a detected object's position.

    Coordinates are expressed in the requested output frame, normally
    base_link before transformation into map.
    """

    x_m: float
    y_m: float
    z_m: float

    frame_id: str

    range_m: float
    bearing_rad: float

    supporting_ray_count: int


@dataclass(frozen=True)
class StandoffPose:
    """
    Navigation pose generated from an object's grounded position.

    The robot is positioned standoff_distance_m away from the object
    and oriented toward the object.
    """

    x_m: float
    y_m: float
    yaw_rad: float

    frame_id: str

    standoff_distance_m: float


@dataclass(frozen=True)
class GroundedObject:
    """
    Perception result combined with geometric grounding.

    The label and bounding box originate from the VLM. The position
    originates from the geometric grounding pipeline.
    """

    label: str

    confidence: float | None

    bbox: Any

    point: GroundedPoint

    standoff_pose: StandoffPose

    supporting_ray_count: int

    grounding_confidence: float

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GroundingResult:
    """
    Result of attempting to geometrically ground a VLM detection.
    """

    success: bool

    object: GroundedObject | None = None

    reason: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def succeeded(
        cls,
        object: GroundedObject,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> GroundingResult:
        """Create a successful grounding result."""

        return cls(
            success=True,
            object=object,
            reason=None,
            metadata={} if metadata is None else metadata,
        )

    @classmethod
    def failed(
        cls,
        reason: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> GroundingResult:
        """Create a failed grounding result without raising an exception."""

        return cls(
            success=False,
            object=None,
            reason=reason,
            metadata={} if metadata is None else metadata,
        )