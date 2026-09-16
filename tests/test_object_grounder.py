from __future__ import annotations

import math

import pytest

from perception.grounding.models import ProjectedScanRay
from perception.grounding.object_grounder import (
    GroundingConfig,
    ObjectGrounder,
)
from perception.models import BoundingBox, DetectedObject


def make_ray(
    scan_index: int,
    range_m: float,
    pixel_u: float,
    pixel_v: float,
    angle_rad: float,
) -> ProjectedScanRay:
    return ProjectedScanRay(
        scan_index=scan_index,
        range_m=range_m,
        angle_rad=angle_rad,
        camera_x_m=range_m,
        camera_y_m=0.0,
        camera_z_m=range_m,
        pixel_u=pixel_u,
        pixel_v=pixel_v,
    )


def make_detection(
    *,
    label: str = "chair",
    confidence: float | None = 0.95,
    bbox: BoundingBox | None = None,
) -> DetectedObject:
    return DetectedObject(
        label=label,
        confidence=confidence,
        bbox=bbox
        or BoundingBox(
            x_min=100.0,
            y_min=100.0,
            x_max=300.0,
            y_max=300.0,
        ),
    )


def test_bbox_filter_keeps_rays_inside_bbox() -> None:
    grounder = ObjectGrounder()

    rays = [
        make_ray(0, 2.0, 150.0, 150.0, 0.0),
        make_ray(1, 2.0, 250.0, 250.0, 0.1),
        make_ray(2, 2.0, 350.0, 150.0, 0.2),
    ]

    detection = make_detection()

    filtered = grounder.filter_rays_by_bbox(
        rays,
        detection.bbox,
    )

    assert len(filtered) == 2
    assert [ray.scan_index for ray in filtered] == [0, 1]


def test_grounding_produces_base_position() -> None:
    grounder = ObjectGrounder()

    rays = [
        make_ray(10, 2.0, 150.0, 150.0, 0.0),
        make_ray(11, 2.05, 160.0, 155.0, 0.01),
        make_ray(12, 2.02, 170.0, 160.0, 0.02),
    ]

    result = grounder.ground(
        make_detection(),
        rays,
    )

    assert result.success
    assert result.object is not None

    point = result.object.point

    assert point.frame_id == "base_link"
    assert point.x_m == pytest.approx(2.02, abs=0.03)
    assert point.y_m == pytest.approx(0.02, abs=0.02)
    assert point.range_m == pytest.approx(2.02, abs=0.03)
    assert point.supporting_ray_count == 3


def test_grounding_computes_standoff_pose() -> None:
    grounder = ObjectGrounder(
        GroundingConfig(
            standoff_distance_m=0.60,
        )
    )

    rays = [
        make_ray(10, 2.0, 150.0, 150.0, 0.0),
        make_ray(11, 2.0, 160.0, 150.0, 0.0),
        make_ray(12, 2.0, 170.0, 150.0, 0.0),
    ]

    result = grounder.ground(
        make_detection(),
        rays,
    )

    assert result.success
    assert result.object is not None

    pose = result.object.standoff_pose

    assert pose.frame_id == "base_link"
    assert pose.x_m == pytest.approx(1.40)
    assert pose.y_m == pytest.approx(0.0)
    assert pose.yaw_rad == pytest.approx(0.0)
    assert pose.standoff_distance_m == pytest.approx(0.60)


def test_depth_discontinuity_selects_nearer_cluster() -> None:
    grounder = ObjectGrounder()

    rays = [
        make_ray(10, 2.0, 150.0, 150.0, 0.0),
        make_ray(11, 2.05, 160.0, 150.0, 0.01),
        make_ray(12, 4.5, 170.0, 150.0, 0.02),
        make_ray(13, 4.55, 180.0, 150.0, 0.03),
    ]

    result = grounder.ground(
        make_detection(),
        rays,
    )

    assert result.success
    assert result.object is not None

    assert result.object.point.range_m == pytest.approx(
        2.025,
        abs=0.03,
    )

    assert result.object.supporting_ray_count == 2


def test_missing_bbox_fails() -> None:
    grounder = ObjectGrounder()

    detection = make_detection(
        bbox=None,
    )

    detection = DetectedObject(
        label=detection.label,
        confidence=detection.confidence,
        bbox=None,
    )

    result = grounder.ground(
        detection,
        [],
    )

    assert not result.success
    assert result.object is None
    assert result.reason is not None


def test_invalid_bbox_fails() -> None:
    grounder = ObjectGrounder()

    detection = make_detection(
        bbox=BoundingBox(
            x_min=300.0,
            y_min=300.0,
            x_max=100.0,
            y_max=100.0,
        )
    )

    result = grounder.ground(
        detection,
        [],
    )

    assert not result.success
    assert "invalid bounding box" in result.reason


def test_no_rays_inside_bbox_fails() -> None:
    grounder = ObjectGrounder()

    rays = [
        make_ray(0, 2.0, 500.0, 400.0, 0.0),
    ]

    result = grounder.ground(
        make_detection(),
        rays,
    )

    assert not result.success
    assert "no projected LaserScan rays" in result.reason

def test_insufficient_support_fails() -> None:
    grounder = ObjectGrounder(
        GroundingConfig(
            min_supporting_rays=2,
        )
    )

    rays = [
        make_ray(0, 2.0, 150.0, 150.0, 0.0),
    ]

    result = grounder.ground(
        make_detection(),
        rays,
    )

    assert not result.success
    assert result.reason is not None
    assert (
        "no sufficiently supported LaserScan cluster"
        in result.reason
    )
    
def test_close_object_cannot_get_standoff_pose() -> None:
    grounder = ObjectGrounder(
        GroundingConfig(
            standoff_distance_m=0.60,
            min_object_range_m=0.20,
        )
    )

    rays = [
        make_ray(0, 0.40, 150.0, 150.0, 0.0),
        make_ray(1, 0.42, 160.0, 150.0, 0.01),
    ]

    result = grounder.ground(
        make_detection(),
        rays,
    )

    assert not result.success
    assert result.object is None


def test_grounding_confidence_is_bounded() -> None:
    grounder = ObjectGrounder()

    rays = [
        make_ray(10, 2.0, 150.0, 150.0, 0.0),
        make_ray(11, 2.0, 160.0, 150.0, 0.01),
        make_ray(12, 2.0, 170.0, 150.0, 0.02),
    ]

    result = grounder.ground(
        make_detection(confidence=0.9),
        rays,
    )

    assert result.success
    assert result.object is not None

    confidence = result.object.grounding_confidence

    assert math.isfinite(confidence)
    assert 0.0 <= confidence <= 1.0
