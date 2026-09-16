from __future__ import annotations

import math

import numpy as np
import pytest

from perception.grounding.lidar_projector import (
    CameraIntrinsics,
    LaserScanPoint,
    LaserScanProjector,
)


def make_identity_projector() -> LaserScanProjector:
    """Create a projector with an identity LiDAR-to-camera transform."""

    camera = CameraIntrinsics(
        fx=320.0,
        fy=320.0,
        cx=320.0,
        cy=240.0,
        width=640,
        height=480,
    )

    return LaserScanProjector(
        camera,
        np.eye(4),
    )


def test_camera_intrinsics_reject_invalid_focal_length() -> None:
    with pytest.raises(ValueError):
        CameraIntrinsics(
            fx=0.0,
            fy=320.0,
            cx=320.0,
            cy=240.0,
            width=640,
            height=480,
        )


def test_laser_scan_to_points() -> None:
    points = LaserScanProjector.laser_scan_to_points(
        [1.0, 2.0, 3.0],
        angle_min=0.0,
        angle_increment=math.pi / 2.0,
        range_min=0.1,
        range_max=10.0,
    )

    assert len(points) == 3

    assert points[0].scan_index == 0
    assert points[0].range_m == pytest.approx(1.0)
    assert points[0].angle_rad == pytest.approx(0.0)
    assert points[0].x_m == pytest.approx(1.0)
    assert points[0].y_m == pytest.approx(0.0)

    assert points[1].x_m == pytest.approx(0.0, abs=1e-9)
    assert points[1].y_m == pytest.approx(2.0)

    assert points[2].x_m == pytest.approx(-3.0)
    assert points[2].y_m == pytest.approx(0.0, abs=1e-9)


def test_invalid_scan_ranges_are_rejected() -> None:
    points = LaserScanProjector.laser_scan_to_points(
        [
            float("nan"),
            float("inf"),
            -float("inf"),
            0.05,
            0.5,
            10.0,
            11.0,
        ],
        angle_min=0.0,
        angle_increment=0.1,
        range_min=0.1,
        range_max=10.0,
    )

    assert len(points) == 2

    assert points[0].range_m == pytest.approx(0.5)
    assert points[1].range_m == pytest.approx(10.0)


def test_identity_projection_center_pixel() -> None:
    projector = make_identity_projector()

    point = LaserScanPoint(
        scan_index=0,
        range_m=2.0,
        angle_rad=0.0,
        x_m=0.0,
        y_m=0.0,
        z_m=2.0,
    )

    projected = projector.project_point(point)

    assert projected is not None
    assert projected.pixel_u == pytest.approx(320.0)
    assert projected.pixel_v == pytest.approx(240.0)
    assert projected.camera_z_m == pytest.approx(2.0)


def test_identity_projection_off_center() -> None:
    projector = make_identity_projector()

    point = LaserScanPoint(
        scan_index=0,
        range_m=2.0,
        angle_rad=0.0,
        x_m=1.0,
        y_m=-0.5,
        z_m=2.0,
    )

    projected = projector.project_point(point)

    assert projected is not None
    assert projected.pixel_u == pytest.approx(480.0)
    assert projected.pixel_v == pytest.approx(160.0)


def test_point_behind_camera_is_rejected() -> None:
    projector = make_identity_projector()

    point = LaserScanPoint(
        scan_index=0,
        range_m=2.0,
        angle_rad=0.0,
        x_m=0.0,
        y_m=0.0,
        z_m=-1.0,
    )

    assert projector.project_point(point) is None


def test_point_outside_image_is_rejected() -> None:
    projector = make_identity_projector()

    point = LaserScanPoint(
        scan_index=0,
        range_m=2.0,
        angle_rad=0.0,
        x_m=10.0,
        y_m=0.0,
        z_m=1.0,
    )

    assert projector.project_point(point) is None


def test_transform_is_applied_before_projection() -> None:
    camera = CameraIntrinsics(
        fx=100.0,
        fy=100.0,
        cx=320.0,
        cy=240.0,
        width=640,
        height=480,
    )

    transform = np.eye(4)
    transform[0, 3] = 1.0

    projector = LaserScanProjector(
        camera,
        transform,
    )

    point = LaserScanPoint(
        scan_index=4,
        range_m=2.0,
        angle_rad=0.0,
        x_m=0.0,
        y_m=0.0,
        z_m=2.0,
    )

    projected = projector.project_point(point)

    assert projected is not None

    # X becomes 1.0 after the transform:
    #
    # u = fx * X / Z + cx
    #   = 100 * 1 / 2 + 320
    #   = 370
    assert projected.pixel_u == pytest.approx(370.0)
    assert projected.pixel_v == pytest.approx(240.0)


def test_project_scan_returns_only_visible_points() -> None:
    projector = make_identity_projector()

    # With identity transform, these are treated as camera-frame
    # Cartesian coordinates.
    #
    # The first two points are valid.
    # The third has negative Z and must be rejected.
    points = (
        LaserScanPoint(
            scan_index=0,
            range_m=2.0,
            angle_rad=0.0,
            x_m=0.0,
            y_m=0.0,
            z_m=2.0,
        ),
        LaserScanPoint(
            scan_index=1,
            range_m=2.0,
            angle_rad=0.1,
            x_m=0.5,
            y_m=0.0,
            z_m=2.0,
        ),
        LaserScanPoint(
            scan_index=2,
            range_m=2.0,
            angle_rad=0.2,
            x_m=0.0,
            y_m=0.0,
            z_m=-1.0,
        ),
    )

    projected = projector.project_points(points)

    assert len(projected) == 2
    assert projected[0].scan_index == 0
    assert projected[1].scan_index == 1