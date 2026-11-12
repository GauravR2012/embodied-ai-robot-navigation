"""Unit tests for RGB-D geometry (no ROS needed)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from perception.geometry import (
    central_patch,
    compute_standoff_pose,
    median_valid_depth,
    normalize_angle,
    optical_to_body,
    pixel_to_optical_point,
    quaternion_to_rotation_matrix,
    scale_bbox_to_image,
    transform_point,
    yaw_from_quaternion,
)


def quat_from_rpy(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)

    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def test_identity_quaternion():
    assert np.allclose(quaternion_to_rotation_matrix(0, 0, 0, 1), np.eye(3))


def test_yaw_90_rotates_x_to_y():
    q = quat_from_rpy(0, 0, math.pi / 2)

    x, y, z = transform_point((0, 0, 0), q, (1, 0, 0))

    assert (x, y, z) == pytest.approx((0, 1, 0), abs=1e-9)
    assert yaw_from_quaternion(*q) == pytest.approx(math.pi / 2)


def test_translation_is_applied_after_rotation():
    q = quat_from_rpy(0, 0, math.pi / 2)

    assert transform_point((10, 20, 0), q, (1, 0, 0)) == pytest.approx(
        (10, 21, 0), abs=1e-9
    )


def test_zero_quaternion_rejected():
    with pytest.raises(ValueError):
        quaternion_to_rotation_matrix(0, 0, 0, 0)


def test_normalize_angle():
    assert normalize_angle(3 * math.pi / 2) == pytest.approx(-math.pi / 2)
    assert normalize_angle(-3 * math.pi) == pytest.approx(math.pi, abs=1e-9) or \
        normalize_angle(-3 * math.pi) == pytest.approx(-math.pi, abs=1e-9)


# ---------------------------------------------------------------------------
# Depth sampling
# ---------------------------------------------------------------------------

def test_scale_bbox_rgb_to_half_resolution_depth():
    assert scale_bbox_to_image((100, 80, 300, 240), (640, 480), (320, 240)) == (
        50, 40, 150, 120,
    )


def test_scale_bbox_is_clamped():
    assert scale_bbox_to_image((-50, -50, 900, 700), (640, 480), (320, 240)) == (
        0, 0, 320, 240,
    )


def test_scale_bbox_collapsing_raises():
    with pytest.raises(ValueError):
        scale_bbox_to_image((700, 10, 800, 50), (640, 480), (320, 240))


def test_central_patch_stays_inside_image():
    x0, y0, x1, y1 = central_patch((0, 0, 10, 10), (320, 240))

    assert 0 <= x0 < x1 <= 320
    assert 0 <= y0 < y1 <= 240


def test_median_valid_depth_filters_invalid_values():
    region = np.array([[np.inf, np.nan, 0.0], [0.01, 2.0, 2.2], [20.0, 2.1, -1.0]])

    assert median_valid_depth(region) == pytest.approx(2.1)


def test_median_valid_depth_none_when_nothing_valid():
    assert median_valid_depth(np.full((4, 4), np.inf)) is None


# ---------------------------------------------------------------------------
# Back-projection and frame conventions
# ---------------------------------------------------------------------------

def test_principal_point_projects_straight_ahead():
    assert pixel_to_optical_point(160, 120, 2.0, 277, 277, 160, 120) == (0.0, 0.0, 2.0)


def test_pixel_right_of_center_has_positive_optical_x():
    x, y, z = pixel_to_optical_point(260, 120, 2.0, 277.0, 277.0, 160, 120)

    assert x > 0 and y == 0 and z == 2.0


def test_invalid_focal_length_rejected():
    with pytest.raises(ValueError):
        pixel_to_optical_point(0, 0, 1, 0, 1, 0, 0)


def test_optical_to_body():
    # 2 m ahead, 0.5 m to the right, 0.1 m below the axis.
    assert optical_to_body((0.5, 0.1, 2.0)) == (2.0, -0.5, -0.1)


def test_urdf_optical_joint_matches_manual_conversion():
    """
    TurtleBot3 URDF: camera_depth_optical_frame is camera_depth_frame
    rotated by rpy = (-1.57, 0, -1.57). Looking up the point in the
    optical frame must agree with converting it by hand into the body
    frame; this is why the code uses the optical frame.
    """

    q = quat_from_rpy(-1.57, 0.0, -1.57)
    optical = (0.5, 0.1, 2.0)

    via_tf = transform_point((0, 0, 0), q, optical)
    manual = optical_to_body(optical)

    assert via_tf == pytest.approx(manual, abs=0.01)


def test_using_optical_point_in_body_frame_would_be_wrong():
    """Documents the bug in the old test script (point treated as body)."""

    optical = (0.0, 0.0, 2.0)  # object dead ahead at 2 m

    assert optical_to_body(optical) == (2.0, 0.0, 0.0)
    assert optical != optical_to_body(optical)


# ---------------------------------------------------------------------------
# Standoff pose
# ---------------------------------------------------------------------------

def test_standoff_goal_is_short_of_the_object_and_faces_it():
    gx, gy, yaw, within = compute_standoff_pose((0, 0), (3, 0), 0.8)

    assert (gx, gy, yaw, within) == pytest.approx((2.2, 0.0, 0.0, False))


def test_standoff_diagonal():
    gx, gy, yaw, within = compute_standoff_pose((1, 1), (4, 5), 1.0)

    # Distance is 5; goal is 1 m short along the (0.6, 0.8) direction.
    assert (gx, gy) == pytest.approx((3.4, 4.2))
    assert yaw == pytest.approx(math.atan2(4, 3))
    assert not within


def test_already_within_standoff_only_turns_to_face():
    gx, gy, yaw, within = compute_standoff_pose((0, 0), (0, 0.5), 0.8)

    assert (gx, gy) == (0, 0)
    assert yaw == pytest.approx(math.pi / 2)
    assert within


def test_object_on_top_of_robot_rejected():
    with pytest.raises(ValueError):
        compute_standoff_pose((0, 0), (0.01, 0), 0.8)
