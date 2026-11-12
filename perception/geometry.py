"""
Pure geometry helpers for RGB-D object grounding.

Nothing in this module imports ROS, so every function can be unit-tested
on a machine without a ROS installation.

Camera conventions
------------------

Optical frame (e.g. ``camera_depth_optical_frame``):

    x -> right in the image
    y -> down in the image
    z -> forward (the direction the depth is measured along)

Body / REP-103 frame (e.g. ``camera_depth_frame``):

    x -> forward
    y -> left
    z -> up

A depth pixel back-projects to a point in the OPTICAL convention.
If that point is later expressed in a non-optical frame it must first be
converted with :func:`optical_to_body`.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Angles / quaternions
# ---------------------------------------------------------------------------

def normalize_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    """Planar yaw of a quaternion."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_to_rotation_matrix(
    x: float,
    y: float,
    z: float,
    w: float,
) -> np.ndarray:
    """Convert a quaternion (x, y, z, w) to a 3x3 rotation matrix."""

    norm = math.sqrt(x * x + y * y + z * z + w * w)

    if norm < 1e-12:
        raise ValueError("Quaternion has zero length.")

    x, y, z, w = x / norm, y / norm, z / norm, w / norm

    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=float,
    )


def transform_point(
    translation: Sequence[float],
    quaternion_xyzw: Sequence[float],
    point: Sequence[float],
) -> tuple[float, float, float]:
    """
    Apply a rigid transform to a point.

    ``translation`` / ``quaternion_xyzw`` describe the pose of the source
    frame expressed in the target frame (this is exactly what a ROS
    ``TransformStamped`` from ``lookup_transform(target, source)`` holds).
    """

    rotation = quaternion_to_rotation_matrix(*quaternion_xyzw)

    result = rotation @ np.asarray(point, dtype=float) + np.asarray(
        translation,
        dtype=float,
    )

    return float(result[0]), float(result[1]), float(result[2])


# ---------------------------------------------------------------------------
# Image / depth helpers
# ---------------------------------------------------------------------------

def scale_bbox_to_image(
    bbox: tuple[float, float, float, float],
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """
    Map an (x_min, y_min, x_max, y_max) box between two image resolutions.

    Both sizes are (width, height). The returned box is clamped to the
    target image and is guaranteed to have positive width and height.

    Raises:
        ValueError: if the box collapses after scaling/clamping.
    """

    x_min, y_min, x_max, y_max = bbox
    src_w, src_h = source_size
    dst_w, dst_h = target_size

    if src_w <= 0 or src_h <= 0 or dst_w <= 0 or dst_h <= 0:
        raise ValueError("Image sizes must be positive.")

    sx = dst_w / float(src_w)
    sy = dst_h / float(src_h)

    x0 = int(round(x_min * sx))
    y0 = int(round(y_min * sy))
    x1 = int(round(x_max * sx))
    y1 = int(round(y_max * sy))

    # Clamp both corners into [0, size]. A box that lies completely
    # outside the image collapses to zero width/height and is rejected
    # below instead of becoming a one-pixel sliver at the border.
    x0 = max(0, min(dst_w, x0))
    y0 = max(0, min(dst_h, y0))
    x1 = max(0, min(dst_w, x1))
    y1 = max(0, min(dst_h, y1))

    if x1 <= x0 or y1 <= y0:
        raise ValueError(
            "Bounding box is empty after scaling to the depth image: "
            f"({x0}, {y0}, {x1}, {y1})"
        )

    return x0, y0, x1, y1


def central_patch(
    box: tuple[int, int, int, int],
    image_size: tuple[int, int],
    fraction: float = 0.15,
    min_radius: int = 2,
) -> tuple[int, int, int, int]:
    """
    Return (x0, y0, x1, y1) of a small patch centred in ``box``.

    Using a patch instead of one pixel makes depth sampling robust to
    isolated invalid / noisy pixels.
    """

    x0, y0, x1, y1 = box
    img_w, img_h = image_size

    center_x = int(round((x0 + x1) / 2.0))
    center_y = int(round((y0 + y1) / 2.0))

    radius_x = max(min_radius, int((x1 - x0) * fraction))
    radius_y = max(min_radius, int((y1 - y0) * fraction))

    return (
        max(0, center_x - radius_x),
        max(0, center_y - radius_y),
        min(img_w, center_x + radius_x + 1),
        min(img_h, center_y + radius_y + 1),
    )


def median_valid_depth(
    depth_region: np.ndarray,
    min_depth_m: float = 0.05,
    max_depth_m: float = 10.0,
) -> float | None:
    """
    Median of the finite depth values inside [min_depth_m, max_depth_m].

    Returns None if no pixel is valid.
    """

    values = np.asarray(depth_region, dtype=float).reshape(-1)

    valid = values[
        np.isfinite(values)
        & (values > min_depth_m)
        & (values < max_depth_m)
    ]

    if valid.size == 0:
        return None

    return float(np.median(valid))


def pixel_to_optical_point(
    u: float,
    v: float,
    depth_m: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[float, float, float]:
    """
    Pinhole back-projection of pixel (u, v) at ``depth_m``.

    Returns the point in the OPTICAL convention (x right, y down, z fwd).
    """

    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("Camera focal lengths must be positive.")

    x = (u - cx) * depth_m / fx
    y = (v - cy) * depth_m / fy

    return x, y, depth_m


def optical_to_body(
    point: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Convert an optical-convention point to the REP-103 body convention."""

    x, y, z = point

    return z, -x, -y


# ---------------------------------------------------------------------------
# Navigation goal
# ---------------------------------------------------------------------------

def compute_standoff_pose(
    robot_xy: tuple[float, float],
    object_xy: tuple[float, float],
    standoff_distance_m: float,
    min_object_distance_m: float = 0.05,
) -> tuple[float, float, float, bool]:
    """
    Compute a goal that stops ``standoff_distance_m`` short of the object,
    facing it.

    Returns:
        (goal_x, goal_y, goal_yaw, already_within_standoff)

    If the robot is already closer than the standoff distance the goal is
    the robot's own position, rotated to face the object (the robot never
    reverses away from, or drives closer to, the object).

    Raises:
        ValueError: if the object is essentially on top of the robot.
    """

    robot_x, robot_y = robot_xy
    object_x, object_y = object_xy

    dx = object_x - robot_x
    dy = object_y - robot_y

    distance = math.hypot(dx, dy)

    if distance < min_object_distance_m:
        raise ValueError(
            "Object is too close to the robot to compute a goal."
        )

    ux = dx / distance
    uy = dy / distance

    yaw = math.atan2(dy, dx)

    if distance <= standoff_distance_m:
        return robot_x, robot_y, yaw, True

    goal_x = object_x - standoff_distance_m * ux
    goal_y = object_y - standoff_distance_m * uy

    return goal_x, goal_y, yaw, False
