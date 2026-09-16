from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np

from .models import ProjectedScanRay


@dataclass(frozen=True)
class CameraIntrinsics:
    """
    Pinhole camera intrinsic parameters.

    Coordinates follow the standard ROS camera optical convention:
        X = right
        Y = down
        Z = forward

    Pixel coordinates:
        u = fx * X / Z + cx
        v = fy * Y / Z + cy
    """

    fx: float
    fy: float
    cx: float
    cy: float

    width: int
    height: int

    def __post_init__(self) -> None:
        if self.fx <= 0.0:
            raise ValueError("fx must be positive")

        if self.fy <= 0.0:
            raise ValueError("fy must be positive")

        if self.width <= 0:
            raise ValueError("width must be positive")

        if self.height <= 0:
            raise ValueError("height must be positive")


@dataclass(frozen=True)
class LaserScanPoint:
    """
    A single LaserScan measurement represented as a Cartesian point.

    The coordinates are expressed in the LaserScan frame, which is
    currently `base_scan` in the simulator.
    """

    scan_index: int

    range_m: float
    angle_rad: float

    x_m: float
    y_m: float
    z_m: float = 0.0


class LaserScanProjector:
    """
    Projects 2D LaserScan measurements into the camera image.

    The pipeline is:

        LaserScan polar measurement
            ↓
        Cartesian point in LiDAR frame
            ↓
        rigid transform into camera optical frame
            ↓
        pinhole projection
            ↓
        image bounds filtering

    No ROS dependencies are required here. TF and ROS message handling
    belong in the ROS adapter layer.
    """

    def __init__(
        self,
        camera_intrinsics: CameraIntrinsics,
        lidar_to_camera_transform: Sequence[Sequence[float]],
        *,
        min_range_m: float = 0.05,
        max_range_m: float = 20.0,
        min_camera_depth_m: float = 0.05,
    ) -> None:
        self._camera = camera_intrinsics

        self._transform = self._validate_transform(
            lidar_to_camera_transform
        )

        if min_range_m <= 0.0:
            raise ValueError("min_range_m must be positive")

        if max_range_m <= min_range_m:
            raise ValueError(
                "max_range_m must be greater than min_range_m"
            )

        if min_camera_depth_m <= 0.0:
            raise ValueError("min_camera_depth_m must be positive")

        self._min_range_m = min_range_m
        self._max_range_m = max_range_m
        self._min_camera_depth_m = min_camera_depth_m

    @staticmethod
    def _validate_transform(
        transform: Sequence[Sequence[float]],
    ) -> np.ndarray:
        """Validate and normalize a homogeneous 4x4 transform."""

        matrix = np.asarray(transform, dtype=float)

        if matrix.shape != (4, 4):
            raise ValueError(
                "lidar_to_camera_transform must be a 4x4 matrix"
            )

        if not np.all(np.isfinite(matrix)):
            raise ValueError(
                "lidar_to_camera_transform contains non-finite values"
            )

        if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0]):
            raise ValueError(
                "lidar_to_camera_transform must be homogeneous"
            )

        return matrix

    @staticmethod
    def laser_scan_to_points(
        ranges: Iterable[float],
        *,
        angle_min: float,
        angle_increment: float,
        range_min: float,
        range_max: float,
        z_m: float = 0.0,
    ) -> tuple[LaserScanPoint, ...]:
        """
        Convert LaserScan polar measurements into Cartesian points.

        Invalid measurements are rejected here. In particular, ROS
        LaserScan streams may contain NaN, +inf, -inf, or values outside
        the sensor's declared range limits.
        """

        if angle_increment == 0.0:
            raise ValueError("angle_increment must not be zero")

        if range_min < 0.0:
            raise ValueError("range_min must not be negative")

        if range_max <= range_min:
            raise ValueError(
                "range_max must be greater than range_min"
            )

        points: list[LaserScanPoint] = []

        for index, raw_range in enumerate(ranges):
            range_m = float(raw_range)

            if not math.isfinite(range_m):
                continue

            if range_m < range_min or range_m > range_max:
                continue

            angle_rad = angle_min + index * angle_increment

            x_m = range_m * math.cos(angle_rad)
            y_m = range_m * math.sin(angle_rad)

            points.append(
                LaserScanPoint(
                    scan_index=index,
                    range_m=range_m,
                    angle_rad=angle_rad,
                    x_m=x_m,
                    y_m=y_m,
                    z_m=z_m,
                )
            )

        return tuple(points)

    def transform_point_to_camera(
        self,
        point: LaserScanPoint,
    ) -> tuple[float, float, float]:
        """
        Transform a LaserScan point into the camera optical frame.
        """

        point_h = np.array(
            [
                point.x_m,
                point.y_m,
                point.z_m,
                1.0,
            ],
            dtype=float,
        )

        camera_point = self._transform @ point_h

        return (
            float(camera_point[0]),
            float(camera_point[1]),
            float(camera_point[2]),
        )

    def project_point(
        self,
        point: LaserScanPoint,
    ) -> ProjectedScanRay | None:
        """
        Transform and project one LaserScan measurement.

        Returns None when the measurement cannot produce a valid pixel,
        for example when:

        - the range is outside configured limits,
        - the point is behind the camera,
        - the point is too close to the camera plane,
        - the projected pixel lies outside the image.
        """

        if not math.isfinite(point.range_m):
            return None

        if (
            point.range_m < self._min_range_m
            or point.range_m > self._max_range_m
        ):
            return None

        camera_x, camera_y, camera_z = (
            self.transform_point_to_camera(point)
        )

        if not all(
            math.isfinite(value)
            for value in (camera_x, camera_y, camera_z)
        ):
            return None

        if camera_z <= self._min_camera_depth_m:
            return None

        pixel_u = (
            self._camera.fx * camera_x / camera_z
            + self._camera.cx
        )

        pixel_v = (
            self._camera.fy * camera_y / camera_z
            + self._camera.cy
        )

        if not math.isfinite(pixel_u) or not math.isfinite(pixel_v):
            return None

        if not (
            0.0 <= pixel_u < self._camera.width
            and 0.0 <= pixel_v < self._camera.height
        ):
            return None

        return ProjectedScanRay(
            scan_index=point.scan_index,
            range_m=point.range_m,
            angle_rad=point.angle_rad,
            camera_x_m=camera_x,
            camera_y_m=camera_y,
            camera_z_m=camera_z,
            pixel_u=pixel_u,
            pixel_v=pixel_v,
        )

    def project_points(
        self,
        points: Iterable[LaserScanPoint],
    ) -> tuple[ProjectedScanRay, ...]:
        """
        Project all valid LaserScan points into the camera image.
        """

        projected: list[ProjectedScanRay] = []

        for point in points:
            result = self.project_point(point)

            if result is not None:
                projected.append(result)

        return tuple(projected)

    def project_scan(
        self,
        ranges: Iterable[float],
        *,
        angle_min: float,
        angle_increment: float,
        range_min: float,
        range_max: float,
    ) -> tuple[ProjectedScanRay, ...]:
        """
        Convenience method performing the complete LaserScan → pixel
        projection pipeline.
        """

        points = self.laser_scan_to_points(
            ranges,
            angle_min=angle_min,
            angle_increment=angle_increment,
            range_min=range_min,
            range_max=range_max,
        )

        return self.project_points(points)