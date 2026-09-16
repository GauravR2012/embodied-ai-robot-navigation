from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np

from perception.models import BoundingBox, DetectedObject

from .models import (
    GroundedObject,
    GroundedPoint,
    GroundingResult,
    ProjectedScanRay,
    StandoffPose,
)
from .point_clusterer import (
    ClustererConfig,
    ScanCluster,
    ScanPointClusterer,
)


@dataclass(frozen=True)
class GroundingConfig:
    """
    Configuration for VLM bounding-box to geometric-object grounding.
    """

    min_supporting_rays: int = 2

    min_grounding_confidence: float = 0.20

    standoff_distance_m: float = 0.60

    min_object_range_m: float = 0.20
    max_object_range_m: float = 10.0

    cluster: ClustererConfig = ClustererConfig()

    def __post_init__(self) -> None:
        if self.min_supporting_rays < 1:
            raise ValueError(
                "min_supporting_rays must be at least 1"
            )

        if not 0.0 <= self.min_grounding_confidence <= 1.0:
            raise ValueError(
                "min_grounding_confidence must be between 0 and 1"
            )

        if self.standoff_distance_m <= 0.0:
            raise ValueError(
                "standoff_distance_m must be positive"
            )

        if self.min_object_range_m <= 0.0:
            raise ValueError(
                "min_object_range_m must be positive"
            )

        if self.max_object_range_m <= self.min_object_range_m:
            raise ValueError(
                "max_object_range_m must be greater than "
                "min_object_range_m"
            )


class ObjectGrounder:
    """
    Grounds a VLM-detected object using projected LaserScan rays.

    The VLM provides semantic information:

        "this image region contains a chair"

    The LaserScan provides metric information:

        "these rays intersect geometry at approximately this range"

    This component combines the two without depending on ROS.
    """

    def __init__(
        self,
        config: GroundingConfig | None = None,
    ) -> None:
        self._config = config or GroundingConfig()

        self._clusterer = ScanPointClusterer(
            self._config.cluster
        )

    @staticmethod
    def _validate_bbox(
        bbox: BoundingBox,
    ) -> bool:
        """Return whether a bounding box is geometrically valid."""

        values = (
            bbox.x_min,
            bbox.y_min,
            bbox.x_max,
            bbox.y_max,
        )

        if not all(math.isfinite(value) for value in values):
            return False

        return (
            bbox.x_min < bbox.x_max
            and bbox.y_min < bbox.y_max
        )

    @staticmethod
    def _ray_inside_bbox(
        ray: ProjectedScanRay,
        bbox: BoundingBox,
    ) -> bool:
        """
        Determine whether a projected scan ray falls inside a VLM bbox.
        """

        return (
            bbox.x_min <= ray.pixel_u <= bbox.x_max
            and bbox.y_min <= ray.pixel_v <= bbox.y_max
        )

    def filter_rays_by_bbox(
        self,
        rays: Iterable[ProjectedScanRay],
        bbox: BoundingBox,
    ) -> tuple[ProjectedScanRay, ...]:
        """
        Keep projected LaserScan rays whose image projections lie
        inside the VLM bounding box.
        """

        if not self._validate_bbox(bbox):
            return ()

        return tuple(
            ray
            for ray in rays
            if self._ray_inside_bbox(ray, bbox)
        )

    def _filter_range(
        self,
        rays: Iterable[ProjectedScanRay],
    ) -> tuple[ProjectedScanRay, ...]:
        """Reject geometrically implausible object ranges."""

        return tuple(
            ray
            for ray in rays
            if (
                self._config.min_object_range_m
                <= ray.range_m
                <= self._config.max_object_range_m
            )
        )

    @staticmethod
    def _robust_point(
        cluster: ScanCluster,
    ) -> tuple[float, float, float]:
        """
        Estimate a robust 3D point from a scan cluster.

        Median coordinates are used rather than the arithmetic mean so
        that a small number of noisy measurements has less influence.
        """

        x = float(
            np.median(
                [ray.camera_x_m for ray in cluster.rays]
            )
        )

        y = float(
            np.median(
                [ray.camera_y_m for ray in cluster.rays]
            )
        )

        z = float(
            np.median(
                [ray.camera_z_m for ray in cluster.rays]
            )
        )

        return x, y, z

    @staticmethod
    def _base_planar_position(
        cluster: ScanCluster,
    ) -> tuple[float, float]:
        """
        Estimate planar object position directly from LaserScan
        geometry.

        The LaserScan itself is expressed in the robot base frame.
        This gives us a stable navigation-plane estimate without
        requiring the camera-frame point to be transformed back.

        For each ray:

            x = r cos(theta)
            y = r sin(theta)

        The median is used for robustness.
        """

        x_values = [
            ray.range_m * math.cos(ray.angle_rad)
            for ray in cluster.rays
        ]

        y_values = [
            ray.range_m * math.sin(ray.angle_rad)
            for ray in cluster.rays
        ]

        return (
            float(np.median(x_values)),
            float(np.median(y_values)),
        )

    def _compute_grounding_confidence(
        self,
        cluster: ScanCluster,
        *,
        total_bbox_rays: int,
        detection_confidence: float | None,
    ) -> float:
        """
        Compute a conservative grounding confidence.

        The score combines:

        - geometric support,
        - cluster dominance within the bbox,
        - optional VLM confidence.

        This is not a probabilistic calibration; it is a deterministic
        quality score used to decide whether the result is sufficiently
        supported to continue.
        """

        if total_bbox_rays <= 0:
            return 0.0

        support_score = min(
            1.0,
            cluster.size / 8.0,
        )

        dominance_score = min(
            1.0,
            cluster.size / total_bbox_rays,
        )

        geometry_score = (
            0.6 * support_score
            + 0.4 * dominance_score
        )

        if detection_confidence is None:
            return geometry_score

        if not math.isfinite(detection_confidence):
            return geometry_score

        detection_score = max(
            0.0,
            min(1.0, detection_confidence),
        )

        return (
            0.7 * geometry_score
            + 0.3 * detection_score
        )

    @staticmethod
    def _compute_standoff_pose(
        x_m: float,
        y_m: float,
        *,
        standoff_distance_m: float,
        frame_id: str,
    ) -> StandoffPose | None:
        """
        Compute a navigation pose facing the grounded object.

        The robot is placed on the line from the origin toward the
        object, backed off by the configured standoff distance.
        """

        distance = math.hypot(x_m, y_m)

        if not math.isfinite(distance):
            return None

        if distance <= standoff_distance_m:
            return None

        object_yaw = math.atan2(y_m, x_m)

        goal_distance = distance - standoff_distance_m

        goal_x = goal_distance * math.cos(object_yaw)
        goal_y = goal_distance * math.sin(object_yaw)

        # The robot should face the object.
        goal_yaw = object_yaw

        return StandoffPose(
            x_m=goal_x,
            y_m=goal_y,
            yaw_rad=goal_yaw,
            frame_id=frame_id,
            standoff_distance_m=standoff_distance_m,
        )

    def ground(
        self,
        detected_object: DetectedObject,
        projected_rays: Sequence[ProjectedScanRay],
        *,
        output_frame_id: str = "base_link",
    ) -> GroundingResult:
        """
        Ground one VLM detection.

        `projected_rays` must already have been generated using the
        same camera image geometry as the VLM bounding box.

        The returned point is expressed in `output_frame_id`.
        For the current V1 pipeline this should be `base_link`.
        """

        if detected_object.bbox is None:
            return GroundingResult.failed(
                "detected object has no bounding box"
            )

        bbox = detected_object.bbox

        if not self._validate_bbox(bbox):
            return GroundingResult.failed(
                "detected object has an invalid bounding box"
            )

        if not output_frame_id:
            return GroundingResult.failed(
                "output frame ID is empty"
            )

        bbox_rays = self.filter_rays_by_bbox(
            projected_rays,
            bbox,
        )

        if not bbox_rays:
            return GroundingResult.failed(
                "no projected LaserScan rays fall inside the "
                "detected object's bounding box"
            )

        range_filtered_rays = self._filter_range(
            bbox_rays
        )

        if not range_filtered_rays:
            return GroundingResult.failed(
                "no valid LaserScan rays remain after range filtering"
            )

        clusters = self._clusterer.cluster(
            range_filtered_rays
        )

        if not clusters:
            return GroundingResult.failed(
                "no sufficiently supported LaserScan cluster "
                "inside the bounding box"
            )

        best_cluster = (
            self._clusterer.select_best_cluster(clusters)
        )

        if best_cluster is None:
            return GroundingResult.failed(
                "unable to select a geometric cluster"
            )

        if (
            best_cluster.size
            < self._config.min_supporting_rays
        ):
            return GroundingResult.failed(
                "selected cluster has insufficient LaserScan support"
            )

        grounding_confidence = (
            self._compute_grounding_confidence(
                best_cluster,
                total_bbox_rays=len(range_filtered_rays),
                detection_confidence=detected_object.confidence,
            )
        )

        if (
            grounding_confidence
            < self._config.min_grounding_confidence
        ):
            return GroundingResult.failed(
                "grounding confidence is below the configured threshold",
                metadata={
                    "grounding_confidence": grounding_confidence,
                    "supporting_rays": best_cluster.size,
                },
            )

        base_x, base_y = self._base_planar_position(
            best_cluster
        )

        distance = math.hypot(base_x, base_y)

        if not math.isfinite(distance):
            return GroundingResult.failed(
                "grounded object position is non-finite"
            )

        if distance < self._config.min_object_range_m:
            return GroundingResult.failed(
                "grounded object is closer than the minimum "
                "allowed object range"
            )

        if distance > self._config.max_object_range_m:
            return GroundingResult.failed(
                "grounded object is farther than the maximum "
                "allowed object range"
            )

        standoff_pose = self._compute_standoff_pose(
            base_x,
            base_y,
            standoff_distance_m=self._config.standoff_distance_m,
            frame_id=output_frame_id,
        )

        if standoff_pose is None:
            return GroundingResult.failed(
                "cannot compute a safe standoff pose because "
                "the object is too close to the robot"
            )

        bearing = math.atan2(
            base_y,
            base_x,
        )

        camera_x, camera_y, camera_z = (
            self._robust_point(best_cluster)
        )

        grounded_point = GroundedPoint(
            x_m=base_x,
            y_m=base_y,
            z_m=0.0,
            frame_id=output_frame_id,
            range_m=distance,
            bearing_rad=bearing,
            supporting_ray_count=best_cluster.size,
        )

        grounded_object = GroundedObject(
            label=detected_object.label,
            confidence=detected_object.confidence,
            bbox=detected_object.bbox,
            point=grounded_point,
            standoff_pose=standoff_pose,
            supporting_ray_count=best_cluster.size,
            grounding_confidence=grounding_confidence,
            metadata={
                "camera_median_x_m": camera_x,
                "camera_median_y_m": camera_y,
                "camera_median_z_m": camera_z,
                "bbox_ray_count": len(range_filtered_rays),
                "cluster_count": len(clusters),
                "cluster_min_range_m": (
                    best_cluster.min_range_m
                ),
                "cluster_max_range_m": (
                    best_cluster.max_range_m
                ),
                "cluster_median_range_m": (
                    best_cluster.median_range_m
                ),
            },
        )

        return GroundingResult.succeeded(
            grounded_object,
            metadata={
                "grounding_confidence": grounding_confidence,
                "supporting_rays": best_cluster.size,
            },
        )