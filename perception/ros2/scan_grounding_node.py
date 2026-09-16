from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener

from perception.models import BoundingBox, DetectedObject
from perception.grounding import GroundingResult, ObjectGrounder
from perception.grounding.lidar_projector import (
    CameraIntrinsics,
    LaserScanProjector,
)


class ScanGroundingNode(Node):
    """
    ROS 2 adapter for camera/LiDAR object grounding.

    The node:
      1. Receives LaserScan data.
      2. Looks up the LiDAR -> camera TF.
      3. Projects valid scan rays into the camera image.
      4. Grounds a VLM-style 2D detection using ObjectGrounder.

    A deterministic synthetic test mode is provided so the complete
    scan -> TF -> projection -> grounding path can be exercised before
    connecting the VLM.
    """

    def __init__(self) -> None:
        super().__init__("scan_grounding_node")

        # ------------------------------------------------------------------
        # ROS parameters
        # ------------------------------------------------------------------

        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter(
            "camera_frame",
            "camera_rgb_optical_frame",
        )
        self.declare_parameter("output_frame", "base_link")

        self.declare_parameter("image_width", 640)
        self.declare_parameter("image_height", 480)
        self.declare_parameter("horizontal_fov_rad", 1.047)

        self.declare_parameter("projector_min_range_m", 0.05)
        self.declare_parameter("projector_max_range_m", 20.0)
        self.declare_parameter("projector_min_camera_depth_m", 0.05)

        # Deterministic live integration test.
        self.declare_parameter("synthetic_test_enabled", False)
        self.declare_parameter("synthetic_test_timer_period_s", 1.0)

        # ------------------------------------------------------------------
        # Parameter values
        # ------------------------------------------------------------------

        self.scan_topic = str(
            self.get_parameter("scan_topic").value
        )
        self.camera_frame = str(
            self.get_parameter("camera_frame").value
        )
        self.output_frame = str(
            self.get_parameter("output_frame").value
        )

        self.image_width = int(
            self.get_parameter("image_width").value
        )
        self.image_height = int(
            self.get_parameter("image_height").value
        )
        self.horizontal_fov_rad = float(
            self.get_parameter("horizontal_fov_rad").value
        )

        self.projector_min_range_m = float(
            self.get_parameter("projector_min_range_m").value
        )
        self.projector_max_range_m = float(
            self.get_parameter("projector_max_range_m").value
        )
        self.projector_min_camera_depth_m = float(
            self.get_parameter("projector_min_camera_depth_m").value
        )

        self.synthetic_test_enabled = bool(
            self.get_parameter("synthetic_test_enabled").value
        )
        self.synthetic_test_timer_period_s = float(
            self.get_parameter("synthetic_test_timer_period_s").value
        )

        # ------------------------------------------------------------------
        # Camera intrinsics
        # ------------------------------------------------------------------

        fx = self.image_width / (
            2.0 * math.tan(self.horizontal_fov_rad / 2.0)
        )

        # The simulated camera uses square pixels.
        fy = fx

        cx = (self.image_width - 1) / 2.0
        cy = (self.image_height - 1) / 2.0

        self.camera_intrinsics = CameraIntrinsics(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=self.image_width,
            height=self.image_height,
        )

        # ------------------------------------------------------------------
        # ROS state
        # ------------------------------------------------------------------

        self.latest_scan: LaserScan | None = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
        )

        self.grounder = ObjectGrounder()

        self.scan_subscription = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self._scan_callback,
            10,
        )

        self._synthetic_test_completed = False
        self._synthetic_test_timer = None

        if self.synthetic_test_enabled:
            self._synthetic_test_timer = self.create_timer(
                self.synthetic_test_timer_period_s,
                self._synthetic_grounding_callback,
            )

        self.get_logger().info(
            "Scan grounding node initialized."
        )
        self.get_logger().info(
            f"Scan topic: {self.scan_topic}"
        )
        self.get_logger().info(
            f"Camera frame: {self.camera_frame}"
        )
        self.get_logger().info(
            f"Output frame: {self.output_frame}"
        )
        self.get_logger().info(
            "Camera intrinsics: "
            f"fx={fx:.3f}, "
            f"fy={fy:.3f}, "
            f"cx={cx:.3f}, "
            f"cy={cy:.3f}, "
            f"size={self.image_width}x{self.image_height}"
        )

        if self.synthetic_test_enabled:
            self.get_logger().info(
                "Synthetic grounding test ENABLED."
            )

    # ------------------------------------------------------------------
    # Scan handling
    # ------------------------------------------------------------------

    def _scan_callback(self, msg: LaserScan) -> None:
        self.latest_scan = msg

    # ------------------------------------------------------------------
    # TF handling
    # ------------------------------------------------------------------

    @staticmethod
    def _quaternion_to_rotation_matrix(
        x: float,
        y: float,
        z: float,
        w: float,
    ) -> np.ndarray:
        """
        Convert quaternion (x, y, z, w) to a 3x3 rotation matrix.
        """

        norm = math.sqrt(
            x * x
            + y * y
            + z * z
            + w * w
        )

        if norm <= 1e-12:
            raise ValueError(
                "Cannot construct rotation matrix from zero quaternion."
            )

        x /= norm
        y /= norm
        z /= norm
        w /= norm

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

    @classmethod
    def _transform_to_matrix(
        cls,
        transform: TransformStamped,
    ) -> np.ndarray:
        """
        Convert geometry_msgs/TransformStamped to a 4x4 homogeneous matrix.

        The returned transform maps points from the source frame to the
        target frame.
        """

        translation = transform.transform.translation
        rotation = transform.transform.rotation

        rotation_matrix = cls._quaternion_to_rotation_matrix(
            rotation.x,
            rotation.y,
            rotation.z,
            rotation.w,
        )

        matrix = np.eye(4, dtype=float)
        matrix[:3, :3] = rotation_matrix
        matrix[:3, 3] = [
            translation.x,
            translation.y,
            translation.z,
        ]

        return matrix

    def _lookup_scan_to_camera_transform(
        self,
        scan: LaserScan,
    ) -> np.ndarray | None:
        """
        Look up the transform:

            scan frame -> camera optical frame

        TF2's lookup_transform(target, source, time) therefore uses:

            target = camera_frame
            source = scan.header.frame_id
        """

        scan_frame = scan.header.frame_id

        if not scan_frame:
            self.get_logger().warning(
                "LaserScan has an empty frame_id."
            )
            return None

        try:
            transform = self.tf_buffer.lookup_transform(
                self.camera_frame,
                scan_frame,
                scan.header.stamp,
            )

            return self._transform_to_matrix(transform)

        except Exception as exc:
            self.get_logger().warning(
                "Unable to lookup TF "
                f"{scan_frame} -> {self.camera_frame}: {exc}"
            )
            return None

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------

    def _build_projector(
        self,
        scan_to_camera_transform: Sequence[Sequence[float]],
    ) -> LaserScanProjector:
        return LaserScanProjector(
            self.camera_intrinsics,
            scan_to_camera_transform,
            min_range_m=self.projector_min_range_m,
            max_range_m=self.projector_max_range_m,
            min_camera_depth_m=self.projector_min_camera_depth_m,
        )

    def project_latest_scan(self):
        """
        Project the latest LaserScan into the camera image.

        Returns:
            tuple[ProjectedScanRay, ...] on success,
            None when scan/TF is unavailable.
        """

        scan = self.latest_scan

        if scan is None:
            self.get_logger().warning(
                "No LaserScan received yet."
            )
            return None

        transform = self._lookup_scan_to_camera_transform(scan)

        if transform is None:
            return None

        projector = self._build_projector(transform)

        projected_rays = projector.project_scan(
            scan.ranges,
            angle_min=scan.angle_min,
            angle_increment=scan.angle_increment,
            range_min=scan.range_min,
            range_max=scan.range_max,
        )

        return projected_rays

    # ------------------------------------------------------------------
    # Object grounding
    # ------------------------------------------------------------------

    def ground_detection(
        self,
        detected_object: DetectedObject,
    ) -> GroundingResult:
        """
        Ground a 2D detection using the latest LaserScan.
        """

        projected_rays = self.project_latest_scan()

        if projected_rays is None:
            return GroundingResult.failed(
                "Unable to project latest LaserScan."
            )

        result = self.grounder.ground(
            detected_object,
            projected_rays,
            output_frame_id=self.output_frame,
        )

        return result

    # ------------------------------------------------------------------
    # Synthetic deterministic integration test
    # ------------------------------------------------------------------

    @staticmethod
    def _find_contiguous_range_runs(
        projected_rays,
        *,
        max_range_jump_m: float = 0.35,
    ):
        """
        Find runs of projected rays that satisfy the same basic continuity
        assumptions used by ScanPointClusterer.

        A run continues when:
          - scan indices are consecutive
          - adjacent ranges do not jump by more than max_range_jump_m
        """

        if not projected_rays:
            return []

        runs = []
        current = [projected_rays[0]]

        for ray in projected_rays[1:]:
            previous = current[-1]

            consecutive_index = (
                ray.scan_index == previous.scan_index + 1
            )

            range_continuous = (
                abs(ray.range_m - previous.range_m)
                <= max_range_jump_m
            )

            if consecutive_index and range_continuous:
                current.append(ray)
            else:
                if len(current) >= 2:
                    runs.append(current)

                current = [ray]

        if len(current) >= 2:
            runs.append(current)

        return runs

    def _make_synthetic_bbox(
        self,
        projected_rays,
    ) -> BoundingBox | None:
        """
        Build a deterministic synthetic detection bbox around real projected
        scan rays.

        This is intentionally derived from actual sensor data rather than
        choosing an arbitrary image region. Therefore the bbox is guaranteed
        to overlap a real projected scan cluster whenever a suitable cluster
        exists.
        """

        runs = self._find_contiguous_range_runs(
            projected_rays,
            max_range_jump_m=0.35,
        )

        if not runs:
            return None

        # Prefer the longest geometrically continuous run.
        best_run = max(
            runs,
            key=len,
        )

        # Use a small segment so that the synthetic detection represents
        # one local object region rather than the entire scan.
        segment_length = min(5, len(best_run))

        if len(best_run) > segment_length:
            start = (
                len(best_run) - segment_length
            ) // 2
            segment = best_run[
                start : start + segment_length
            ]
        else:
            segment = best_run

        u_values = [ray.pixel_u for ray in segment]
        v_values = [ray.pixel_v for ray in segment]

        # Generous margins make the test robust to pixel quantization.
        horizontal_margin_px = 8.0
        vertical_margin_px = 25.0

        x_min = max(
            0.0,
            min(u_values) - horizontal_margin_px,
        )
        y_min = max(
            0.0,
            min(v_values) - vertical_margin_px,
        )
        x_max = min(
            float(self.image_width - 1),
            max(u_values) + horizontal_margin_px,
        )
        y_max = min(
            float(self.image_height - 1),
            max(v_values) + vertical_margin_px,
        )

        if x_max <= x_min or y_max <= y_min:
            return None

        return BoundingBox(
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
        )

    def _synthetic_grounding_callback(self) -> None:
        """
        Run the complete grounding pipeline once using a synthetic detection.

        The detection bbox is generated from actual projected LaserScan
        measurements, so this exercises:

            /scan
              -> TF
              -> LaserScanProjector
              -> image-space rays
              -> synthetic VLM detection
              -> ObjectGrounder
              -> GroundedPoint
              -> StandoffPose
        """

        if self._synthetic_test_completed:
            return

        if self.latest_scan is None:
            return

        projected_rays = self.project_latest_scan()

        if projected_rays is None:
            return

        if len(projected_rays) < 2:
            self.get_logger().warning(
                "Synthetic grounding test: fewer than 2 projected "
                "rays are available."
            )
            return

        bbox = self._make_synthetic_bbox(
            projected_rays
        )

        if bbox is None:
            self.get_logger().warning(
                "Synthetic grounding test: could not find a "
                "contiguous projected ray cluster."
            )
            return

        detected_object = DetectedObject(
            label="synthetic_test_object",
            confidence=1.0,
            bbox=bbox,
        )

        self.get_logger().info(
            "Synthetic grounding test:"
            f" projected_rays={len(projected_rays)},"
            f" bbox=("
            f"{bbox.x_min:.1f}, "
            f"{bbox.y_min:.1f}, "
            f"{bbox.x_max:.1f}, "
            f"{bbox.y_max:.1f})"
        )

        result = self.ground_detection(
            detected_object
        )

        if result.success and result.object is not None:
            grounded = result.object
            point = grounded.point
            pose = grounded.standoff_pose

            self.get_logger().info(
                "=================================================="
            )
            self.get_logger().info(
                "SYNTHETIC GROUNDING TEST: SUCCESS"
            )
            self.get_logger().info(
                f"label={grounded.label}"
            )
            self.get_logger().info(
                f"supporting_rays={grounded.supporting_ray_count}"
            )
            self.get_logger().info(
                f"grounding_confidence="
                f"{grounded.grounding_confidence:.3f}"
            )
            self.get_logger().info(
                "Grounded point "
                f"[{point.frame_id}]: "
                f"x={point.x_m:.3f} m, "
                f"y={point.y_m:.3f} m, "
                f"z={point.z_m:.3f} m"
            )
            self.get_logger().info(
                "Standoff pose "
                f"[{pose.frame_id}]: "
                f"x={pose.x_m:.3f} m, "
                f"y={pose.y_m:.3f} m, "
                f"yaw={pose.yaw_rad:.3f} rad, "
                f"distance={pose.standoff_distance_m:.3f} m"
            )
            self.get_logger().info(
                "=================================================="
            )

            self._synthetic_test_completed = True

        else:
            self.get_logger().warning(
                "=================================================="
            )
            self.get_logger().warning(
                "SYNTHETIC GROUNDING TEST: FAILED"
            )
            self.get_logger().warning(
                f"reason={result.reason}"
            )
            self.get_logger().warning(
                "=================================================="
            )

            # Do not permanently mark the test complete on failure.
            # The next timer iteration can retry with fresh sensor data.

    # ------------------------------------------------------------------
    # Logging helper
    # ------------------------------------------------------------------

    def log_grounding_result(
        self,
        result: GroundingResult,
    ) -> None:
        """
        Human-readable logging helper for future VLM integration.
        """

        if not result.success:
            self.get_logger().warning(
                f"Grounding failed: {result.reason}"
            )
            return

        if result.object is None:
            self.get_logger().warning(
                "Grounding reported success but no object was returned."
            )
            return

        grounded = result.object
        point = grounded.point
        pose = grounded.standoff_pose

        self.get_logger().info(
            f"Grounded '{grounded.label}': "
            f"point=({point.x_m:.3f}, "
            f"{point.y_m:.3f}, "
            f"{point.z_m:.3f}) "
            f"{point.frame_id}, "
            f"support={grounded.supporting_ray_count}, "
            f"confidence={grounded.grounding_confidence:.3f}"
        )

        self.get_logger().info(
            f"Standoff pose: "
            f"({pose.x_m:.3f}, "
            f"{pose.y_m:.3f}, "
            f"yaw={pose.yaw_rad:.3f}) "
            f"{pose.frame_id}"
        )


def main(args=None) -> None:
    rclpy.init(args=args)

    node = ScanGroundingNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()