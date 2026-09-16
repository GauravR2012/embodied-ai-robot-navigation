#!/usr/bin/env python3

from __future__ import annotations

import math
import sys
import traceback

import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener

from perception.grounding.lidar_projector import (
    CameraIntrinsics,
    LaserScanPoint,
    LaserScanProjector,
)
from perception.grounding.models import GroundingResult
from perception.grounding.object_grounder import ObjectGrounder
from perception.grounding.point_clusterer import ScanPointClusterer
from perception.models import BoundingBox, DetectedObject


class DetectionToGoalTest(Node):
    """
    Real integration test for:

        pseudo VLM detection
                |
                | label + bbox only
                v
        ROS LaserScan
                |
                v
        TF + LiDAR projection
                |
                v
        point clustering
                |
                v
        object grounding
                |
                v
        standoff pose
                |
                v
        Nav2 NavigateToPose
                |
                v
        robot navigation

    IMPORTANT:

    Only the VLM output is simulated.

    Everything after the simulated bounding box is the actual
    perception/navigation pipeline.
    """

    def __init__(self) -> None:
        super().__init__("detection_to_goal_test")

        # ---------------------------------------------------------
        # ROS PARAMETERS
        # ---------------------------------------------------------

        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter(
            "camera_frame",
            "camera_rgb_optical_frame",
        )
        self.declare_parameter(
            "scan_frame",
            "base_scan",
        )
        self.declare_parameter(
            "output_frame",
            "base_link",
        )
        self.declare_parameter(
            "scan_timeout",
            10.0,
        )

        self.scan_topic = (
            self.get_parameter("scan_topic")
            .get_parameter_value()
            .string_value
        )

        self.camera_frame = (
            self.get_parameter("camera_frame")
            .get_parameter_value()
            .string_value
        )

        self.scan_frame = (
            self.get_parameter("scan_frame")
            .get_parameter_value()
            .string_value
        )

        self.output_frame = (
            self.get_parameter("output_frame")
            .get_parameter_value()
            .string_value
        )

        self.scan_timeout = (
            self.get_parameter("scan_timeout")
            .get_parameter_value()
            .double_value
        )

        # ---------------------------------------------------------
        # NAV2
        # ---------------------------------------------------------

        self.nav_client = ActionClient(
            self,
            NavigateToPose,
            "/navigate_to_pose",
        )

        # ---------------------------------------------------------
        # TF
        # ---------------------------------------------------------

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
        )

        # ---------------------------------------------------------
        # LASER SCAN
        # ---------------------------------------------------------

        self.latest_scan: LaserScan | None = None

        self.scan_subscription = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self._scan_callback,
            10,
        )

        # ---------------------------------------------------------
        # CAMERA INTRINSICS
        # ---------------------------------------------------------
        #
        # These are derived from the camera we added to the
        # TurtleBot Gazebo model:
        #
        # width       = 640
        # height      = 480
        # horizontal FOV = 1.047 rad
        #
        # fx = width / (2*tan(horizontal_fov/2))
        #
        # For a centered pinhole camera:
        #
        # fx ≈ 554.38
        # fy ≈ 554.38
        # cx = 319.5
        # cy = 239.5
        #

        self.camera_intrinsics = CameraIntrinsics(
            fx=554.383,
            fy=554.383,
            cx=319.5,
            cy=239.5,
            width=640,
            height=480,
        )

        # ---------------------------------------------------------
        # REAL GROUNDING PIPELINE
        # ---------------------------------------------------------

        self.point_clusterer = ScanPointClusterer()

        self.object_grounder = ObjectGrounder()

        # ---------------------------------------------------------
        # FAKE VLM OUTPUT
        # ---------------------------------------------------------
        #
        # This is the ONLY part of the test that is simulated.
        #
        # The VLM would normally produce this:
        #
        # {
        #     "label": "red_box",
        #     "bbox": {
        #         "x_min": ...,
        #         "y_min": ...,
        #         "x_max": ...,
        #         "y_max": ...
        #     }
        # }
        #
        # We initially leave the bbox unset because we determine
        # it from the real LiDAR projection once the camera/scan
        # pipeline is running.
        #
        # This means the test can locate a visible projected
        # LiDAR cluster and construct a deterministic pseudo-VLM
        # bbox around it.
        #

        self.fake_label = "red_box"
        self.fake_confidence = 1.0

    def _scan_callback(self, msg: LaserScan) -> None:
        """Store the most recent real ROS LaserScan."""

        self.latest_scan = msg

    # ------------------------------------------------------------------
    # STEP 1: WAIT FOR REAL SCAN
    # ------------------------------------------------------------------

    def wait_for_scan(self) -> LaserScan | None:
        print()
        print("[1/7] Waiting for real ROS LaserScan...")
        print(f"      Topic: {self.scan_topic}")

        start = self.get_clock().now()

        while rclpy.ok():
            rclpy.spin_once(
                self,
                timeout_sec=0.1,
            )

            if self.latest_scan is not None:
                print("      Received real /scan message.")
                return self.latest_scan

            elapsed = (
                self.get_clock().now() - start
            ).nanoseconds / 1e9

            if elapsed >= self.scan_timeout:
                print()
                print("FAILED")
                print(
                    f"No LaserScan received on "
                    f"{self.scan_topic} within "
                    f"{self.scan_timeout:.1f} seconds."
                )
                return None

        return None

    # ------------------------------------------------------------------
    # STEP 2: GET TF
    # ------------------------------------------------------------------

    def get_lidar_to_camera_transform(
        self,
        scan: LaserScan,
    ):
        print()
        print("[2/7] Looking up LiDAR → camera TF")

        source_frame = scan.header.frame_id or self.scan_frame

        print(f"      Source frame: {source_frame}")
        print(f"      Camera frame: {self.camera_frame}")

        try:
            transform = self.tf_buffer.lookup_transform(
                self.camera_frame,
                source_frame,
                rclpy.time.Time(),
            )
        except Exception as exc:
            print()
            print("FAILED")
            print(
                "Could not obtain transform from "
                f"{source_frame} to {self.camera_frame}."
            )
            print(f"Exception: {type(exc).__name__}: {exc}")
            return None

        print("      TF available.")

        return transform

    # ------------------------------------------------------------------
    # STEP 3: PROJECT REAL LASER SCAN
    # ------------------------------------------------------------------

    def project_scan(
        self,
        scan: LaserScan,
        transform_msg,
    ):
        print()
        print("[3/7] Projecting real LiDAR scan into camera")

        translation = transform_msg.transform.translation
        rotation = transform_msg.transform.rotation

        # Quaternion -> rotation matrix.
        qx = rotation.x
        qy = rotation.y
        qz = rotation.z
        qw = rotation.w

        rotation_matrix = [
            [
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qz * qw),
                2.0 * (qx * qz + qy * qw),
            ],
            [
                2.0 * (qx * qy + qz * qw),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qx * qw),
            ],
            [
                2.0 * (qx * qz - qy * qw),
                2.0 * (qy * qz + qx * qw),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ],
        ]

        transform = [
            [
                rotation_matrix[0][0],
                rotation_matrix[0][1],
                rotation_matrix[0][2],
                translation.x,
            ],
            [
                rotation_matrix[1][0],
                rotation_matrix[1][1],
                rotation_matrix[1][2],
                translation.y,
            ],
            [
                rotation_matrix[2][0],
                rotation_matrix[2][1],
                rotation_matrix[2][2],
                translation.z,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ]

        projector = LaserScanProjector(
            camera_intrinsics=self.camera_intrinsics,
            lidar_to_camera_transform=transform,
        )

        projected_rays = projector.project_scan(
            scan.ranges,
            angle_min=scan.angle_min,
            angle_increment=scan.angle_increment,
            range_min=scan.range_min,
            range_max=scan.range_max,
        )

        print(
            f"      Valid projected rays: "
            f"{len(projected_rays)}"
        )

        return projected_rays

    # ------------------------------------------------------------------
    # STEP 4: FAKE VLM BBOX
    # ------------------------------------------------------------------

    def create_pseudo_detection(
        self,
        projected_rays,
    ) -> DetectedObject | None:
        print()
        print("[4/7] Creating pseudo-VLM detection")

        if not projected_rays:
            print("FAILED: No projected LiDAR rays.")
            return None

        # ----------------------------------------------------------
        # Find a contiguous projected ray sequence.
        #
        # This is NOT pretending to be the VLM.
        #
        # We simply need a deterministic bbox for the test.
        #
        # The bbox represents what a VLM would have returned.
        # ----------------------------------------------------------

        sorted_rays = sorted(
            projected_rays,
            key=lambda ray: ray.scan_index,
        )

        cluster = []

        for ray in sorted_rays:
            if not cluster:
                cluster = [ray]
                continue

            previous = cluster[-1]

            if ray.scan_index == previous.scan_index + 1:
                cluster.append(ray)
            else:
                if len(cluster) >= 3:
                    break
                cluster = [ray]

        if len(cluster) < 3:
            print(
                "FAILED: Could not find a contiguous projected "
                "LiDAR region suitable for the pseudo detection."
            )
            return None

        # ----------------------------------------------------------
        # Construct a camera bbox around the selected rays.
        # Add a margin because a real VLM bbox would normally
        # contain more than the exact LiDAR points.
        # ----------------------------------------------------------

        margin_x = 35.0
        margin_y = 35.0

        x_min = max(
            0.0,
            min(ray.pixel_u for ray in cluster) - margin_x,
        )

        x_max = min(
            float(self.camera_intrinsics.width - 1),
            max(ray.pixel_u for ray in cluster) + margin_x,
        )

        y_min = max(
            0.0,
            min(ray.pixel_v for ray in cluster) - margin_y,
        )

        y_max = min(
            float(self.camera_intrinsics.height - 1),
            max(ray.pixel_v for ray in cluster) + margin_y,
        )

        bbox = BoundingBox(
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
        )

        detection = DetectedObject(
            label=self.fake_label,
            confidence=self.fake_confidence,
            bbox=bbox,
        )

        print(f"      Label: {detection.label}")
        print(
            f"      Confidence: "
            f"{detection.confidence:.2f}"
        )
        print(
            "      Fake VLM bbox: "
            f"({bbox.x_min:.1f}, "
            f"{bbox.y_min:.1f}, "
            f"{bbox.x_max:.1f}, "
            f"{bbox.y_max:.1f})"
        )

        print(
            f"      Bbox-supporting projected rays available: "
            f"{len(cluster)}"
        )

        return detection

    # ------------------------------------------------------------------
    # STEP 5: REAL OBJECT GROUNDING
    # ------------------------------------------------------------------

    def ground_object(
        self,
        detection: DetectedObject,
        projected_rays,
    ) -> GroundingResult:
        print()
        print("[5/7] Running REAL object grounding")

        result = self.object_grounder.ground(
            detected_object=detection,
            projected_rays=projected_rays,
            output_frame_id=self.output_frame,
        )

        if not result.success:
            print()
            print("GROUNDING FAILED")
            print(f"      Reason: {result.reason}")
            return result

        grounded = result.object

        if grounded is None:
            print()
            print("GROUNDING FAILED")
            print("      Grounding returned no object.")
            return GroundingResult.failed(
                "Grounding succeeded without a grounded object."
            )

        print("      Grounding SUCCESS")
        print()
        print(
            f"      Grounded object: {grounded.label}"
        )
        print(
            f"      Supporting rays: "
            f"{grounded.supporting_ray_count}"
        )
        print(
            f"      Grounding confidence: "
            f"{grounded.grounding_confidence:.3f}"
        )

        print()
        print(
            f"      Object point [{grounded.point.frame_id}]"
        )
        print(
            f"          x = {grounded.point.x_m:.3f} m"
        )
        print(
            f"          y = {grounded.point.y_m:.3f} m"
        )
        print(
            f"          z = {grounded.point.z_m:.3f} m"
        )
        print(
            f"          range = {grounded.point.range_m:.3f} m"
        )
        print(
            f"          bearing = "
            f"{grounded.point.bearing_rad:.3f} rad"
        )

        print()
        print("      Generated standoff pose")
        print(
            f"          x = "
            f"{grounded.standoff_pose.x_m:.3f} m"
        )
        print(
            f"          y = "
            f"{grounded.standoff_pose.y_m:.3f} m"
        )
        print(
            f"          yaw = "
            f"{grounded.standoff_pose.yaw_rad:.3f} rad"
        )
        print(
            f"          distance = "
            f"{grounded.standoff_pose.standoff_distance_m:.3f} m"
        )

        return result

    # ------------------------------------------------------------------
    # STEP 6: CONVERT REAL STANDOFF POSE TO NAV2 GOAL
    # ------------------------------------------------------------------

    def create_nav2_goal(
        self,
        grounding_result: GroundingResult,
    ) -> NavigateToPose.Goal | None:
        print()
        print("[6/7] Converting grounded pose into Nav2 goal")

        if not grounding_result.success:
            print("FAILED: No successful grounding result.")
            return None

        grounded = grounding_result.object

        if grounded is None:
            print("FAILED: Grounding result contains no object.")
            return None

        standoff = grounded.standoff_pose

        # ----------------------------------------------------------
        # Grounding is currently expressed in base_link.
        #
        # Nav2 requires the goal in the global navigation frame.
        #
        # We therefore look up:
        #
        #       map -> base_link
        #
        # and transform the relative standoff position into map.
        #
        # NOTE:
        # This is intentionally deterministic and uses TF rather
        # than inventing a global coordinate.
        # ----------------------------------------------------------

        try:
            base_to_map = self.tf_buffer.lookup_transform(
                "map",
                standoff.frame_id,
                rclpy.time.Time(),
            )
        except Exception as exc:
            print()
            print("FAILED")
            print(
                "Could not transform grounded standoff pose "
                f"from {standoff.frame_id} to map."
            )
            print(
                f"Exception: {type(exc).__name__}: {exc}"
            )
            return None

        tx = base_to_map.transform.translation.x
        ty = base_to_map.transform.translation.y

        qx = base_to_map.transform.rotation.x
        qy = base_to_map.transform.rotation.y
        qz = base_to_map.transform.rotation.z
        qw = base_to_map.transform.rotation.w

        # Current TurtleBot is planar, so calculate yaw from TF.
        robot_yaw = math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )

        # Rotate base_link-relative standoff position into map.
        cos_yaw = math.cos(robot_yaw)
        sin_yaw = math.sin(robot_yaw)

        goal_x = (
            tx
            + cos_yaw * standoff.x_m
            - sin_yaw * standoff.y_m
        )

        goal_y = (
            ty
            + sin_yaw * standoff.x_m
            + cos_yaw * standoff.y_m
        )

        goal_yaw = robot_yaw + standoff.yaw_rad

        # Normalize yaw.
        goal_yaw = math.atan2(
            math.sin(goal_yaw),
            math.cos(goal_yaw),
        )

        pose = PoseStamped()

        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()

        pose.pose.position.x = goal_x
        pose.pose.position.y = goal_y
        pose.pose.position.z = 0.0

        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = math.sin(
            goal_yaw / 2.0
        )
        pose.pose.orientation.w = math.cos(
            goal_yaw / 2.0
        )

        nav_goal = NavigateToPose.Goal()
        nav_goal.pose = pose

        print("      Nav2 goal generated from REAL grounded pose.")
        print(
            f"      map x = {goal_x:.3f} m"
        )
        print(
            f"      map y = {goal_y:.3f} m"
        )
        print(
            f"      map yaw = {goal_yaw:.3f} rad"
        )

        return nav_goal

    # ------------------------------------------------------------------
    # STEP 7: SEND REAL NAV2 GOAL
    # ------------------------------------------------------------------

    def navigate(
        self,
        nav_goal: NavigateToPose.Goal,
    ) -> int:
        print()
        print("[7/7] Sending REAL goal to Nav2")

        print(
            "      Waiting for NavigateToPose action server..."
        )

        if not self.nav_client.wait_for_server(
            timeout_sec=10.0
        ):
            print()
            print("FAILED")
            print(
                "NavigateToPose action server unavailable."
            )
            return 1

        print("      Nav2 action server available.")

        try:
            future = self.nav_client.send_goal_async(
                nav_goal
            )

            rclpy.spin_until_future_complete(
                self,
                future,
                timeout_sec=10.0,
            )

            if not future.done():
                print(
                    "FAILED: Timed out waiting for Nav2 "
                    "goal response."
                )
                return 1

            goal_handle = future.result()

        except Exception as exc:
            print()
            print("FAILED: Could not send Nav2 goal.")
            print(
                f"Exception: {type(exc).__name__}: {exc!r}"
            )
            traceback.print_exc()
            return 1

        if goal_handle is None:
            print("FAILED: Nav2 returned no goal handle.")
            return 1

        if not goal_handle.accepted:
            print("FAILED: Nav2 rejected the goal.")
            return 1

        print("      Nav2 accepted the REAL grounded goal.")

        print()
        print("      Waiting for navigation result...")

        try:
            result_future = goal_handle.get_result_async()

            while (
                rclpy.ok()
                and not result_future.done()
            ):
                rclpy.spin_once(
                    self,
                    timeout_sec=0.2,
                )

                print(
                    "\r      Robot navigating...",
                    end="",
                    flush=True,
                )

            print()

            if not result_future.done():
                print(
                    "FAILED: No navigation result received."
                )
                return 1

            result = result_future.result()

        except Exception as exc:
            print()
            print(
                "FAILED while waiting for navigation result."
            )
            print(
                f"Exception: {type(exc).__name__}: {exc!r}"
            )
            traceback.print_exc()
            return 1

        if result is None:
            print("FAILED: Empty navigation result.")
            return 1

        status = result.status

        print()
        print("=" * 65)
        print(f"FINAL NAV2 STATUS: {status}")
        print("=" * 65)

        # action_msgs/msg/GoalStatus
        # STATUS_SUCCEEDED = 4
        # STATUS_CANCELED  = 5
        # STATUS_ABORTED   = 6

        if status == 4:
            print()
            print("SUCCESS")
            print()
            print(
                "Complete pipeline succeeded:"
            )
            print()
            print(
                "  pseudo VLM detection"
            )
            print(
                "        ↓"
            )
            print(
                "  real LiDAR projection"
            )
            print(
                "        ↓"
            )
            print(
                "  real object grounding"
            )
            print(
                "        ↓"
            )
            print(
                "  real standoff pose"
            )
            print(
                "        ↓"
            )
            print(
                "  real Nav2 goal"
            )
            print(
                "        ↓"
            )
            print(
                "  robot navigation"
            )
            print()
            return 0

        if status == 5:
            print()
            print("CANCELED")
            print()
            return 1

        if status == 6:
            print()
            print("ABORTED")
            print()
            return 1

        print()
        print(
            f"Navigation finished with unexpected "
            f"status {status}."
        )
        print()

        return 1

    # ------------------------------------------------------------------
    # COMPLETE TEST
    # ------------------------------------------------------------------

    def run(self) -> int:
        print()
        print("=" * 65)
        print("  REAL DETECTION → GROUNDING → GOAL → NAV2 TEST")
        print("=" * 65)
        print()
        print("Only the VLM detection is simulated.")
        print("LiDAR grounding and navigation are REAL.")
        print()

        # 1. Real scan.
        scan = self.wait_for_scan()

        if scan is None:
            return 1

        # 2. Real TF.
        transform = self.get_lidar_to_camera_transform(
            scan
        )

        if transform is None:
            return 1

        # 3. Real LiDAR projection.
        projected_rays = self.project_scan(
            scan,
            transform,
        )

        if not projected_rays:
            print()
            print(
                "FAILED: Real LiDAR produced no projected rays."
            )
            return 1

        # 4. Pseudo VLM detection.
        detection = self.create_pseudo_detection(
            projected_rays
        )

        if detection is None:
            return 1

        # 5. Real grounding.
        grounding_result = self.ground_object(
            detection,
            projected_rays,
        )

        if not grounding_result.success:
            return 1

        # 6. Real Nav2 goal.
        nav_goal = self.create_nav2_goal(
            grounding_result
        )

        if nav_goal is None:
            return 1

        # 7. Real navigation.
        return self.navigate(nav_goal)


def main() -> None:
    rclpy.init()

    node = DetectionToGoalTest()

    try:
        exit_code = node.run()

    except KeyboardInterrupt:
        print()
        print("Test interrupted.")
        exit_code = 130

    except BaseException as exc:
        print()
        print("=" * 65)
        print("UNHANDLED TEST ERROR")
        print("=" * 65)
        print()
        print(
            f"Exception type: {type(exc).__name__}"
        )
        print(
            f"Exception repr: {exc!r}"
        )
        print()
        traceback.print_exc()

        exit_code = 1

    finally:
        node.destroy_node()
        rclpy.shutdown()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()