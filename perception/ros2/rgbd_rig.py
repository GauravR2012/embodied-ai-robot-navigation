from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from cv_bridge import CvBridge
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image

from perception.geometry import (
    central_patch,
    median_valid_depth,
    optical_to_body,
    pixel_to_optical_point,
    scale_bbox_to_image,
    transform_point,
    yaw_from_quaternion,
)
from perception.models import BoundingBox
from perception.object_search import GroundingError


_GZ_PREFIX = (
    "/world/embodied_ai_boxes/model/turtlebot3_waffle/"
    "link/camera_link/sensor/intel_realsense_r200_depth"
)


class SensorTimeoutError(RuntimeError):
    """Raised when RGB / depth / CameraInfo data does not arrive."""


@dataclass(frozen=True)
class RGBDRigConfig:
    """Topics, frames and limits for the RGB-D rig."""

    rgb_topic: str = "/camera/image_raw"
    depth_topic: str = f"{_GZ_PREFIX}/depth_image"
    camera_info_topic: str = f"{_GZ_PREFIX}/camera_info"

    # Depth back-projection produces an OPTICAL-convention point
    # (x right, y down, z forward), so it must be expressed in the
    # optical frame. The TurtleBot3 URDF defines this frame as a child of
    # ``camera_depth_frame`` with the correct optical rotation.
    depth_optical_frame: str = "camera_depth_optical_frame"

    # Fallback if the optical frame is not in the TF tree. This frame is
    # NOT optical (x forward), so the point is converted first.
    depth_body_frame: str = "camera_depth_frame"

    map_frame: str = "map"
    base_frame: str = "base_footprint"

    min_depth_m: float = 0.05
    max_depth_m: float = 10.0

    tf_timeout_sec: float = 5.0
    sensor_timeout_sec: float = 10.0

    frame_directory: str = "/tmp/embodied_ai_frames"


@dataclass(frozen=True)
class RGBDSnapshot:
    """RGB + depth + intrinsics captured while the robot was stationary."""

    rgb: Image
    depth: Image
    camera_info: CameraInfo
    captured_monotonic: float


@dataclass(frozen=True)
class GroundedObject:
    """Measured position of a detected object."""

    map_x: float
    map_y: float
    depth_m: float
    depth_pixel: tuple[float, float]
    point_frame: str


class RGBDRig:
    """
    Subscribes to the RGB-D sensors on an existing node and turns a VLM
    bounding box into a map-frame position.

    The rig does not spin anything itself. The owning node must be spun
    by an executor running in another thread.
    """

    def __init__(
        self,
        node: Node,
        tf_buffer: Any,
        config: RGBDRigConfig | None = None,
    ) -> None:
        self._node = node
        self._tf = tf_buffer
        self.config = config or RGBDRigConfig()

        self._bridge = CvBridge()
        self._lock = threading.Lock()

        self._rgb: Image | None = None
        self._rgb_time = 0.0
        self._depth: Image | None = None
        self._depth_time = 0.0
        self._camera_info: CameraInfo | None = None

        node.create_subscription(
            Image,
            self.config.rgb_topic,
            self._on_rgb,
            qos_profile_sensor_data,
        )

        node.create_subscription(
            Image,
            self.config.depth_topic,
            self._on_depth,
            qos_profile_sensor_data,
        )

        node.create_subscription(
            CameraInfo,
            self.config.camera_info_topic,
            self._on_camera_info,
            qos_profile_sensor_data,
        )

        Path(self.config.frame_directory).mkdir(
            parents=True,
            exist_ok=True,
        )

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

    def _on_rgb(self, msg: Image) -> None:
        with self._lock:
            self._rgb = msg
            self._rgb_time = time.monotonic()

    def _on_depth(self, msg: Image) -> None:
        with self._lock:
            self._depth = msg
            self._depth_time = time.monotonic()

    def _on_camera_info(self, msg: CameraInfo) -> None:
        with self._lock:
            self._camera_info = msg

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def capture(
        self,
        newer_than: float | None = None,
    ) -> RGBDSnapshot:
        """
        Return an RGB + depth pair.

        If ``newer_than`` (a ``time.monotonic()`` value) is given, both
        images must have been received after it. Call this after the
        robot has stopped moving so the frames are not motion-blurred.

        Raises:
            SensorTimeoutError: if the data does not arrive in time.
        """

        deadline = time.monotonic() + self.config.sensor_timeout_sec

        while True:
            with self._lock:
                rgb = self._rgb
                depth = self._depth
                info = self._camera_info
                rgb_time = self._rgb_time
                depth_time = self._depth_time

            fresh = newer_than is None or (
                rgb_time > newer_than and depth_time > newer_than
            )

            if (
                rgb is not None
                and depth is not None
                and info is not None
                and fresh
            ):
                return RGBDSnapshot(
                    rgb=rgb,
                    depth=depth,
                    camera_info=info,
                    captured_monotonic=time.monotonic(),
                )

            if time.monotonic() > deadline:
                missing = []

                if rgb is None:
                    missing.append(self.config.rgb_topic)
                if depth is None:
                    missing.append(self.config.depth_topic)
                if info is None:
                    missing.append(self.config.camera_info_topic)

                detail = (
                    f"no data on: {', '.join(missing)}"
                    if missing
                    else "no new frames after the robot stopped"
                )

                raise SensorTimeoutError(
                    "Timed out waiting for RGB-D sensor data "
                    f"({detail})."
                )

            time.sleep(0.02)

    def save_rgb(self, snapshot: RGBDSnapshot) -> Path:
        """Write the RGB frame to disk (the VLM client reads a file)."""

        image = self._bridge.imgmsg_to_cv2(
            snapshot.rgb,
            desired_encoding="bgr8",
        )

        path = (
            Path(self.config.frame_directory)
            / f"vlm_search_{int(time.time() * 1000)}.jpg"
        )

        if not cv2.imwrite(
            str(path),
            image,
            [cv2.IMWRITE_JPEG_QUALITY, 90],
        ):
            raise RuntimeError(f"Failed to save RGB frame: {path}")

        return path

    # ------------------------------------------------------------------
    # Robot pose
    # ------------------------------------------------------------------

    def robot_pose(self) -> tuple[float, float, float]:
        """Robot (x, y, yaw) in the map frame, from TF."""

        transform = self._tf.lookup_transform(
            self.config.map_frame,
            self.config.base_frame,
            Time(),
            timeout=Duration(seconds=self.config.tf_timeout_sec),
        )

        translation = transform.transform.translation
        rotation = transform.transform.rotation

        return (
            float(translation.x),
            float(translation.y),
            yaw_from_quaternion(
                rotation.x,
                rotation.y,
                rotation.z,
                rotation.w,
            ),
        )

    # ------------------------------------------------------------------
    # Grounding
    # ------------------------------------------------------------------

    def ground(
        self,
        snapshot: RGBDSnapshot,
        bbox: BoundingBox,
    ) -> GroundedObject:
        """
        Convert a VLM bounding box into a map-frame position.

        Raises:
            GroundingError: bbox / depth cannot give a position
                (object out of depth range, no valid depth, ...).
            RuntimeError: infrastructure problem (TF, CameraInfo).
        """

        config = self.config

        depth_image = np.asarray(
            self._bridge.imgmsg_to_cv2(
                snapshot.depth,
                desired_encoding="passthrough",
            )
        )

        if depth_image.ndim != 2:
            raise RuntimeError(
                "Expected a single-channel depth image, "
                f"got shape {depth_image.shape}."
            )

        depth_h, depth_w = depth_image.shape[:2]

        try:
            box = scale_bbox_to_image(
                (bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max),
                (snapshot.rgb.width, snapshot.rgb.height),
                (depth_w, depth_h),
            )
        except ValueError as exc:
            raise GroundingError(str(exc)) from exc

        # Try the centre of the box first, then the whole box.
        px0, py0, px1, py1 = central_patch(box, (depth_w, depth_h))
        depth_value = median_valid_depth(
            depth_image[py0:py1, px0:px1],
            config.min_depth_m,
            config.max_depth_m,
        )

        if depth_value is None:
            bx0, by0, bx1, by1 = box
            depth_value = median_valid_depth(
                depth_image[by0:by1, bx0:bx1],
                config.min_depth_m,
                config.max_depth_m,
            )

        if depth_value is None:
            raise GroundingError(
                "no valid depth inside the detected bounding box"
            )

        # Gazebo depth is 32FC1 in metres; real sensors often use
        # 16UC1 / mono16 in millimetres.
        if snapshot.depth.encoding in ("16UC1", "mono16"):
            depth_m = depth_value / 1000.0
        else:
            depth_m = depth_value

        # CameraInfo belongs to the depth camera, so use depth pixels.
        k = snapshot.camera_info.k

        u = (box[0] + box[2]) / 2.0
        v = (box[1] + box[3]) / 2.0

        optical_point = pixel_to_optical_point(
            u,
            v,
            depth_m,
            fx=float(k[0]),
            fy=float(k[4]),
            cx=float(k[2]),
            cy=float(k[5]),
        )

        map_x, map_y, frame_used = self._to_map(optical_point)

        return GroundedObject(
            map_x=map_x,
            map_y=map_y,
            depth_m=depth_m,
            depth_pixel=(u, v),
            point_frame=frame_used,
        )

    # ------------------------------------------------------------------

    def _to_map(
        self,
        optical_point: tuple[float, float, float],
    ) -> tuple[float, float, str]:
        """Express an optical-convention point in the map frame."""

        config = self.config

        # Preferred: the optical frame itself.
        try:
            transform = self._tf.lookup_transform(
                config.map_frame,
                config.depth_optical_frame,
                Time(),
                timeout=Duration(seconds=config.tf_timeout_sec),
            )

            point = optical_point
            frame_used = config.depth_optical_frame

        except Exception as optical_exc:
            # Fallback: the non-optical body frame.
            try:
                transform = self._tf.lookup_transform(
                    config.map_frame,
                    config.depth_body_frame,
                    Time(),
                    timeout=Duration(seconds=config.tf_timeout_sec),
                )
            except Exception as body_exc:
                raise RuntimeError(
                    "TF lookup failed for both "
                    f"'{config.depth_optical_frame}' ({optical_exc}) and "
                    f"'{config.depth_body_frame}' ({body_exc})."
                ) from body_exc

            point = optical_to_body(optical_point)
            frame_used = config.depth_body_frame

        translation = transform.transform.translation
        rotation = transform.transform.rotation

        map_x, map_y, _map_z = transform_point(
            (translation.x, translation.y, translation.z),
            (rotation.x, rotation.y, rotation.z, rotation.w),
            point,
        )

        return map_x, map_y, frame_used
