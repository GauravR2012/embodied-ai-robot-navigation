#!/usr/bin/env python3
"""
Pre-flight checks for the LLM -> VLM -> Nav2 pipeline.

Run this BEFORE starting the Gradio UI (and before paying for a GPU
instance) with Gazebo + Nav2 already running. It works against the mock
gateway (tools/mock_gateway.py) or the real one.

Checks (each prints PASS / WARN / FAIL):

  1. Python dependencies the pipeline imports
  2. AI gateway reachable and models loaded            (/health)
  3. Nav2 action servers: /navigate_to_pose and /spin
  4. RGB, depth and CameraInfo topics are publishing
  5. TF: map -> base_footprint and map -> camera depth frame
  6. (--spin-test)  the robot really rotates through Nav2's /spin
  7. (--target)     the REAL perception path on the current camera view:
                    frame -> VLM -> bbox -> depth -> map position -> goal.
                    The robot does NOT move to the object.
  8. (--collect)    interactively save labeled frames (pos_ / neg_) for
                    tools/vlm_batch_check.py

Usage (from the repo root, ROS sourced):

    python3 tools/preflight_check.py
    python3 tools/preflight_check.py --target "red box" --expect 3.0 2.0
    python3 tools/preflight_check.py --spin-test
    python3 tools/preflight_check.py --collect frames/
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import math
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

class Report:
    """Collects PASS / WARN / FAIL lines and decides the exit code."""

    def __init__(self) -> None:
        self.passes = 0
        self.warnings = 0
        self.failures = 0

    def _line(self, tag: str, name: str, detail: str) -> None:
        text = f"[{tag}] {name}"

        if detail:
            text += f" - {detail}"

        print(text, flush=True)

    def ok(self, name: str, detail: str = "") -> None:
        self.passes += 1
        self._line("PASS", name, detail)

    def warn(self, name: str, detail: str = "") -> None:
        self.warnings += 1
        self._line("WARN", name, detail)

    def fail(self, name: str, detail: str = "") -> None:
        self.failures += 1
        self._line("FAIL", name, detail)

    def section(self, title: str) -> None:
        print(f"\n--- {title} ---", flush=True)

    def finish(self) -> int:
        print()
        print("=" * 66)
        print(
            f"{self.passes} passed, {self.warnings} warning(s), "
            f"{self.failures} failed"
        )
        print("=" * 66)

        return 1 if self.failures else 0


# ---------------------------------------------------------------------------
# 1. Dependencies
# ---------------------------------------------------------------------------

REQUIRED_MODULES = [
    ("rclpy", "source /opt/ros/<distro>/setup.bash"),
    ("cv2", "pip install opencv-python"),
    ("cv_bridge", "sudo apt install ros-<distro>-cv-bridge"),
    ("tf2_ros", "part of ROS 2"),
    ("nav2_msgs.action", "sudo apt install ros-<distro>-nav2-msgs"),
    ("requests", "pip install requests"),
    ("pydantic", "pip install pydantic"),
    ("yaml", "pip install pyyaml"),
]

OPTIONAL_MODULES = [
    ("gradio", "pip install gradio (needed only for the UI)"),
]


def check_dependencies(report: Report) -> None:
    report.section("1. Python dependencies")

    for module, hint in REQUIRED_MODULES:
        try:
            importlib.import_module(module)
            report.ok(module)
        except Exception as exc:
            report.fail(module, f"not importable ({exc}). Fix: {hint}")

    for module, hint in OPTIONAL_MODULES:
        try:
            importlib.import_module(module)
            report.ok(module)
        except Exception:
            report.warn(module, hint)


# ---------------------------------------------------------------------------
# 2. Gateway
# ---------------------------------------------------------------------------

def check_gateway(report: Report, url: str, timeout: float) -> None:
    report.section(f"2. AI gateway ({url})")

    try:
        from ai_client.gateway_client import AIGatewayClient

        health = AIGatewayClient(base_url=url, timeout=timeout).health()
    except Exception as exc:
        report.fail(
            "gateway /health",
            f"{exc}. Start tools/mock_gateway.py or open the SSH tunnel.",
        )
        return

    gpu = str(health.get("gpu"))

    if health.get("status") != "ok":
        report.fail("gateway status", f"{health}")
        return

    if not health.get("llm_loaded") or not health.get("vlm_loaded"):
        report.fail(
            "gateway models",
            f"llm_loaded={health.get('llm_loaded')} "
            f"vlm_loaded={health.get('vlm_loaded')}",
        )
        return

    if gpu.upper().startswith("MOCK"):
        report.ok(
            "gateway",
            "MOCK gateway (rule-based LLM + colour VLM, not the real models)",
        )
    else:
        report.ok("gateway", f"real gateway, gpu={gpu}")


# ---------------------------------------------------------------------------
# ROS helpers
# ---------------------------------------------------------------------------

def wait_future(future, timeout_sec: float) -> bool:
    """Wait for a future completed by the background spin thread."""

    deadline = time.monotonic() + timeout_sec

    while not future.done():
        if time.monotonic() > deadline:
            return False

        time.sleep(0.02)

    return True


def spin_once(client, angle_rad: float, timeout_sec: float) -> tuple[bool, str]:
    """Send one Nav2 Spin goal. Returns (success, detail)."""

    from action_msgs.msg import GoalStatus
    from builtin_interfaces.msg import Duration as DurationMsg
    from nav2_msgs.action import Spin

    goal = Spin.Goal()
    goal.target_yaw = float(angle_rad)
    goal.time_allowance = DurationMsg(sec=15, nanosec=0)

    send_future = client.send_goal_async(goal)

    if not wait_future(send_future, timeout_sec):
        return False, "timed out waiting for the goal to be accepted"

    handle = send_future.result()

    if handle is None or not handle.accepted:
        return False, "goal rejected"

    result_future = handle.get_result_async()

    if not wait_future(result_future, timeout_sec + 15.0):
        return False, "timed out waiting for the spin to finish"

    wrapped = result_future.result()

    error_code = getattr(wrapped.result, "error_code", 0)
    error_msg = getattr(wrapped.result, "error_msg", "")

    if wrapped.status != GoalStatus.STATUS_SUCCEEDED or error_code != 0:
        return False, (
            f"status={wrapped.status} error_code={error_code} "
            f"error_msg={error_msg!r}"
        )

    return True, "ok"


# ---------------------------------------------------------------------------
# 3-8. ROS checks
# ---------------------------------------------------------------------------

def run_ros_checks(args: argparse.Namespace, report: Report) -> None:
    report.section("3-8. ROS 2: Nav2, sensors, TF, perception")

    try:
        import cv2
        import numpy as np
        import rclpy
        from cv_bridge import CvBridge
        from nav2_msgs.action import NavigateToPose, Spin
        from rclpy.action import ActionClient
        from rclpy.duration import Duration
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.node import Node
        from rclpy.time import Time
        from tf2_ros import Buffer, TransformListener

        from perception.geometry import (
            compute_standoff_pose,
            yaw_from_quaternion,
        )
        from perception.object_search import (
            ObjectSearchConfig,
            ObjectSearcher,
            GroundingError,
            build_find_prompt,
            sanitize_target,
            target_tokens,
        )
        from perception.ros2.rgbd_rig import (
            RGBDRig,
            RGBDRigConfig,
            SensorTimeoutError,
        )
    except Exception as exc:
        report.fail(
            "ROS imports",
            f"{exc}. Source ROS first: source /opt/ros/<distro>/setup.bash",
        )
        return

    # -- rig configuration (command-line overrides on the defaults) -----
    overrides = {
        "map_frame": args.map_frame,
        "base_frame": args.base_frame,
        "depth_optical_frame": args.depth_optical_frame,
        "depth_body_frame": args.depth_body_frame,
        "rgb_topic": args.rgb_topic,
        "depth_topic": args.depth_topic,
        "camera_info_topic": args.camera_info_topic,
        "tf_timeout_sec": args.timeout,
        "sensor_timeout_sec": args.timeout,
    }

    rig_config = dataclasses.replace(
        RGBDRigConfig(),
        **{k: v for k, v in overrides.items() if v is not None},
    )

    rclpy.init()

    node = Node("preflight_check")
    tf_buffer = Buffer()
    TransformListener(tf_buffer, node, spin_thread=False)

    rig = RGBDRig(node, tf_buffer, rig_config)

    spin_executor = MultiThreadedExecutor(num_threads=4)
    spin_executor.add_node(node)

    spin_thread = threading.Thread(target=spin_executor.spin, daemon=True)
    spin_thread.start()

    try:
        # -- 3. Nav2 action servers -------------------------------------
        navigate_client = ActionClient(node, NavigateToPose, "/navigate_to_pose")
        spin_client = ActionClient(node, Spin, "/spin")

        for name, client in (
            ("/navigate_to_pose", navigate_client),
            ("/spin", spin_client),
        ):
            if client.wait_for_server(timeout_sec=args.timeout):
                report.ok(f"Nav2 action server {name}")
            else:
                report.fail(
                    f"Nav2 action server {name}",
                    f"not available after {args.timeout:.0f} s. Is Nav2 "
                    "running and active (lifecycle)?",
                )

        # -- 4. Sensors ---------------------------------------------------
        snapshot = None

        try:
            snapshot = rig.capture()
        except SensorTimeoutError as exc:
            report.fail("camera topics", str(exc))

        if snapshot is not None:
            bridge = CvBridge()

            rgb, depth, info = snapshot.rgb, snapshot.depth, snapshot.camera_info

            report.ok(
                "RGB image",
                f"{rgb.width}x{rgb.height} {rgb.encoding} "
                f"frame_id='{rgb.header.frame_id}'",
            )

            depth_array = np.asarray(
                bridge.imgmsg_to_cv2(depth, desired_encoding="passthrough"),
                dtype=float,
            )

            if depth.encoding in ("16UC1", "mono16"):
                depth_array = depth_array / 1000.0

            finite = depth_array[np.isfinite(depth_array) & (depth_array > 0)]

            detail = (
                f"{depth.width}x{depth.height} {depth.encoding} "
                f"frame_id='{depth.header.frame_id}'"
            )

            if finite.size:
                detail += (
                    f", {100.0 * finite.size / depth_array.size:.0f}% valid, "
                    f"range {finite.min():.2f}-{finite.max():.2f} m"
                )
                report.ok("depth image", detail)
            else:
                report.warn(
                    "depth image",
                    detail + ", no valid depth pixels (facing open space or "
                    "everything beyond the sensor range?)",
                )

            fx, fy, cx, cy = info.k[0], info.k[4], info.k[2], info.k[5]

            if fx <= 0 or fy <= 0:
                report.fail("CameraInfo", f"invalid intrinsics fx={fx} fy={fy}")
            else:
                fov = math.degrees(2.0 * math.atan(info.width / (2.0 * fx)))

                report.ok(
                    "CameraInfo",
                    f"{info.width}x{info.height} fx={fx:.1f} fy={fy:.1f} "
                    f"cx={cx:.1f} cy={cy:.1f} (horizontal FOV ~{fov:.0f} deg)",
                )

                if (info.width, info.height) != (depth.width, depth.height):
                    report.warn(
                        "CameraInfo vs depth size",
                        f"CameraInfo is {info.width}x{info.height} but the "
                        f"depth image is {depth.width}x{depth.height}; "
                        "back-projection would be wrong.",
                    )

                if fov < 50.0:
                    report.warn(
                        "camera field of view",
                        f"{fov:.0f} deg is narrower than the 45 deg search "
                        "step assumes; raise the view count or lower "
                        "rotation_step_deg.",
                    )

        # -- 5. TF ---------------------------------------------------------
        try:
            x, y, yaw = rig.robot_pose()

            report.ok(
                f"TF {rig_config.map_frame} -> {rig_config.base_frame}",
                f"x={x:.2f} y={y:.2f} yaw={math.degrees(yaw):.0f} deg",
            )
        except Exception as exc:
            report.fail(
                f"TF {rig_config.map_frame} -> {rig_config.base_frame}",
                f"{exc}. Is localization / SLAM running?",
            )

        depth_frame_used = None

        for frame, label in (
            (rig_config.depth_optical_frame, "optical (preferred)"),
            (rig_config.depth_body_frame, "body (fallback, axis-converted)"),
        ):
            try:
                transform = tf_buffer.lookup_transform(
                    rig_config.map_frame,
                    frame,
                    Time(),
                    timeout=Duration(seconds=args.timeout),
                )
            except Exception as exc:
                report.warn(
                    f"TF {rig_config.map_frame} -> {frame}",
                    f"unavailable: {exc}",
                )
                continue

            t = transform.transform.translation
            q = transform.transform.rotation

            report.ok(
                f"TF {rig_config.map_frame} -> {frame}",
                f"{label}: x={t.x:.2f} y={t.y:.2f} z={t.z:.2f} "
                f"yaw={math.degrees(yaw_from_quaternion(q.x, q.y, q.z, q.w)):.0f}"
                " deg",
            )

            if depth_frame_used is None:
                depth_frame_used = frame

            if frame == rig_config.depth_optical_frame:
                # The preferred frame works; the fallback is not needed.
                break

        if depth_frame_used is None:
            report.fail(
                "depth camera TF",
                "neither the optical nor the body frame exists, so object "
                "positions cannot be computed.",
            )

        # -- 6. Spin test --------------------------------------------------
        if args.spin_test:
            report.section("6. Spin test (the robot will turn ~15 deg and back)")

            if not spin_client.server_is_ready():
                report.fail("spin test", "/spin is not available")
            else:
                for angle in (math.radians(15.0), math.radians(-15.0)):
                    ok, detail = spin_once(spin_client, angle, args.timeout)

                    if ok:
                        report.ok(f"spin {math.degrees(angle):+.0f} deg", detail)
                    else:
                        report.fail(f"spin {math.degrees(angle):+.0f} deg", detail)
                        break

        # -- 7. Perception path -------------------------------------------
        if args.target:
            report.section(f"7. Perception path for '{args.target}'")

            if snapshot is None or depth_frame_used is None:
                report.fail(
                    "perception path",
                    "skipped: needs working sensors and TF (see above)",
                )
            else:
                run_perception_check(
                    args,
                    report,
                    rig,
                    ObjectSearchConfig(min_confidence=args.min_confidence),
                    GroundingError,
                    ObjectSearcher,
                    build_find_prompt,
                    sanitize_target,
                    target_tokens,
                    compute_standoff_pose,
                )

        # -- 8. Collect frames --------------------------------------------
        if args.collect:
            if snapshot is None:
                report.fail("collect", "skipped: no camera data")
            else:
                collect_frames(args.collect, rig, cv2, CvBridge())

    finally:
        spin_executor.shutdown(timeout_sec=2.0)
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


# ---------------------------------------------------------------------------
# 7. Real perception path (no motion)
# ---------------------------------------------------------------------------

def run_perception_check(
    args,
    report: Report,
    rig,
    search_config,
    GroundingError,
    ObjectSearcher,
    build_find_prompt,
    sanitize_target,
    target_tokens,
    compute_standoff_pose,
) -> None:
    from ai_client.gateway_client import AIGatewayClient
    from ai_client.vlm_client import RemoteVLMClient
    from perception.remote_vlm import RemoteVLMPerception

    try:
        snapshot = rig.capture(newer_than=time.monotonic())
        image_path = rig.save_rgb(snapshot)
    except Exception as exc:
        report.fail("capture frame", str(exc))
        return

    target = sanitize_target(args.target)
    prompt = build_find_prompt(target, snapshot.rgb.width, snapshot.rgb.height)

    perception = RemoteVLMPerception(
        RemoteVLMClient(
            AIGatewayClient(base_url=args.gateway_url, timeout=args.vlm_timeout)
        )
    )

    started = time.monotonic()

    try:
        perceived = perception.analyze(image_path=image_path, instruction=prompt)
    except Exception as exc:
        report.fail("VLM request", str(exc))
        return

    report.ok(
        "VLM request",
        f"{time.monotonic() - started:.2f} s, {len(perceived.objects)} "
        f"object(s), scene={perceived.scene_description!r}",
    )

    for detected in perceived.objects:
        print(
            f"        {detected.label!r} confidence={detected.confidence} "
            f"bbox={detected.bbox}"
        )

    searcher = ObjectSearcher(None, search_config)  # extract_candidates only
    candidates = searcher.extract_candidates(
        perceived,
        target_tokens(target),
        0,
        0.0,
    )

    if not candidates:
        report.warn(
            "target detection",
            f"'{target}' not detected in the current view. Point the robot "
            "at it (or this is the correct 'not found' behaviour).",
        )
        return

    best = candidates[0]

    if best.confidence >= search_config.min_confidence:
        report.ok(
            "confidence gate",
            f"{best.confidence:.2f} >= {search_config.min_confidence:.2f}: "
            "the robot would accept this detection",
        )
    else:
        report.warn(
            "confidence gate",
            f"{best.confidence:.2f} < {search_config.min_confidence:.2f}: "
            "the robot would reject it and keep searching",
        )

    try:
        grounded = rig.ground(snapshot, best.bbox)
    except GroundingError as exc:
        report.fail("depth grounding", f"not measurable: {exc}")
        return
    except Exception as exc:
        report.fail("depth grounding", str(exc))
        return

    report.ok(
        "depth grounding",
        f"object at map ({grounded.map_x:.2f}, {grounded.map_y:.2f}), "
        f"depth {grounded.depth_m:.2f} m, point frame "
        f"'{grounded.point_frame}'",
    )

    try:
        rx, ry, _ = rig.robot_pose()

        goal_x, goal_y, goal_yaw, within = compute_standoff_pose(
            (rx, ry),
            (grounded.map_x, grounded.map_y),
            0.8,
        )

        report.ok(
            "navigation goal",
            f"({goal_x:.2f}, {goal_y:.2f}) yaw={math.degrees(goal_yaw):.0f} deg"
            + (" (already within standoff)" if within else "")
            + "  [not sent: this check never moves the robot]",
        )
    except Exception as exc:
        report.fail("navigation goal", str(exc))

    if args.expect:
        ex, ey = args.expect
        error = math.hypot(grounded.map_x - ex, grounded.map_y - ey)

        detail = (
            f"measured ({grounded.map_x:.2f}, {grounded.map_y:.2f}) vs "
            f"expected ({ex:.2f}, {ey:.2f}): error {error:.2f} m "
            f"(tolerance {args.expect_tol:.2f} m; the camera sees the "
            "near face, so ~half the object size is normal)"
        )

        if error <= args.expect_tol:
            report.ok("position accuracy", detail)
        else:
            report.fail("position accuracy", detail)


# ---------------------------------------------------------------------------
# 8. Frame collection
# ---------------------------------------------------------------------------

def collect_frames(directory: str, rig, cv2, bridge) -> None:
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)

    print(
        "\nFrame collection. Drive the robot (teleop in another terminal) or "
        "move the box, then:\n"
        "  p + Enter  save as pos_  (target IS visible)\n"
        "  n + Enter  save as neg_  (target is NOT visible)\n"
        "  q + Enter  quit\n"
    )

    counts = {"pos": 0, "neg": 0}

    while True:
        try:
            choice = input("[p/n/q] > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if choice == "q":
            break

        if choice not in ("p", "n"):
            continue

        label = "pos" if choice == "p" else "neg"

        try:
            snapshot = rig.capture(newer_than=time.monotonic())
            image = bridge.imgmsg_to_cv2(snapshot.rgb, desired_encoding="bgr8")
        except Exception as exc:
            print(f"  capture failed: {exc}")
            continue

        counts[label] += 1
        path = out / f"{label}_{counts[label]:03d}.jpg"

        cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 92])

        print(f"  saved {path}")

    print(f"\nSaved {counts['pos']} pos_ and {counts['neg']} neg_ frame(s) to {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pre-flight checks for the LLM -> VLM -> Nav2 pipeline.",
    )

    parser.add_argument("--gateway-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for each ROS resource (default 10).",
    )
    parser.add_argument(
        "--vlm-timeout",
        type=float,
        default=120.0,
        help="Seconds to wait for a VLM reply (default 120).",
    )
    parser.add_argument(
        "--target",
        help="Run the real perception path for this object, e.g. 'red box'.",
    )
    parser.add_argument(
        "--expect",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        help="Known map position of the target, to measure accuracy.",
    )
    parser.add_argument("--expect-tol", type=float, default=0.35)
    parser.add_argument("--min-confidence", type=float, default=0.60)
    parser.add_argument(
        "--spin-test",
        action="store_true",
        help="Rotate the robot 15 deg and back through Nav2 /spin.",
    )
    parser.add_argument(
        "--collect",
        metavar="DIR",
        help="Interactively save pos_/neg_ frames into DIR.",
    )
    parser.add_argument("--skip-gateway", action="store_true")
    parser.add_argument("--skip-ros", action="store_true")

    for name in (
        "map-frame",
        "base-frame",
        "depth-optical-frame",
        "depth-body-frame",
        "rgb-topic",
        "depth-topic",
        "camera-info-topic",
    ):
        parser.add_argument(f"--{name}", default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    report = Report()

    check_dependencies(report)

    if not args.skip_gateway:
        check_gateway(report, args.gateway_url, args.timeout)

    if not args.skip_ros:
        run_ros_checks(args, report)

    return report.finish()


if __name__ == "__main__":
    raise SystemExit(main())
