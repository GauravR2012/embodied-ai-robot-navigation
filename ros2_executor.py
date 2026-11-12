from __future__ import annotations

import argparse
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import rclpy
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration as DurationMsg
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose, Spin
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener

from execution_context import ExecutionContext
from execution_result import ExecutionResult
from perception.geometry import compute_standoff_pose, normalize_angle
from perception.interface import PerceptionBackend
from perception.models import BoundingBox, PerceptionResult
from perception.object_search import (
    ObjectSearchConfig,
    ObjectSearcher,
    RotationError,
    SearchResult,
    build_find_prompt,
)
from task_planner import Skill
from target_resolver import Pose2D, TargetResolver, yaw_to_quaternion


class ExecutorError(RuntimeError):
    """Raised when a robot skill cannot be executed successfully."""


class ObjectNotFoundError(ExecutorError):
    """
    Raised when the requested object could not be found with enough
    confidence. No navigation goal has been sent when this is raised.
    """

    def __init__(self, search_result: SearchResult) -> None:
        super().__init__(search_result.message())
        self.search_result = search_result


@dataclass(frozen=True)
class ExecutorConfig:
    """Configuration for the ROS 2 skill executor."""

    action_name: str = "/navigate_to_pose"

    # Maximum time to wait for the Nav2 action server.
    server_timeout_sec: float = 10.0

    # Maximum time to wait for Nav2 to accept/reject a goal.
    goal_send_timeout_sec: float = 10.0

    # Maximum time to wait for a navigation result.
    result_timeout_sec: float = 300.0

    # Maximum time to wait for goal cancellation.
    cancel_timeout_sec: float = 5.0

    # Minimum distance change required before another feedback
    # message is logged.
    feedback_log_delta_m: float = 0.05

    # ------------------------------------------------------------------
    # Rotation (Nav2 behavior server "spin" action)
    # ------------------------------------------------------------------

    spin_action_name: str = "/spin"

    # Minimum time allowance given to Nav2 for one spin.
    spin_time_allowance_sec: float = 20.0

    # Let the robot come to rest before taking a picture.
    settle_time_sec: float = 0.6

    # ------------------------------------------------------------------
    # Object navigation
    # ------------------------------------------------------------------

    # Distance in front of the object at which the robot stops.
    standoff_distance_m: float = 0.8

    object_search: ObjectSearchConfig = field(
        default_factory=ObjectSearchConfig
    )

    # perception.ros2.rgbd_rig.RGBDRigConfig, or None for the defaults.
    rig_config: Any = None

    map_frame: str = "map"
    base_frame: str = "base_footprint"


class _SearchEnvironment:
    """
    Connects the ROS-free ObjectSearcher to this robot.

    One instance is created per search, so the starting heading is the
    heading the robot had when the search began.
    """

    def __init__(self, executor: "ROS2SkillExecutor") -> None:
        self._executor = executor
        self._moved = False
        self._commanded_rad = 0.0
        self._start_yaw: float | None = None

        try:
            _x, _y, self._start_yaw = executor._rig.robot_pose()
        except Exception as exc:
            executor.get_logger().warn(
                "Could not read the starting heading from TF; "
                f"will rely on commanded rotation instead: {exc}"
            )

    # -- SearchEnvironment ------------------------------------------------

    def capture(self):
        if self._moved:
            # The robot has just stopped rotating; let it come to rest so
            # the picture is sharp and the depth matches the RGB frame.
            time.sleep(self._executor.config.settle_time_sec)

        return self._executor._rig.capture(newer_than=time.monotonic())

    def detect(self, frame, target: str) -> PerceptionResult:
        executor = self._executor
        logger = executor.get_logger()

        image_path = executor._rig.save_rgb(frame)

        prompt = build_find_prompt(
            target,
            frame.rgb.width,
            frame.rgb.height,
        )

        logger.info(f"Asking the VLM to find '{target}'...")

        result = executor._perception.analyze(
            image_path=image_path,
            instruction=prompt,
        )

        logger.info(
            f"VLM scene: {result.scene_description!r}, "
            f"{len(result.objects)} object(s) returned."
        )

        for detected in result.objects:
            confidence = (
                f"{detected.confidence:.2f}"
                if detected.confidence is not None
                else "n/a"
            )

            logger.info(
                f"  - {detected.label!r} confidence={confidence} "
                f"bbox={detected.bbox}"
            )

        return result

    def ground(self, frame, bbox: BoundingBox):
        return self._executor._rig.ground(frame, bbox)

    def rotate(self, delta_deg: float) -> None:
        angle = math.radians(delta_deg)

        self._executor._spin_by(angle)

        self._moved = True
        self._commanded_rad += angle

    def return_to_start(self) -> None:
        if not self._moved:
            return

        delta: float | None = None

        if self._start_yaw is not None:
            try:
                _x, _y, yaw = self._executor._rig.robot_pose()
                delta = normalize_angle(self._start_yaw - yaw)
            except Exception:
                delta = None

        if delta is None:
            delta = normalize_angle(-self._commanded_rad)

        # Ignore errors smaller than ~3 degrees.
        if abs(delta) > math.radians(3.0):
            self._executor.get_logger().info(
                "Returning to the starting heading "
                f"({math.degrees(delta):+.0f} deg)."
            )

            self._executor._spin_by(delta)


class ROS2SkillExecutor(Node):
    """ROS 2 adapter that executes planned robot skills.

    The executor is deliberately separated from the language/LLM layer.

    Supported skill flows:

        Named location:

            resolve_target
                ↓
            navigate_to_target
                ↓
            wait_for_navigation_result

        Physical object:

            find_object              (VLM + depth, rotating if needed)
                ↓
            navigate_to_target
                ↓
            wait_for_navigation_result

    Also implemented: rotate, stop, wait.
    Not implemented yet: move.

    The node spins itself on a background thread, so skills may simply
    block while ROS callbacks (camera, TF, Nav2 feedback) keep running.
    Callers must NOT spin this node themselves.
    """

    def __init__(
        self,
        target_resolver: TargetResolver,
        config: ExecutorConfig | None = None,
        *,
        perception: PerceptionBackend | None = None,
    ) -> None:
        super().__init__("robot_skill_executor")

        self.target_resolver = target_resolver
        self.config = config or ExecutorConfig()
        self._perception = perception

        # IMPORTANT:
        #
        # rclpy.node.Node already defines a read-only property named
        # ``context``. Therefore our application-level execution context
        # must use a different name.
        self._execution_context = ExecutionContext()

        # ------------------------------------------------------------------
        # Nav2 action clients
        # ------------------------------------------------------------------

        self._navigate_client = ActionClient(
            self,
            NavigateToPose,
            self.config.action_name,
        )

        self._spin_client = ActionClient(
            self,
            Spin,
            self.config.spin_action_name,
        )

        self._active_goal_handle = None
        self._active_target_name: Optional[str] = None

        # Used to throttle high-frequency Nav2 feedback logging.
        self._last_logged_distance: Optional[float] = None

        # ------------------------------------------------------------------
        # TF + RGB-D rig (only needed for object navigation)
        # ------------------------------------------------------------------

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(
            self._tf_buffer,
            self,
            spin_thread=False,
        )

        self._rig = None

        if perception is not None:
            # Imported lazily so named-location navigation still works on
            # a machine without cv_bridge / OpenCV.
            from perception.ros2.rgbd_rig import RGBDRig, RGBDRigConfig

            rig_config = self.config.rig_config or RGBDRigConfig(
                map_frame=self.config.map_frame,
                base_frame=self.config.base_frame,
            )

            self._rig = RGBDRig(
                self,
                self._tf_buffer,
                rig_config,
            )

        # ------------------------------------------------------------------
        # Background spinning
        # ------------------------------------------------------------------

        self._spin_executor = MultiThreadedExecutor(num_threads=4)
        self._spin_executor.add_node(self)

        self._spin_thread = threading.Thread(
            target=self._spin_loop,
            name="robot_skill_executor_spin",
            daemon=True,
        )
        self._spin_thread.start()

    # ------------------------------------------------------------------
    # Spinning helpers
    # ------------------------------------------------------------------

    def _spin_loop(self) -> None:
        try:
            self._spin_executor.spin()
        except Exception as exc:
            if rclpy.ok():
                self.get_logger().error(
                    f"Executor spin thread stopped: {exc}"
                )

    @staticmethod
    def _wait_for_future(future, timeout_sec: float) -> bool:
        """
        Wait for a future that is completed by the background spin thread.

        Returns True if the future finished, False on timeout.
        """

        deadline = time.monotonic() + timeout_sec

        while not future.done():
            if time.monotonic() > deadline or not rclpy.ok():
                return False

            time.sleep(0.02)

        return True

    # ------------------------------------------------------------------
    # Public execution interface
    # ------------------------------------------------------------------

    def execute(self, skills: list[Skill]) -> ExecutionResult:
        """Execute skills sequentially.

        Returns:
            ExecutionResult describing success or failure.

        Execution stops immediately if any skill fails. If an object
        cannot be found the result has error_code OBJECT_NOT_FOUND and
        no navigation goal has been sent.
        """

        total_skills = len(skills)

        if total_skills == 0:
            return ExecutionResult.failed(
                error_code="EMPTY_PLAN",
                error_message="Cannot execute an empty skill list.",
                metadata={
                    "skill_count": 0,
                },
            )

        # Fresh state per command so nothing stale leaks between runs.
        self._execution_context = ExecutionContext()

        current_index = 0
        current_skill: Skill | None = None

        try:
            for index, skill in enumerate(skills, start=1):
                current_index = index
                current_skill = skill

                self.get_logger().info(
                    f"Executing skill {index}/{total_skills}: "
                    f"{skill.name} {skill.parameters}"
                )

                self._execute_skill(skill)

            return ExecutionResult.succeeded(
                message="All skills executed successfully.",
                metadata={
                    "skill_count": total_skills,
                    "skills": [
                        skill.name
                        for skill in skills
                    ],
                    "object_searches": list(
                        self._execution_context.object_searches
                    ),
                },
            )

        except ObjectNotFoundError as exc:
            self.get_logger().warn(
                f"Object not found: {exc}"
            )

            return ExecutionResult.failed(
                error_code="OBJECT_NOT_FOUND",
                error_message=str(exc),
                metadata={
                    "skill_count": total_skills,
                    "completed_until_failure": max(
                        current_index - 1,
                        0,
                    ),
                    "failed_skill": (
                        current_skill.name
                        if current_skill is not None
                        else None
                    ),
                    "object_searches": list(
                        self._execution_context.object_searches
                    ),
                },
            )

        except Exception as exc:
            failed_skill = (
                current_skill.name
                if current_skill is not None
                else None
            )

            self.get_logger().error(
                f"Skill execution failed: {exc}"
            )

            return ExecutionResult.failed(
                error_code="SKILL_EXECUTION_FAILED",
                error_message=str(exc),
                metadata={
                    "skill_count": total_skills,
                    "completed_until_failure": max(
                        current_index - 1,
                        0,
                    ),
                    "failed_skill": failed_skill,
                },
            )

    # ------------------------------------------------------------------
    # Skill dispatch
    # ------------------------------------------------------------------

    def _execute_skill(self, skill: Skill) -> None:
        """Dispatch a planned skill to its implementation."""

        handlers = {
            "resolve_target": self._resolve_target,
            "find_object": self._find_object,
            "navigate_to_target": self._navigate_to_target,
            "wait_for_navigation_result": (
                self._wait_for_navigation_result
            ),
            "rotate": self._rotate_skill,
            "stop": self._stop_skill,
            "wait": self._wait_skill,
            "move": self._unsupported_skill,
        }

        handler = handlers.get(skill.name)

        if handler is None:
            raise ExecutorError(
                f"Unknown skill: {skill.name}"
            )

        handler(skill)

    # ------------------------------------------------------------------
    # Target resolution
    # ------------------------------------------------------------------

    def _resolve_target(self, skill: Skill) -> None:
        """Resolve a logical target into a concrete 2D pose."""

        try:
            target_type = skill.parameters["target_type"]
            target_value = skill.parameters["target_value"]
        except KeyError as exc:
            raise ExecutorError(
                "resolve_target skill is missing required "
                f"parameter: {exc}"
            ) from exc

        try:
            pose = self.target_resolver.resolve(
                target_type,
                target_value,
            )
        except Exception as exc:
            raise ExecutorError(
                f"Failed to resolve target "
                f"'{target_value}': {exc}"
            ) from exc

        self._execution_context.set_target(
            ExecutionContext.target_key(target_type, target_value),
            pose,
        )

        self.get_logger().info(
            f"Resolved '{target_value}' -> "
            f"frame={pose.frame_id}, "
            f"x={pose.x:.3f}, "
            f"y={pose.y:.3f}, "
            f"yaw={pose.yaw:.3f} rad"
        )

    # ------------------------------------------------------------------
    # Object search
    # ------------------------------------------------------------------

    def _find_object(self, skill: Skill) -> None:
        """
        Locate a described object with the VLM and depth.

        If the object is not in view, or is detected with too little
        confidence, the robot rotates in place and looks again. If a
        full sweep fails, the robot returns to its starting heading and
        ObjectNotFoundError is raised so that NO navigation goal is sent.
        """

        try:
            target_type = skill.parameters.get("target_type", "object")
            target_value = skill.parameters["target_value"]
        except KeyError as exc:
            raise ExecutorError(
                "find_object skill is missing required "
                f"parameter: {exc}"
            ) from exc

        if self._perception is None or self._rig is None:
            raise ExecutorError(
                "Object navigation needs a perception backend, but "
                "this executor was created without one."
            )

        logger = self.get_logger()

        logger.info(
            f"Searching for '{target_value}' "
            f"(min confidence "
            f"{self.config.object_search.min_confidence:.2f}, "
            f"{self.config.object_search.view_count} view(s) of "
            f"{self.config.object_search.rotation_step_deg:.0f} deg)."
        )

        searcher = ObjectSearcher(
            _SearchEnvironment(self),
            self.config.object_search,
            log=logger.info,
        )

        result = searcher.search(target_value)

        summary = result.to_summary()

        if not result.found:
            self._execution_context.record_object_search(summary)

            raise ObjectNotFoundError(result)

        grounding = result.grounding

        try:
            robot_x, robot_y, _robot_yaw = self._rig.robot_pose()

            goal_x, goal_y, goal_yaw, within_standoff = (
                compute_standoff_pose(
                    (robot_x, robot_y),
                    (grounding.map_x, grounding.map_y),
                    self.config.standoff_distance_m,
                )
            )
        except Exception as exc:
            raise ExecutorError(
                f"Found '{target_value}' but could not compute a "
                f"navigation goal: {exc}"
            ) from exc

        pose = Pose2D(
            frame_id=self.config.map_frame,
            x=goal_x,
            y=goal_y,
            yaw=goal_yaw,
        )

        self._execution_context.set_target(
            ExecutionContext.target_key(target_type, target_value),
            pose,
        )

        summary["object_position_map"] = [
            grounding.map_x,
            grounding.map_y,
        ]
        summary["depth_m"] = grounding.depth_m
        summary["point_frame"] = grounding.point_frame
        summary["goal_pose"] = {
            "x": goal_x,
            "y": goal_y,
            "yaw": goal_yaw,
            "already_within_standoff": within_standoff,
        }

        self._execution_context.record_object_search(summary)

        logger.info(result.message())

        logger.info(
            f"Object '{target_value}' at map "
            f"({grounding.map_x:.3f}, {grounding.map_y:.3f}), "
            f"depth {grounding.depth_m:.2f} m "
            f"(point frame '{grounding.point_frame}')."
        )

        logger.info(
            f"Navigation goal: x={goal_x:.3f}, y={goal_y:.3f}, "
            f"yaw={goal_yaw:.3f} rad"
            + (
                " (already within the standoff distance; "
                "only turning to face it)"
                if within_standoff
                else ""
            )
        )

    # ------------------------------------------------------------------
    # Rotation (Nav2 spin behavior)
    # ------------------------------------------------------------------

    def _spin_by(self, angle_rad: float) -> None:
        """
        Rotate in place by ``angle_rad`` (positive = counter-clockwise)
        using the Nav2 behavior server, so the motion goes through the
        velocity smoother and collision monitor.

        Raises:
            RotationError: the spin was rejected, failed or timed out.
        """

        if abs(angle_rad) < 1e-3:
            return

        config = self.config

        if not self._spin_client.wait_for_server(
            timeout_sec=config.server_timeout_sec
        ):
            raise RotationError(
                f"Nav2 spin action server '{config.spin_action_name}' "
                f"is unavailable after {config.server_timeout_sec:.1f} s."
            )

        allowance = max(
            config.spin_time_allowance_sec,
            abs(angle_rad) / 0.3 + 8.0,
        )

        goal = Spin.Goal()
        goal.target_yaw = float(angle_rad)
        goal.time_allowance = DurationMsg(
            sec=int(allowance),
            nanosec=0,
        )

        self.get_logger().info(
            f"Rotating {math.degrees(angle_rad):+.1f} deg..."
        )

        send_future = self._spin_client.send_goal_async(goal)

        if not self._wait_for_future(
            send_future,
            config.goal_send_timeout_sec,
        ):
            raise RotationError(
                "Timed out waiting for Nav2 to accept the spin goal."
            )

        goal_handle = send_future.result()

        if goal_handle is None or not goal_handle.accepted:
            raise RotationError("Nav2 rejected the spin goal.")

        result_future = goal_handle.get_result_async()

        if not self._wait_for_future(result_future, allowance + 10.0):
            try:
                goal_handle.cancel_goal_async()
            except Exception:
                pass

            raise RotationError("Timed out waiting for the spin to finish.")

        wrapped = result_future.result()

        if wrapped is None:
            raise RotationError("Nav2 returned no spin result.")

        error_code = getattr(wrapped.result, "error_code", 0)
        error_msg = getattr(wrapped.result, "error_msg", "")

        if wrapped.status != GoalStatus.STATUS_SUCCEEDED or error_code != 0:
            raise RotationError(
                f"Spin failed: action_status={wrapped.status}, "
                f"error_code={error_code}, error_msg={error_msg!r}"
            )

    def _rotate_skill(self, skill: Skill) -> None:
        """Rotate in place by the commanded angle."""

        try:
            direction = skill.parameters["direction"]
            angle_deg = float(skill.parameters["angle"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ExecutorError(
                f"rotate skill has invalid parameters: {exc}"
            ) from exc

        angle_rad = math.radians(angle_deg)

        if direction == "right":
            angle_rad = -angle_rad

        try:
            self._spin_by(angle_rad)
        except RotationError as exc:
            raise ExecutorError(str(exc)) from exc

    # ------------------------------------------------------------------
    # Stop / wait
    # ------------------------------------------------------------------

    def _stop_skill(self, _skill: Skill) -> None:
        """Cancel any active navigation goal."""

        if self._active_goal_handle is None:
            self.get_logger().info(
                "Stop requested; no active navigation goal."
            )
            return

        self._cancel_active_goal()

    def _wait_skill(self, skill: Skill) -> None:
        """Pause for the commanded duration."""

        try:
            duration = float(skill.parameters["duration"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ExecutorError(
                f"wait skill has invalid parameters: {exc}"
            ) from exc

        deadline = time.monotonic() + duration

        while time.monotonic() < deadline and rclpy.ok():
            time.sleep(min(0.1, max(deadline - time.monotonic(), 0.0)))

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _navigate_to_target(self, skill: Skill) -> None:
        """Send a resolved target to the Nav2 NavigateToPose action."""

        try:
            target_value = skill.parameters["target_value"]
        except KeyError as exc:
            raise ExecutorError(
                "navigate_to_target skill is missing "
                "'target_value'."
            ) from exc

        target_type = skill.parameters.get(
            "target_type",
            "named_location",
        )

        try:
            pose = self._execution_context.get_target(
                ExecutionContext.target_key(target_type, target_value)
            )
        except Exception as exc:
            raise ExecutorError(
                f"Target '{target_value}' has not been "
                f"successfully resolved: {exc}"
            ) from exc

        # --------------------------------------------------------------
        # Wait for Nav2
        # --------------------------------------------------------------

        self.get_logger().info(
            f"Waiting for Nav2 action server "
            f"'{self.config.action_name}'..."
        )

        if not self._navigate_client.wait_for_server(
            timeout_sec=self.config.server_timeout_sec
        ):
            raise ExecutorError(
                f"Nav2 action server "
                f"'{self.config.action_name}' is unavailable "
                f"after {self.config.server_timeout_sec:.1f} seconds."
            )

        self.get_logger().info(
            f"Nav2 action server "
            f"'{self.config.action_name}' is available."
        )

        # --------------------------------------------------------------
        # Construct NavigateToPose goal
        # --------------------------------------------------------------

        goal = NavigateToPose.Goal()

        goal.pose = self._pose2d_to_pose_stamped(
            pose
        )

        goal.pose.header.stamp = (
            self.get_clock().now().to_msg()
        )

        # Empty behavior_tree means Nav2 uses its configured default
        # behavior tree.
        goal.behavior_tree = ""

        self.get_logger().info(
            f"Sending navigation goal: "
            f"frame={pose.frame_id}, "
            f"x={pose.x:.3f}, "
            f"y={pose.y:.3f}, "
            f"yaw={pose.yaw:.3f} rad"
        )

        # Reset feedback state for this new navigation goal.
        self._last_logged_distance = None

        # --------------------------------------------------------------
        # Send goal
        # --------------------------------------------------------------

        send_future = (
            self._navigate_client.send_goal_async(
                goal,
                feedback_callback=self._feedback_callback,
            )
        )

        if not self._wait_for_future(
            send_future,
            self.config.goal_send_timeout_sec,
        ):
            raise ExecutorError(
                "Timed out waiting for Nav2 goal acceptance."
            )

        try:
            goal_handle = send_future.result()
        except Exception as exc:
            raise ExecutorError(
                f"Exception while sending Nav2 goal: {exc}"
            ) from exc

        if goal_handle is None:
            raise ExecutorError(
                "Nav2 returned no goal handle."
            )

        if not goal_handle.accepted:
            raise ExecutorError(
                f"Nav2 rejected navigation goal "
                f"for '{target_value}'."
            )

        self._active_goal_handle = goal_handle
        self._active_target_name = target_value

        self.get_logger().info(
            f"Nav2 accepted goal for '{target_value}'."
        )

    # ------------------------------------------------------------------
    # Navigation result
    # ------------------------------------------------------------------

    def _wait_for_navigation_result(
        self,
        _skill: Skill,
    ) -> None:
        """Wait for the currently active Nav2 goal to finish."""

        if self._active_goal_handle is None:
            raise ExecutorError(
                "No active navigation goal exists for "
                "wait_for_navigation_result."
            )

        target_name = (
            self._active_target_name
            or "<unknown>"
        )

        self.get_logger().info(
            f"Waiting for navigation result "
            f"for '{target_name}'..."
        )

        result_future = (
            self._active_goal_handle.get_result_async()
        )

        if not self._wait_for_future(
            result_future,
            self.config.result_timeout_sec,
        ):
            self._cancel_active_goal()

            raise ExecutorError(
                f"Timed out waiting for navigation result "
                f"for '{target_name}' after "
                f"{self.config.result_timeout_sec:.1f} seconds."
            )

        try:
            wrapped_result = result_future.result()
        except Exception as exc:
            self._active_goal_handle = None
            self._active_target_name = None

            raise ExecutorError(
                f"Exception while receiving Nav2 result "
                f"for '{target_name}': {exc}"
            ) from exc

        if wrapped_result is None:
            self._active_goal_handle = None
            self._active_target_name = None

            raise ExecutorError(
                f"Nav2 returned no navigation result "
                f"for '{target_name}'."
            )

        result = wrapped_result.result
        status = wrapped_result.status

        # Clear active-goal state before processing the result.
        self._active_goal_handle = None
        self._active_target_name = None

        # --------------------------------------------------------------
        # Check ROS action status
        # --------------------------------------------------------------

        if status != GoalStatus.STATUS_SUCCEEDED:
            raise ExecutorError(
                f"Navigation to '{target_name}' failed: "
                f"action_status={status}, "
                f"error_code={result.error_code}, "
                f"error_msg={result.error_msg!r}"
            )

        # --------------------------------------------------------------
        # Check Nav2 result
        # --------------------------------------------------------------

        if result.error_code != 0:
            raise ExecutorError(
                f"Navigation to '{target_name}' returned "
                f"Nav2 error {result.error_code}: "
                f"{result.error_msg}"
            )

        self.get_logger().info(
            f"Navigation to '{target_name}' succeeded."
        )

    # ------------------------------------------------------------------
    # Goal cancellation
    # ------------------------------------------------------------------

    def _cancel_active_goal(self) -> None:
        """Cancel the currently active Nav2 goal."""

        if self._active_goal_handle is None:
            return

        target_name = (
            self._active_target_name
            or "<unknown>"
        )

        self.get_logger().warn(
            f"Cancelling active navigation goal "
            f"for '{target_name}'."
        )

        try:
            cancel_future = (
                self._active_goal_handle.cancel_goal_async()
            )

            if not self._wait_for_future(
                cancel_future,
                self.config.cancel_timeout_sec,
            ):
                self.get_logger().error(
                    f"Timed out while cancelling goal "
                    f"for '{target_name}'."
                )

        except Exception as exc:
            self.get_logger().error(
                f"Exception while cancelling goal "
                f"for '{target_name}': {exc}"
            )

        finally:
            self._active_goal_handle = None
            self._active_target_name = None

    # ------------------------------------------------------------------
    # Nav2 feedback
    # ------------------------------------------------------------------

    def _feedback_callback(self, feedback_msg) -> None:
        """Process Nav2 feedback with throttled logging."""

        feedback = feedback_msg.feedback

        distance_remaining = getattr(
            feedback,
            "distance_remaining",
            None,
        )

        if distance_remaining is None:
            return

        previous_distance = self._last_logged_distance

        if (
            previous_distance is not None
            and abs(
                distance_remaining - previous_distance
            ) < self.config.feedback_log_delta_m
        ):
            return

        self._last_logged_distance = distance_remaining

        self.get_logger().info(
            f"Navigation feedback: "
            f"{distance_remaining:.3f} m remaining"
        )

    # ------------------------------------------------------------------
    # Unsupported skills
    # ------------------------------------------------------------------

    def _unsupported_skill(
        self,
        skill: Skill,
    ) -> None:
        """Reject skills that have no ROS implementation yet."""

        raise ExecutorError(
            f"Skill '{skill.name}' is not implemented "
            f"in the ROS 2 executor yet."
        )

    # ------------------------------------------------------------------
    # Pose conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _pose2d_to_pose_stamped(
        pose: Pose2D,
    ) -> PoseStamped:
        """Convert internal Pose2D to ROS PoseStamped."""

        qx, qy, qz, qw = yaw_to_quaternion(
            pose.yaw
        )

        msg = PoseStamped()

        msg.header.frame_id = pose.frame_id

        msg.pose.position.x = pose.x
        msg.pose.position.y = pose.y
        msg.pose.position.z = 0.0

        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw

        return msg

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Cancel an active goal, stop spinning and destroy the node."""

        if (
            self._active_goal_handle is not None
            and rclpy.ok()
        ):
            self._cancel_active_goal()

        try:
            self._spin_executor.shutdown(timeout_sec=2.0)
        except Exception:
            pass

        if self._spin_thread.is_alive():
            self._spin_thread.join(timeout=2.0)

        self.destroy_node()


# ======================================================================
# CLI
# ======================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Execute a validated robot command through Nav2."
    )

    parser.add_argument(
        "target",
        help=(
            "Named location from the YAML configuration "
            "(e.g. 'Station 1'), or, with --object, a description "
            "of an object (e.g. 'red box')."
        ),
    )

    parser.add_argument(
        "--object",
        action="store_true",
        help=(
            "Treat TARGET as a physical object: find it with the VLM "
            "(rotating in place if needed) and navigate to it."
        ),
    )

    parser.add_argument(
        "--config",
        default="config/locations.yaml",
        help="Path to the location YAML configuration.",
    )

    parser.add_argument(
        "--gateway-url",
        default="http://127.0.0.1:8000",
        help="URL of the AI gateway (used with --object).",
    )

    parser.add_argument(
        "--min-confidence",
        type=float,
        default=ObjectSearchConfig().min_confidence,
        help="Minimum VLM confidence to accept an object.",
    )

    args = parser.parse_args()

    rclpy.init()

    executor: ROS2SkillExecutor | None = None

    try:
        # Import here deliberately:
        # command_pipeline imports ROS2SkillExecutor, so importing it at
        # module scope would create a circular import.
        from command_pipeline import execute_command

        resolver = TargetResolver(
            Path(args.config)
        )

        perception = None

        if args.object:
            from ai_client.gateway_client import AIGatewayClient
            from ai_client.vlm_client import RemoteVLMClient
            from perception.remote_vlm import RemoteVLMPerception

            perception = RemoteVLMPerception(
                RemoteVLMClient(
                    AIGatewayClient(
                        base_url=args.gateway_url,
                        timeout=60.0,
                    )
                )
            )

        executor = ROS2SkillExecutor(
            resolver,
            ExecutorConfig(
                object_search=ObjectSearchConfig(
                    min_confidence=args.min_confidence,
                ),
            ),
            perception=perception,
        )

        command = {
            "action": "navigate",
            "target": {
                "type": "object" if args.object else "named_location",
                "value": args.target,
            },
        }

        execute_command(
            command,
            executor,
        )

        executor.get_logger().info(
            "Command pipeline completed successfully."
        )

        return 0

    except Exception as exc:
        if executor is not None:
            executor.get_logger().error(
                f"Command pipeline failed: {exc}"
            )
        else:
            print(
                f"Command pipeline failed: {exc}"
            )

        return 1

    finally:
        if executor is not None:
            executor.shutdown()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
