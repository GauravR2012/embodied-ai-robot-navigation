from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node

from execution_context import ExecutionContext
from execution_result import ExecutionResult
from task_planner import Skill
from target_resolver import Pose2D, TargetResolver, yaw_to_quaternion


class ExecutorError(RuntimeError):
    """Raised when a robot skill cannot be executed successfully."""


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


class ROS2SkillExecutor(Node):
    """ROS 2 adapter that executes planned robot skills.

    The executor is deliberately separated from the language/LLM layer.

    Current supported ROS skill flow:

        resolve_target
            ↓
        navigate_to_target
            ↓
        wait_for_navigation_result
            ↓
        Nav2 /navigate_to_pose

    The following skills are intentionally not implemented yet:

        move
        rotate
        stop
        wait

    Those will be added through dedicated ROS 2 interfaces rather than
    allowing the LLM to directly publish arbitrary ROS messages.
    """

    def __init__(
        self,
        target_resolver: TargetResolver,
        config: ExecutorConfig | None = None,
    ) -> None:
        super().__init__("robot_skill_executor")

        self.target_resolver = target_resolver

        # IMPORTANT:
        #
        # rclpy.node.Node already defines a read-only property named
        # ``context``. Therefore our application-level execution context
        # must use a different name.
        self._execution_context = ExecutionContext()

        self.config = config or ExecutorConfig()

        self._navigate_client = ActionClient(
            self,
            NavigateToPose,
            self.config.action_name,
        )

        self._active_goal_handle = None
        self._active_target_name: Optional[str] = None

        # Used to throttle high-frequency Nav2 feedback logging.
        self._last_logged_distance: Optional[float] = None

    # ------------------------------------------------------------------
    # Public execution interface
    # ------------------------------------------------------------------

    def execute(self, skills: list[Skill]) -> ExecutionResult:
        """Execute skills sequentially.

        Returns:
            ExecutionResult describing success or failure.

        Execution stops immediately if any skill fails.
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
            "navigate_to_target": self._navigate_to_target,
            "wait_for_navigation_result": (
                self._wait_for_navigation_result
            ),
            "move": self._unsupported_skill,
            "rotate": self._unsupported_skill,
            "stop": self._unsupported_skill,
            "wait": self._unsupported_skill,
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
            target_value,
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

        try:
            pose = self._execution_context.get_target(
                target_value
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

        rclpy.spin_until_future_complete(
            self,
            send_future,
            timeout_sec=self.config.goal_send_timeout_sec,
        )

        if not send_future.done():
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

        rclpy.spin_until_future_complete(
            self,
            result_future,
            timeout_sec=self.config.result_timeout_sec,
        )

        if not result_future.done():
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

            rclpy.spin_until_future_complete(
                self,
                cancel_future,
                timeout_sec=self.config.cancel_timeout_sec,
            )

            if not cancel_future.done():
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
        """Cancel an active goal and destroy the ROS node."""

        if (
            self._active_goal_handle is not None
            and rclpy.ok()
        ):
            self._cancel_active_goal()

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
            "Named location from config/locations.yaml, "
            "e.g. 'Station 1'."
        ),
    )

    parser.add_argument(
        "--config",
        default="config/locations.yaml",
        help="Path to the location YAML configuration.",
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

        executor = ROS2SkillExecutor(
            resolver
        )

        command = {
            "action": "navigate",
            "target": {
                "type": "named_location",
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