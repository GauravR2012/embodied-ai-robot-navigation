from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import gradio as gr
import rclpy

from ai_client.gateway_client import AIGatewayClient
from ai_client.llm_client import RemoteLLMClient
from ai_client.vlm_client import RemoteVLMClient
from command_pipeline import CommandPipelineError, execute_command_detailed
from llm_adapter import LLMCommandAdapter, RemoteLLMBackend
from perception.object_search import ObjectSearchConfig
from perception.remote_vlm import RemoteVLMPerception
from ros2_executor import ExecutorConfig, ROS2SkillExecutor
from target_resolver import TargetResolver


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TEST_LOCATIONS_FILE = Path("config/test_locations.yaml")

AI_GATEWAY_URL = "http://127.0.0.1:8000"

GRADIO_HOST = "127.0.0.1"
GRADIO_PORT = 7860

# Objects detected with less confidence than this are not trusted.
# The robot rotates and looks again; if nothing better is found it
# reports "not found" and does not move.
OBJECT_MIN_CONFIDENCE = 0.60

# Rotation between looks (degrees). The simulated camera FOV is 60 deg.
OBJECT_SEARCH_STEP_DEG = 45.0


# ---------------------------------------------------------------------------
# Status formatting
# ---------------------------------------------------------------------------

def _format_search(search: dict[str, Any]) -> str:
    """Render one object-search summary as readable text."""

    lines = [f"Object search: '{search.get('target')}'"]

    lines.append(f"  {search.get('message')}")

    if search.get("found"):
        position = search.get("object_position_map")
        goal = search.get("goal_pose")

        if position:
            lines.append(
                f"  Object position (map): "
                f"x={position[0]:.2f}, y={position[1]:.2f}"
            )

        if goal:
            lines.append(
                f"  Navigation goal: x={goal['x']:.2f}, "
                f"y={goal['y']:.2f}, yaw={goal['yaw']:.2f} rad"
            )

    views = search.get("views") or []

    if views:
        lines.append("  Scan log:")

        for view in views:
            lines.append(
                f"    view {view['index'] + 1} "
                f"({view['heading_offset_deg']:+.0f} deg): "
                f"{view['note']}"
            )

    return "\n".join(lines)


def _format_success(skill_names: list[str], metadata: dict | None) -> str:
    text = (
        "✅ Navigation completed successfully.\n\n"
        f"Executed skills: {skill_names}"
    )

    searches = (metadata or {}).get("object_searches") or []

    for search in searches:
        text += "\n\n" + _format_search(search)

    return text


def _format_failure(exc: CommandPipelineError) -> str:
    result = exc.result

    if result is not None and result.error_code == "OBJECT_NOT_FOUND":
        text = (
            "🔍 Object not found. The robot did not navigate.\n\n"
            f"{result.error_message}"
        )

        searches = (result.metadata or {}).get("object_searches") or []

        for search in searches:
            text += "\n\n" + _format_search(search)

        return text

    return f"❌ Robot command failed:\n{exc}"


# ---------------------------------------------------------------------------
# Robot controller
# ---------------------------------------------------------------------------


class RobotController:
    """Connects the Gradio UI to the existing robot command pipeline."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

        # --------------------------------------------------------------
        # Remote AI gateway
        # --------------------------------------------------------------

        gateway = AIGatewayClient(
            base_url=AI_GATEWAY_URL,
            timeout=60.0,
        )

        remote_llm = RemoteLLMClient(
            gateway
        )

        backend = RemoteLLMBackend(
            remote_llm
        )

        self.llm = LLMCommandAdapter(
            backend
        )

        # VLM used to find objects ("go to the red box").
        perception = RemoteVLMPerception(
            RemoteVLMClient(
                gateway
            )
        )

        # --------------------------------------------------------------
        # ROS 2
        # --------------------------------------------------------------

        if not rclpy.ok():
            rclpy.init()

        self.resolver = TargetResolver(
            TEST_LOCATIONS_FILE
        )

        # The executor spins itself on a background thread, so the UI
        # must NOT call rclpy.spin() on it.
        self.executor = ROS2SkillExecutor(
            self.resolver,
            ExecutorConfig(
                object_search=ObjectSearchConfig(
                    min_confidence=OBJECT_MIN_CONFIDENCE,
                    rotation_step_deg=OBJECT_SEARCH_STEP_DEG,
                ),
            ),
            perception=perception,
        )

    # ------------------------------------------------------------------
    # Command execution
    # ------------------------------------------------------------------

    def execute(
        self,
        user_text: str,
    ) -> tuple[str, str]:
        """
        Convert natural language into a validated robot command
        and execute it through the existing command pipeline.

        Returns:
            structured command JSON,
            execution status.
        """

        user_text = user_text.strip()

        if not user_text:
            return (
                "",
                "❌ Please enter a robot command.",
            )

        # Serialize robot commands so that two UI requests cannot
        # simultaneously control the same executor.
        with self._lock:
            command_json = ""

            try:
                # ------------------------------------------------------
                # Natural language → validated RobotCommand
                # ------------------------------------------------------

                command = self.llm.interpret_to_dict(
                    user_text
                )

                command_json = json.dumps(
                    command,
                    indent=2,
                )

                # ------------------------------------------------------
                # Command → planner → ROS 2 → (VLM) → Nav2
                # ------------------------------------------------------

                skills, result = execute_command_detailed(
                    command,
                    self.executor,
                )

                skill_names = [
                    skill.name
                    for skill in skills
                ]

                return (
                    command_json,
                    _format_success(
                        skill_names,
                        result.metadata,
                    ),
                )

            except CommandPipelineError as exc:
                return (
                    command_json,
                    _format_failure(exc),
                )

            except Exception as exc:
                return (
                    command_json,
                    f"❌ Unexpected error:\n{exc}",
                )

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Shutdown the ROS 2 executor."""

        if self.executor is not None:
            self.executor.shutdown()

        if rclpy.ok():
            rclpy.shutdown()


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------


def build_interface(
    controller: RobotController,
) -> gr.Blocks:
    """Construct the Gradio application."""

    with gr.Blocks(
        title="Embodied AI Robot Navigation"
    ) as app:

        gr.Markdown(
            """
# Embodied AI Robot Navigation

Control the simulated robot using natural-language commands.

**Pipeline**

`Natural Language → Qwen LLM → Structured Command → Task Planner → (Qwen-VL object search, rotating if needed) → ROS 2 → Nav2`

Examples: `Go to Executor Test` · `Go to the red box` · `Go to the blue box`

If the object cannot be found with enough confidence, the robot does not move.
"""
        )

        # --------------------------------------------------------------
        # Command input
        # --------------------------------------------------------------

        command_input = gr.Textbox(
            label="Robot Command",
            placeholder=(
                "Example: Go to the red box"
            ),
            lines=2,
        )

        execute_button = gr.Button(
            "Execute Command",
            variant="primary",
        )

        # --------------------------------------------------------------
        # Structured command output
        # --------------------------------------------------------------

        command_output = gr.Code(
            label="Structured Robot Command",
            language="json",
            interactive=False,
        )

        # --------------------------------------------------------------
        # Execution status
        # --------------------------------------------------------------

        status_output = gr.Textbox(
            label="Execution Status",
            lines=12,
            interactive=False,
        )

        # --------------------------------------------------------------
        # Button handler
        # --------------------------------------------------------------

        execute_button.click(
            fn=controller.execute,
            inputs=command_input,
            outputs=[
                command_output,
                status_output,
            ],
        )

        # Allow Enter/Ctrl+Enter-style submission through the textbox.
        command_input.submit(
            fn=controller.execute,
            inputs=command_input,
            outputs=[
                command_output,
                status_output,
            ],
        )

    return app


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    controller: RobotController | None = None

    try:
        controller = RobotController()

        app = build_interface(
            controller
        )

        print(
            f"Starting Gradio at "
            f"http://{GRADIO_HOST}:{GRADIO_PORT}"
        )

        app.launch(
            server_name=GRADIO_HOST,
            server_port=GRADIO_PORT,
            show_error=True,
        )

        return 0

    except KeyboardInterrupt:
        print(
            "\nStopping Gradio robot interface..."
        )

        return 0

    except Exception as exc:
        print(
            f"Failed to start Gradio robot interface: {exc}"
        )

        return 1

    finally:
        if controller is not None:
            controller.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
