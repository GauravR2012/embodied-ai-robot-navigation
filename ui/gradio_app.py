from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import gradio as gr
import rclpy

from ai_client.gateway_client import AIGatewayClient
from ai_client.llm_client import RemoteLLMClient
from command_pipeline import CommandPipelineError, execute_command
from llm_adapter import LLMCommandAdapter, RemoteLLMBackend
from ros2_executor import ROS2SkillExecutor
from target_resolver import TargetResolver


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TEST_LOCATIONS_FILE = Path("config/test_locations.yaml")

AI_GATEWAY_URL = "http://127.0.0.1:8000"

GRADIO_HOST = "127.0.0.1"
GRADIO_PORT = 7860


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

        # --------------------------------------------------------------
        # ROS 2
        # --------------------------------------------------------------

        if not rclpy.ok():
            rclpy.init()

        self.resolver = TargetResolver(
            TEST_LOCATIONS_FILE
        )

        self.executor = ROS2SkillExecutor(
            self.resolver
        )

        # --------------------------------------------------------------
        # ROS 2 spinning
        # --------------------------------------------------------------

        self._ros_thread = threading.Thread(
            target=self._spin_ros,
            daemon=True,
        )

        self._ros_thread.start()

    # ------------------------------------------------------------------
    # ROS 2 spin
    # ------------------------------------------------------------------

    def _spin_ros(self) -> None:
        """Keep the ROS 2 executor node alive."""

        try:
            rclpy.spin(
                self.executor
            )
        except Exception as exc:
            print(
                f"ROS 2 spin thread stopped: {exc}"
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
                # Command → planner → ROS 2 → Nav2
                # ------------------------------------------------------

                skills = execute_command(
                    command,
                    self.executor,
                )

                skill_names = [
                    skill.name
                    for skill in skills
                ]

                status = (
                    "✅ Navigation completed successfully.\n\n"
                    f"Executed skills: {skill_names}"
                )

                return (
                    command_json,
                    status,
                )

            except CommandPipelineError as exc:
                return (
                    command_json
                    if "command_json" in locals()
                    else "",
                    f"❌ Robot command failed:\n{exc}",
                )

            except Exception as exc:
                return (
                    command_json
                    if "command_json" in locals()
                    else "",
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

`Natural Language → Qwen LLM → Structured Command → Task Planner → ROS 2 → Nav2`
"""
        )

        # --------------------------------------------------------------
        # Command input
        # --------------------------------------------------------------

        command_input = gr.Textbox(
            label="Robot Command",
            placeholder=(
                "Example: Go to Executor Test"
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
            lines=4,
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
