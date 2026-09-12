from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

from command_schema import RobotCommand, command_to_dict, parse_command


class LLMAdapterError(RuntimeError):
    """Raised when the LLM cannot produce a valid robot command."""


class LLMBackend(ABC):
    """Backend interface for natural-language-to-command inference."""

    @abstractmethod
    def generate(self, user_text: str) -> str:
        """Generate a raw model response."""
        raise NotImplementedError


class MockLLMBackend(LLMBackend):
    """Deterministic backend used for development and testing."""

    def generate(self, user_text: str) -> str:
        text = user_text.strip().lower()

        if text == "go to executor test":
            return json.dumps(
                {
                    "action": "navigate",
                    "target": {
                        "type": "named_location",
                        "value": "Executor Test",
                    },
                }
            )

        if text == "stop":
            return json.dumps(
                {
                    "action": "stop",
                }
            )

        if text == "wait five seconds":
            return json.dumps(
                {
                    "action": "wait",
                    "duration": 5,
                    "unit": "s",
                }
            )

        raise LLMAdapterError(
            f"Mock backend has no response for: {user_text!r}"
        )


class OllamaLLMBackend(LLMBackend):
    """
    Ollama-backed natural-language command generator.

    The model is constrained using the RobotCommand JSON Schema.
    The backend itself has no ROS access.
    """

    def __init__(
        self,
        model: str = "llama3.2:3b",
        host: str | None = None,
        temperature: float = 0.0,
    ):
        self.model = model
        self.temperature = temperature

        try:
            from ollama import Client
        except ImportError as exc:
            raise LLMAdapterError(
                "Ollama Python client is not installed. "
                "Run: python -m pip install ollama"
            ) from exc

        self._client = Client(host=host) if host else Client()

    @staticmethod
    def _system_prompt() -> str:
        schema = json.dumps(
            RobotCommandSchema.schema(),
            indent=2,
        )

        return f"""
You are the command interpreter for a mobile robot.

Your ONLY job is to convert the user's natural-language instruction
into one valid robot command.

You MUST return JSON matching the supplied schema.

Never return:
- Python
- ROS commands
- shell commands
- coordinates
- explanations
- markdown
- prose

You do not control the robot directly.

Named locations are symbolic names. Never invent coordinates.

Supported command concepts include:
- navigate
- move
- rotate
- stop
- wait
- sequence

If the user's request cannot be represented by the schema,
return a command that is valid according to the schema only if
the request has an unambiguous safe interpretation. Otherwise,
do not invent unsupported robot behavior.

JSON schema:

{schema}
""".strip()

    def generate(self, user_text: str) -> str:
        if not user_text.strip():
            raise LLMAdapterError(
                "User instruction cannot be empty."
            )

        try:
            response = self._client.chat(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._system_prompt(),
                    },
                    {
                        "role": "user",
                        "content": user_text,
                    },
                ],
                format=RobotCommandSchema.schema(),
                options={
                    "temperature": self.temperature,
                },
                stream=False,
            )
        except Exception as exc:
            raise LLMAdapterError(
                f"Ollama request failed: {exc}"
            ) from exc

        try:
            content = response.message.content
        except AttributeError as exc:
            raise LLMAdapterError(
                "Ollama returned an unexpected response structure."
            ) from exc

        if not isinstance(content, str) or not content.strip():
            raise LLMAdapterError(
                "Ollama returned an empty response."
            )

        return content


class RemoteLLMBackend(LLMBackend):
    """
    Remote LLM backend.

    The backend communicates with the AI gateway through
    RemoteLLMClient. It does not know anything about ROS 2,
    Nav2, GPUs, or the physical robot.
    """

    def __init__(self, client: Any):
        self.client = client

    def generate(self, user_text: str) -> str:
        if not isinstance(user_text, str):
            raise LLMAdapterError(
                "User instruction must be a string."
            )

        if not user_text.strip():
            raise LLMAdapterError(
                "User instruction cannot be empty."
            )

        try:
            response = self.client.generate(
                instruction=user_text,
            )
        except Exception as exc:
            raise LLMAdapterError(
                f"Remote LLM request failed: {exc}"
            ) from exc

        if not isinstance(response, str) or not response.strip():
            raise LLMAdapterError(
                "Remote LLM returned an empty response."
            )

        return response


class LLMCommandAdapter:
    """
    Converts natural-language instructions into validated RobotCommands.

    Pipeline:

        user text
            ↓
        LLM backend
            ↓
        JSON
            ↓
        Pydantic validation
            ↓
        RobotCommand
    """

    def __init__(self, backend: LLMBackend):
        self.backend = backend

    def interpret(self, user_text: str) -> RobotCommand:
        if not isinstance(user_text, str):
            raise LLMAdapterError(
                "User instruction must be a string."
            )

        user_text = user_text.strip()

        if not user_text:
            raise LLMAdapterError(
                "User instruction cannot be empty."
            )

        try:
            raw_response = self.backend.generate(user_text)
        except LLMAdapterError:
            raise
        except Exception as exc:
            raise LLMAdapterError(
                f"LLM backend failed: {exc}"
            ) from exc

        if not isinstance(raw_response, str):
            raise LLMAdapterError(
                "LLM backend must return a string response."
            )

        try:
            command_data: Any = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise LLMAdapterError(
                f"LLM returned invalid JSON: {exc}"
            ) from exc

        if not isinstance(command_data, dict):
            raise LLMAdapterError(
                "LLM output must be a JSON object."
            )

        try:
            return parse_command(command_data)
        except Exception as exc:
            raise LLMAdapterError(
                f"LLM produced an invalid robot command: {exc}"
            ) from exc

    def interpret_to_dict(
        self,
        user_text: str,
    ) -> dict[str, Any]:
        command = self.interpret(user_text)
        return command_to_dict(command)


# ---------------------------------------------------------------------------
# Schema adapter
# ---------------------------------------------------------------------------
#
# Pydantic's discriminated union is the source of truth in command_schema.py.
# We expose its JSON Schema here for Ollama's structured-output interface.
#


def _build_robot_command_schema() -> dict[str, Any]:
    from pydantic import TypeAdapter

    return TypeAdapter(RobotCommand).json_schema()


RobotCommandSchema = type(
    "RobotCommandSchema",
    (),
    {
        "schema": staticmethod(_build_robot_command_schema),
    },
)