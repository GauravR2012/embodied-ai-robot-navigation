from __future__ import annotations

from typing import Any

import requests


class AIGatewayError(RuntimeError):
    """Raised when the remote AI gateway cannot complete a request."""


class AIGatewayClient:
    """
    HTTP client for the remote AI inference gateway.

    The gateway runs on the remote GPU machine.
    ROS 2 remains entirely local to the robot/laptop.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict[str, Any]:
        """Return gateway health information."""

        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise AIGatewayError(
                f"AI gateway health check failed: {exc}"
            ) from exc

        try:
            return response.json()
        except ValueError as exc:
            raise AIGatewayError(
                "AI gateway returned invalid JSON from /health."
            ) from exc

    def chat(
        self,
        instruction: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send a language request to the remote LLM."""

        payload = {
            "instruction": instruction,
            "context": context or {},
        }

        try:
            response = requests.post(
                f"{self.base_url}/v1/chat",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise AIGatewayError(
                f"LLM request failed: {exc}"
            ) from exc

        try:
            return response.json()
        except ValueError as exc:
            raise AIGatewayError(
                "AI gateway returned invalid JSON from /v1/chat."
            ) from exc

    def perception(
        self,
        image_base64: str,
        instruction: str,
    ) -> dict[str, Any]:
        """Send an image and perception instruction to the remote VLM."""

        payload = {
            "image_base64": image_base64,
            "instruction": instruction,
        }

        try:
            response = requests.post(
                f"{self.base_url}/v1/perception",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise AIGatewayError(
                f"VLM request failed: {exc}"
            ) from exc

        try:
            return response.json()
        except ValueError as exc:
            raise AIGatewayError(
                "AI gateway returned invalid JSON from /v1/perception."
            ) from exc
