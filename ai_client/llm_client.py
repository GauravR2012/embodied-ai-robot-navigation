from __future__ import annotations

from typing import Any

from .gateway_client import AIGatewayClient


class RemoteLLMClient:
    """
    High-level client for remote LLM inference.

    This class intentionally knows nothing about ROS 2.
    """

    def __init__(
        self,
        gateway: AIGatewayClient,
    ) -> None:
        self.gateway = gateway

    def generate(
        self,
        instruction: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        """
        Generate a response from the remote LLM.
        """

        result = self.gateway.chat(
            instruction=instruction,
            context=context,
        )

        response = result.get("response")

        if not isinstance(response, str):
            raise RuntimeError(
                "Remote LLM response does not contain a valid "
                "'response' string."
            )

        return response

    def health(self) -> dict[str, Any]:
        """Return remote gateway health information."""

        return self.gateway.health()
