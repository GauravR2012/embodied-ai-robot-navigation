from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from .gateway_client import AIGatewayClient


class RemoteVLMClient:
    """
    High-level client for remote VLM inference.
    """

    def __init__(
        self,
        gateway: AIGatewayClient,
    ) -> None:
        self.gateway = gateway

    @staticmethod
    def encode_image(image_path: str | Path) -> str:
        """Read an image from disk and encode it as base64."""

        path = Path(image_path)

        if not path.is_file():
            raise FileNotFoundError(
                f"Image file does not exist: {path}"
            )

        return base64.b64encode(
            path.read_bytes()
        ).decode("utf-8")

    def analyze(
        self,
        image_path: str | Path,
        instruction: str,
    ) -> str:
        """
        Send an image to the remote VLM.
        """

        image_base64 = self.encode_image(image_path)

        result = self.gateway.perception(
            image_base64=image_base64,
            instruction=instruction,
        )

        response = result.get("response")

        if not isinstance(response, str):
            raise RuntimeError(
                "Remote VLM response does not contain a valid "
                "'response' string."
            )

        return response

    def health(self) -> dict[str, Any]:
        """Return remote gateway health information."""

        return self.gateway.health()
