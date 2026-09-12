from __future__ import annotations

import json
from pathlib import Path

from ai_client.vlm_client import RemoteVLMClient

from .interface import PerceptionBackend
from .models import DetectedObject, PerceptionResult


class RemoteVLMPerception(PerceptionBackend):
    """
    Perception backend backed by the remote VLM.

    The VLM is responsible for visual interpretation.
    This class converts its response into the common
    PerceptionResult representation.
    """

    def __init__(self, client: RemoteVLMClient) -> None:
        self.client = client

    def analyze(
        self,
        image_path: str | Path,
        instruction: str,
    ) -> PerceptionResult:

        response = self.client.analyze(
            image_path=image_path,
            instruction=instruction,
        )

        return self._parse_response(response)

    @staticmethod
    def _parse_response(response: str) -> PerceptionResult:
        """
        Parse a structured VLM response.

        The remote VLM should eventually be constrained to
        return the schema represented here.
        """

        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            return PerceptionResult(
                scene_description=response,
                raw_response=response,
            )

        objects = []

        for item in data.get("objects", []):
            if not isinstance(item, dict):
                continue

            objects.append(
                DetectedObject(
                    label=str(item.get("label", "")),
                    confidence=item.get("confidence"),
                    attributes=item.get("attributes", {}),
                )
            )

        return PerceptionResult(
            objects=tuple(objects),
            scene_description=data.get("scene_description"),
            raw_response=response,
        )
