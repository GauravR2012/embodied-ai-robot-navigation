from __future__ import annotations

import json
from pathlib import Path

from ai_client.vlm_client import RemoteVLMClient

from .interface import PerceptionBackend
from .models import (
    BoundingBox,
    DetectedObject,
    PerceptionResult,
)


class RemoteVLMPerception(PerceptionBackend):
    """
    Perception backend backed by the remote VLM.

    The VLM is responsible for visual interpretation.
    This class converts the VLM response into the common,
    backend-independent PerceptionResult representation.

    The VLM response is treated as untrusted external data.
    Parsing and validation therefore happen deterministically
    at this boundary.
    """

    def __init__(self, client: RemoteVLMClient) -> None:
        self.client = client

    def analyze(
        self,
        image_path: str | Path,
        instruction: str,
    ) -> PerceptionResult:
        """
        Analyze an image using the remote VLM.

        Args:
            image_path: Path to the image to analyze.
            instruction: Perception instruction supplied to the VLM.

        Returns:
            A validated backend-independent PerceptionResult.
        """

        response = self.client.analyze(
            image_path=image_path,
            instruction=instruction,
        )

        return self._parse_response(response)

    @staticmethod
    def _parse_response(response: str) -> PerceptionResult:
        """
        Parse the structured VLM perception response.

        Expected top-level structure:

        {
            "scene_description": "string",
            "objects": [
                {
                    "label": "string",
                    "confidence": number,
                    "bbox": {
                        "x_min": number,
                        "y_min": number,
                        "x_max": number,
                        "y_max": number
                    },
                    "attributes": {}
                }
            ]
        }

        The VLM output is untrusted. Invalid JSON falls back to
        a plain-text scene description. Invalid individual objects
        are ignored or have invalid optional fields removed.
        """

        if not isinstance(response, str):
            response = str(response)

        response = response.strip()

        if not response:
            return PerceptionResult(
                scene_description=None,
                raw_response=response,
            )

        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            return PerceptionResult(
                scene_description=response,
                raw_response=response,
            )

        if not isinstance(data, dict):
            return PerceptionResult(
                scene_description=response,
                raw_response=response,
            )

        objects: list[DetectedObject] = []

        raw_objects = data.get("objects", [])

        if not isinstance(raw_objects, list):
            raw_objects = []

        for item in raw_objects:
            if not isinstance(item, dict):
                continue

            # ---------------------------------------------------------
            # Required object field: label
            # ---------------------------------------------------------

            label = item.get("label")

            if not isinstance(label, str):
                continue

            label = label.strip()

            if not label:
                continue

            # ---------------------------------------------------------
            # Optional field: confidence
            # ---------------------------------------------------------

            confidence = item.get("confidence")

            if confidence is not None:
                try:
                    confidence = float(confidence)
                except (TypeError, ValueError):
                    confidence = None

                if (
                    confidence is not None
                    and not 0.0 <= confidence <= 1.0
                ):
                    confidence = None

            # ---------------------------------------------------------
            # Optional field: bounding box
            # ---------------------------------------------------------

            bbox = RemoteVLMPerception._parse_bbox(
                item.get("bbox")
            )

            # ---------------------------------------------------------
            # Optional field: attributes
            # ---------------------------------------------------------

            attributes = item.get("attributes", {})

            if not isinstance(attributes, dict):
                attributes = {}

            objects.append(
                DetectedObject(
                    label=label,
                    confidence=confidence,
                    bbox=bbox,
                    attributes=attributes,
                )
            )

        # -------------------------------------------------------------
        # Scene description
        # -------------------------------------------------------------

        scene_description = data.get("scene_description")

        if isinstance(scene_description, str):
            scene_description = scene_description.strip()

            if not scene_description:
                scene_description = None
        else:
            scene_description = None

        return PerceptionResult(
            objects=tuple(objects),
            scene_description=scene_description,
            raw_response=response,
        )

    @staticmethod
    def _parse_bbox(
        raw_bbox: object,
    ) -> BoundingBox | None:
        """
        Parse and validate a 2D image-space bounding box.

        Invalid bounding boxes are discarded rather than allowing
        malformed geometry into downstream perception/grounding code.
        """

        if not isinstance(raw_bbox, dict):
            return None

        try:
            x_min = float(raw_bbox["x_min"])
            y_min = float(raw_bbox["y_min"])
            x_max = float(raw_bbox["x_max"])
            y_max = float(raw_bbox["y_max"])
        except (KeyError, TypeError, ValueError):
            return None

        if x_min >= x_max:
            return None

        if y_min >= y_max:
            return None

        return BoundingBox(
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
        )