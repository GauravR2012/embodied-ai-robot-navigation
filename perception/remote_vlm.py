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
    def _clean_json_response(response: str) -> str:
        """
        Normalize common VLM JSON formatting before parsing.

        Supported cases include:

            {"objects": [...]}

        and:

            ```json
            {"objects": [...]}
            ```

        and responses containing explanatory text surrounding a JSON
        object.

        This method only prepares the text for JSON parsing. It does
        not validate the resulting structure.
        """

        cleaned = response.strip()

        if not cleaned:
            return cleaned

        # -------------------------------------------------------------
        # Remove Markdown code fences.
        # -------------------------------------------------------------

        if cleaned.startswith("```"):
            lines = cleaned.splitlines()

            # Remove opening fence, e.g. ```json or ```
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]

            # Remove closing fence.
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            cleaned = "\n".join(lines).strip()

        # -------------------------------------------------------------
        # If the response is already valid JSON, return it directly.
        # -------------------------------------------------------------

        try:
            json.loads(cleaned)
            return cleaned
        except json.JSONDecodeError:
            pass

        # -------------------------------------------------------------
        # Attempt to extract an outer JSON object.
        #
        # This handles responses such as:
        #
        #   Here is the result:
        #   {"objects": [...]}
        #
        # while still relying on json.loads() for actual validation.
        # -------------------------------------------------------------

        first_brace = cleaned.find("{")
        last_brace = cleaned.rfind("}")

        if (
            first_brace != -1
            and last_brace != -1
            and first_brace < last_brace
        ):
            candidate = cleaned[first_brace : last_brace + 1]

            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass

        return cleaned

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

        The VLM output is untrusted.

        The parser therefore:
          - normalizes common Markdown JSON formatting,
          - validates the top-level JSON structure,
          - ignores malformed individual objects,
          - removes invalid optional fields,
          - discards malformed bounding boxes,
          - preserves the original response for debugging.

        Invalid or truncated JSON falls back to a plain-text scene
        description rather than being treated as structured perception.
        """

        if not isinstance(response, str):
            response = str(response)

        raw_response = response.strip()

        if not raw_response:
            return PerceptionResult(
                scene_description=None,
                raw_response=raw_response,
            )

        response_for_parsing = (
            RemoteVLMPerception._clean_json_response(
                raw_response
            )
        )

        try:
            data = json.loads(response_for_parsing)
        except json.JSONDecodeError:
            return PerceptionResult(
                scene_description=raw_response,
                raw_response=raw_response,
            )

        if not isinstance(data, dict):
            return PerceptionResult(
                scene_description=raw_response,
                raw_response=raw_response,
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
            raw_response=raw_response,
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