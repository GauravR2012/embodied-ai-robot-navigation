from __future__ import annotations

import json

from perception.models import (
    BoundingBox,
    DetectedObject,
    PerceptionResult,
)
from perception.remote_vlm import RemoteVLMPerception


def test_perception_models():
    bbox = BoundingBox(
        x_min=10.0,
        y_min=20.0,
        x_max=100.0,
        y_max=200.0,
    )

    obj = DetectedObject(
        label="green target",
        confidence=0.95,
        bbox=bbox,
    )

    result = PerceptionResult(
        objects=(obj,),
        scene_description="A green target is visible.",
    )

    assert len(result.objects) == 1
    assert result.objects[0].label == "green target"
    assert result.objects[0].confidence == 0.95
    assert result.objects[0].bbox == bbox


def test_remote_vlm_parses_structured_response():
    response = json.dumps(
        {
            "scene_description": (
                "A robot sees a red obstacle and a green target."
            ),
            "objects": [
                {
                    "label": "red obstacle",
                    "confidence": 0.91,
                    "bbox": {
                        "x_min": 10,
                        "y_min": 20,
                        "x_max": 100,
                        "y_max": 200,
                    },
                    "attributes": {
                        "role": "obstacle",
                    },
                },
                {
                    "label": "green target",
                    "confidence": 0.96,
                    "attributes": {
                        "role": "navigation_target",
                    },
                },
            ],
        }
    )

    result = RemoteVLMPerception._parse_response(response)

    assert result.scene_description == (
        "A robot sees a red obstacle and a green target."
    )

    assert len(result.objects) == 2

    obstacle = result.objects[0]

    assert obstacle.label == "red obstacle"
    assert obstacle.confidence == 0.91
    assert obstacle.attributes["role"] == "obstacle"

    assert obstacle.bbox == BoundingBox(
        x_min=10.0,
        y_min=20.0,
        x_max=100.0,
        y_max=200.0,
    )

    target = result.objects[1]

    assert target.label == "green target"
    assert target.confidence == 0.96
    assert target.bbox is None
    assert target.attributes["role"] == "navigation_target"


def test_remote_vlm_handles_plain_text_response():
    response = "There is a red obstacle directly ahead."

    result = RemoteVLMPerception._parse_response(response)

    assert result.scene_description == response
    assert result.raw_response == response
    assert result.objects == ()


def test_remote_vlm_rejects_invalid_objects():
    response = json.dumps(
        {
            "scene_description": "Indoor environment.",
            "objects": [
                {
                    "label": "",
                    "confidence": 0.9,
                },
                {
                    "label": "invalid confidence",
                    "confidence": 2.0,
                },
                {
                    "label": "valid obstacle",
                    "confidence": 0.8,
                    "bbox": {
                        "x_min": 100,
                        "y_min": 100,
                        "x_max": 50,
                        "y_max": 200,
                    },
                },
            ],
        }
    )

    result = RemoteVLMPerception._parse_response(response)

    # Empty label is rejected.
    assert len(result.objects) == 2

    # Invalid confidence is removed rather than propagated.
    assert result.objects[0].label == "invalid confidence"
    assert result.objects[0].confidence is None

    # Invalid bounding box is removed rather than propagated.
    assert result.objects[1].label == "valid obstacle"
    assert result.objects[1].bbox is None


def test_remote_vlm_handles_non_object_json():
    response = json.dumps(
        ["not", "an", "object"]
    )

    result = RemoteVLMPerception._parse_response(response)

    assert result.objects == ()
    assert result.scene_description == response
    assert result.raw_response == response


def test_remote_vlm_handles_invalid_bbox():
    response = json.dumps(
        {
            "scene_description": "Indoor environment.",
            "objects": [
                {
                    "label": "obstacle",
                    "confidence": 0.8,
                    "bbox": {
                        "x_min": "invalid",
                        "y_min": 20,
                        "x_max": 100,
                        "y_max": 200,
                    },
                }
            ],
        }
    )

    result = RemoteVLMPerception._parse_response(response)

    assert len(result.objects) == 1
    assert result.objects[0].label == "obstacle"
    assert result.objects[0].bbox is None


def test_remote_vlm_handles_invalid_attributes():
    response = json.dumps(
        {
            "scene_description": "Indoor environment.",
            "objects": [
                {
                    "label": "obstacle",
                    "confidence": 0.8,
                    "attributes": "not a dictionary",
                }
            ],
        }
    )

    result = RemoteVLMPerception._parse_response(response)

    assert len(result.objects) == 1
    assert result.objects[0].label == "obstacle"
    assert result.objects[0].attributes == {}


def test_remote_vlm_handles_empty_response():
    response = ""

    result = RemoteVLMPerception._parse_response(response)

    assert result.objects == ()
    assert result.scene_description is None
    assert result.raw_response == ""


def test_remote_vlm_backend():
    class FakeVLMClient:
        def analyze(self, image_path, instruction):
            assert image_path == "/tmp/frame.jpg"
            assert instruction == "Identify obstacles."

            return json.dumps(
                {
                    "scene_description": "Obstacle detected.",
                    "objects": [
                        {
                            "label": "obstacle",
                            "confidence": 0.9,
                        }
                    ],
                }
            )

    backend = RemoteVLMPerception(FakeVLMClient())

    result = backend.analyze(
        image_path="/tmp/frame.jpg",
        instruction="Identify obstacles.",
    )

    assert result.scene_description == "Obstacle detected."
    assert len(result.objects) == 1
    assert result.objects[0].label == "obstacle"