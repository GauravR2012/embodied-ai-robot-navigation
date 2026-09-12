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
                },
                {
                    "label": "green target",
                    "confidence": 0.96,
                },
            ],
        }
    )

    result = RemoteVLMPerception._parse_response(response)

    assert result.scene_description == (
        "A robot sees a red obstacle and a green target."
    )
    assert len(result.objects) == 2
    assert result.objects[0].label == "red obstacle"
    assert result.objects[1].label == "green target"


def test_remote_vlm_handles_plain_text_response():
    response = "There is a red obstacle directly ahead."

    result = RemoteVLMPerception._parse_response(response)

    assert result.scene_description == response
    assert result.raw_response == response
    assert result.objects == ()


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