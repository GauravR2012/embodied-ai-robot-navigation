from __future__ import annotations

import base64

import pytest

from ai_client.gateway_client import AIGatewayClient
from ai_client.llm_client import RemoteLLMClient
from ai_client.vlm_client import RemoteVLMClient


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def test_gateway_chat(monkeypatch):
    def fake_post(*args, **kwargs):
        return FakeResponse(
            {
                "response": '{"action":"stop"}',
                "model": "test-model",
            }
        )

    monkeypatch.setattr(
        "ai_client.gateway_client.requests.post",
        fake_post,
    )

    client = AIGatewayClient()

    result = client.chat(
        instruction="Stop the robot.",
    )

    assert result["response"] == '{"action":"stop"}'


def test_remote_llm_client():
    class FakeGateway:
        def chat(self, instruction, context=None):
            return {
                "response": '{"action":"stop"}',
                "model": "test-model",
            }

        def health(self):
            return {"status": "ok"}

    client = RemoteLLMClient(FakeGateway())

    response = client.generate(
        "Stop the robot."
    )

    assert response == '{"action":"stop"}'


def test_remote_vlm_client(tmp_path):
    image_path = tmp_path / "test.jpg"
    image_path.write_bytes(b"fake-image-data")

    class FakeGateway:
        def perception(self, image_base64, instruction):
            assert base64.b64decode(image_base64) == b"fake-image-data"

            return {
                "response": "A red obstacle is visible.",
                "model": "test-vlm",
            }

        def health(self):
            return {"status": "ok"}

    client = RemoteVLMClient(FakeGateway())

    response = client.analyze(
        image_path,
        "Describe the scene.",
    )

    assert response == "A red obstacle is visible."

def test_remote_llm_backend():
    from llm_adapter import (
        LLMCommandAdapter,
        RemoteLLMBackend,
    )

    class FakeRemoteLLM:
        def generate(self, instruction):
            assert instruction == "Stop the robot."

            return '{"action":"stop"}'

    backend = RemoteLLMBackend(FakeRemoteLLM())
    adapter = LLMCommandAdapter(backend)

    command = adapter.interpret("Stop the robot.")

    assert command.action == "stop"