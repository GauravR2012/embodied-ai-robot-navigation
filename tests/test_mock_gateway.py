"""
Tests for the local mock gateway and the VLM batch checker.

The mock is started on an ephemeral port and driven through the REAL
client classes (AIGatewayClient, RemoteLLMBackend, RemoteVLMPerception),
so these tests also prove the mock is wire-compatible with the code that
will talk to the real Vast.ai gateway.
"""

from __future__ import annotations

import threading

import cv2
import numpy as np
import pytest

from ai_client.gateway_client import AIGatewayClient
from ai_client.llm_client import RemoteLLMClient
from ai_client.vlm_client import RemoteVLMClient
from command_schema import parse_command
from llm_adapter import LLMCommandAdapter, RemoteLLMBackend
from perception.object_search import (
    ObjectSearchConfig,
    ObjectSearcher,
    build_find_prompt,
)
from perception.remote_vlm import RemoteVLMPerception
from task_planner import plan_command
from tools import mock_gateway, vlm_batch_check as batch


LOCATIONS = {"executor test": "Executor Test", "base station": "Base Station"}

RED = (0, 0, 200)      # BGR
BLUE = (200, 0, 0)
GREEN = (0, 160, 0)


@pytest.fixture(scope="module")
def gateway():
    config = mock_gateway.GatewayConfig(
        locations=LOCATIONS,
        force_confidence=None,
        latency_sec=0.0,
        debug_dir=None,
    )

    server = mock_gateway.make_server("127.0.0.1", 0, config)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    yield AIGatewayClient(
        base_url=f"http://127.0.0.1:{server.server_address[1]}",
        timeout=10.0,
    )

    server.shutdown()
    server.server_close()


def frame(*boxes, size=(640, 480)):
    """Grey floor with solid coloured boxes: (x0, y0, x1, y1, bgr)."""

    image = np.full((size[1], size[0], 3), 120, dtype=np.uint8)

    for x0, y0, x1, y1, color in boxes:
        cv2.rectangle(image, (x0, y0), (x1, y1), color, -1)

    return image


def write(tmp_path, name, image):
    path = tmp_path / name
    cv2.imwrite(str(path), image)
    return path


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def test_health_matches_real_schema(gateway):
    health = gateway.health()

    assert health["status"] == "ok"
    assert health["llm_loaded"] is True
    assert health["vlm_loaded"] is True
    assert health["gpu"].startswith("MOCK")


# ---------------------------------------------------------------------------
# Mock LLM -> the real adapter, schema and planner
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text, expected_skills",
    [
        (
            "Go to the red box",
            ["find_object", "navigate_to_target", "wait_for_navigation_result"],
        ),
        (
            "go to Executor Test",
            ["resolve_target", "navigate_to_target", "wait_for_navigation_result"],
        ),
        ("turn left 45 degrees", ["rotate"]),
        ("turn around", ["rotate"]),
        ("wait 3 seconds", ["wait"]),
        ("stop", ["stop"]),
    ],
)
def test_llm_commands_pass_the_real_pipeline(gateway, text, expected_skills):
    adapter = LLMCommandAdapter(RemoteLLMBackend(RemoteLLMClient(gateway)))

    command = adapter.interpret_to_dict(text)
    skills = plan_command(parse_command(command))

    assert [s.name for s in skills] == expected_skills


def test_llm_object_vs_named_location(gateway):
    llm = RemoteLLMClient(gateway)

    import json

    named = json.loads(llm.generate("Go to Executor Test"))
    obj = json.loads(llm.generate("go to the blue box"))

    assert named["target"] == {"type": "named_location", "value": "Executor Test"}
    assert obj["target"] == {"type": "object", "value": "blue box"}


def test_llm_rejects_gibberish_with_http_500(gateway):
    from ai_client.gateway_client import AIGatewayError

    with pytest.raises(AIGatewayError):
        gateway.chat("flibbertigibbet")


# ---------------------------------------------------------------------------
# Mock VLM through the real parser
# ---------------------------------------------------------------------------

def perceive(gateway, tmp_path, image, target="red box"):
    path = write(tmp_path, "frame.png", image)

    perception = RemoteVLMPerception(RemoteVLMClient(gateway))

    return perception.analyze(
        image_path=path,
        instruction=build_find_prompt(target, 640, 480),
    )


def test_red_box_detected_with_accurate_bbox(gateway, tmp_path):
    result = perceive(gateway, tmp_path, frame((300, 200, 380, 300, RED)))

    assert len(result.objects) == 1

    detected = result.objects[0]

    assert detected.label == "red box"
    assert detected.bbox.x_min == pytest.approx(300, abs=2)
    assert detected.bbox.y_min == pytest.approx(200, abs=2)
    assert detected.bbox.x_max == pytest.approx(381, abs=2)
    assert detected.bbox.y_max == pytest.approx(301, abs=2)
    assert 0.5 <= detected.confidence <= 0.95


def test_wrong_colour_is_not_reported(gateway, tmp_path):
    result = perceive(gateway, tmp_path, frame((300, 200, 380, 300, BLUE)))

    assert result.objects == ()


def test_picks_the_requested_colour_among_several(gateway, tmp_path):
    image = frame(
        (50, 200, 130, 300, BLUE),
        (300, 200, 380, 300, RED),
        (500, 200, 580, 300, GREEN),
    )

    for target, x_min in (("red box", 300), ("blue box", 50), ("green box", 500)):
        result = perceive(gateway, tmp_path, image, target)

        assert len(result.objects) == 1
        assert result.objects[0].bbox.x_min == pytest.approx(x_min, abs=2)


def test_empty_scene_and_noise_give_no_detection(gateway, tmp_path):
    image = frame((10, 10, 14, 14, RED))  # 5x5 px speckle

    assert perceive(gateway, tmp_path, image).objects == ()


def test_target_without_colour_word_is_not_guessed(gateway, tmp_path):
    image = frame((300, 200, 380, 300, RED))

    assert perceive(gateway, tmp_path, image, "chair").objects == ()


def test_bigger_box_gets_higher_confidence(gateway, tmp_path):
    small = perceive(gateway, tmp_path, frame((300, 200, 330, 230, RED)))
    large = perceive(gateway, tmp_path, frame((200, 100, 440, 380, RED)))

    assert large.objects[0].confidence > small.objects[0].confidence


def test_force_confidence_option():
    config = mock_gateway.GatewayConfig(LOCATIONS, 0.3, 0.0, None)
    server = mock_gateway.make_server("127.0.0.1", 0, config)

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        client = AIGatewayClient(
            base_url=f"http://127.0.0.1:{server.server_address[1]}"
        )

        import base64, json

        ok, buffer = cv2.imencode(".png", frame((300, 200, 380, 300, RED)))
        reply = client.perception(
            base64.b64encode(buffer.tobytes()).decode(),
            build_find_prompt("red box", 640, 480),
        )

        objects = json.loads(reply["response"])["objects"]

        assert objects[0]["confidence"] == 0.3
    finally:
        server.shutdown()
        server.server_close()


def test_bad_image_returns_error(gateway):
    from ai_client.gateway_client import AIGatewayError

    with pytest.raises(AIGatewayError):
        gateway.perception("bm90IGFuIGltYWdl", build_find_prompt("red box", 1, 1))


def test_extract_target_from_real_prompt():
    prompt = build_find_prompt("red box", 640, 480)

    assert mock_gateway.extract_target(prompt) == "red box"


# ---------------------------------------------------------------------------
# The mock plugged into the real search logic
# ---------------------------------------------------------------------------

class MockWorldEnv:
    """Robot that sees a red box only at one heading (0-based view index)."""

    def __init__(self, gateway, tmp_path, visible_view):
        self.gateway, self.tmp_path = gateway, tmp_path
        self.visible_view = visible_view
        self.view = 0
        self.returned = False
        self.perception = RemoteVLMPerception(RemoteVLMClient(gateway))

    def capture(self):
        return self.view

    def detect(self, frame_id, target):
        image = frame((280, 180, 360, 300, RED)) if frame_id == self.visible_view \
            else frame()
        path = write(self.tmp_path, "view.png", image)

        return self.perception.analyze(
            image_path=path,
            instruction=build_find_prompt(target, 640, 480),
        )

    def ground(self, frame_id, bbox):
        return "grounded"

    def rotate(self, delta_deg):
        self.view += 1

    def return_to_start(self):
        self.returned = True


def test_search_finds_box_after_spinning_with_mock_vlm(gateway, tmp_path):
    env = MockWorldEnv(gateway, tmp_path, visible_view=3)

    result = ObjectSearcher(env, ObjectSearchConfig()).search("red box")

    assert result.found
    assert result.views_checked == 4
    assert not env.returned


def test_search_reports_not_found_with_mock_vlm(gateway, tmp_path):
    env = MockWorldEnv(gateway, tmp_path, visible_view=999)

    result = ObjectSearcher(env, ObjectSearchConfig()).search("red box")

    assert not result.found
    assert result.views_checked == 8
    assert env.returned


# ---------------------------------------------------------------------------
# Batch checker scoring
# ---------------------------------------------------------------------------

def result(name, expected, conf):
    return batch.FrameResult(name, expected, conf)


def test_expected_from_filename():
    assert batch.expected_from_name("pos_001.jpg") is True
    assert batch.expected_from_name("NEG_cat.png") is False
    assert batch.expected_from_name("other.jpg") is None


def test_confusion_counts_and_threshold_effect():
    rows = [
        result("pos_1", True, 0.9),
        result("pos_2", True, 0.5),
        result("pos_3", True, None),
        result("neg_1", False, 0.7),
        result("neg_2", False, None),
        result("free", None, 0.99),
    ]

    c = batch.confusion(rows, 0.6)

    assert (c.true_positive, c.false_positive, c.false_negative, c.true_negative) \
        == (1, 1, 2, 1)
    assert c.precision == pytest.approx(0.5)
    assert c.recall == pytest.approx(1 / 3)

    assert batch.confusion(rows, 0.8).false_positive == 0


def test_lowest_safe_threshold():
    rows = [result("neg_1", False, 0.7), result("pos_1", True, 0.95)]

    assert batch.lowest_safe_threshold(rows, [0.5, 0.6, 0.7, 0.8]) == 0.8
    assert batch.lowest_safe_threshold(
        [result("neg_1", False, 0.99)], [0.5, 0.9]
    ) is None


def test_errors_are_not_scored():
    rows = [batch.FrameResult("pos_1", True, None, error="boom")]

    c = batch.confusion(rows, 0.5)

    assert (c.true_positive, c.false_negative) == (0, 0)


def test_batch_check_end_to_end_with_mock(gateway, tmp_path, capsys):
    cv2.imwrite(str(tmp_path / "pos_001.png"), frame((280, 180, 360, 300, RED)))
    cv2.imwrite(str(tmp_path / "neg_001.png"), frame((280, 180, 360, 300, BLUE)))
    cv2.imwrite(str(tmp_path / "neg_002.png"), frame())

    code = batch.main(
        [
            "--images", str(tmp_path),
            "--target", "red box",
            "--gateway-url", gateway.base_url,
        ]
    )

    out = capsys.readouterr().out

    assert code == 0
    assert "SUMMARY" in out
    assert "Lowest threshold with no false positives" in out

    rows = batch.run_frames(
        batch.collect_images([str(tmp_path)]),
        "red box",
        gateway.base_url,
        10.0,
        verbose=False,
    )

    c = batch.confusion(rows, 0.6)

    assert (c.true_positive, c.false_positive, c.false_negative, c.true_negative) \
        == (1, 0, 0, 2)
