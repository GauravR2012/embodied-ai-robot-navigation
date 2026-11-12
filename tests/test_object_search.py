"""Unit tests for the ROS-free object search logic (no GPU / ROS needed)."""

from __future__ import annotations

import pytest

from perception.models import BoundingBox, DetectedObject, PerceptionResult
from perception.object_search import (
    GroundingError,
    ObjectSearchConfig,
    ObjectSearcher,
    RotationError,
    SearchResult,
    build_find_prompt,
    sanitize_target,
    target_tokens,
)


def det(label, conf, bbox=(100, 100, 220, 220), attributes=None):
    return DetectedObject(
        label=label,
        confidence=conf,
        bbox=BoundingBox(*bbox) if bbox is not None else None,
        attributes=attributes or {},
    )


def view(*objects, scene="scene"):
    return PerceptionResult(objects=tuple(objects), scene_description=scene)


EMPTY = view()


class FakeEnv:
    """Scripted environment: view i is what the robot sees after i rotations."""

    def __init__(self, views, ungroundable=(), rotate_fails_on=None):
        self.views = views
        self.ungroundable = set(ungroundable)
        self.rotate_fails_on = rotate_fails_on
        self.rotations = []
        self.captures = 0
        self.returned = False
        self.heading_index = 0

    def capture(self):
        self.captures += 1
        return self.heading_index

    def detect(self, frame, target):
        return self.views[min(frame, len(self.views) - 1)]

    def ground(self, frame, bbox):
        if frame in self.ungroundable:
            raise GroundingError("no valid depth")
        return {"frame": frame}

    def rotate(self, delta_deg):
        if self.rotate_fails_on == len(self.rotations) + 1:
            raise RotationError("blocked")
        self.rotations.append(delta_deg)
        self.heading_index += 1

    def return_to_start(self):
        self.returned = True


def run(env, target="red box", **config):
    return ObjectSearcher(env, ObjectSearchConfig(**config)).search(target)


# ---------------------------------------------------------------------------
# Found
# ---------------------------------------------------------------------------

def test_found_in_first_view_does_not_rotate():
    env = FakeEnv([view(det("red box", 0.92))])

    result = run(env)

    assert result.found
    assert result.reason == SearchResult.FOUND
    assert env.rotations == []
    assert env.captures == 1
    assert not env.returned
    assert result.candidate.confidence == pytest.approx(0.92)
    assert result.grounding == {"frame": 0}


def test_rotates_until_object_comes_into_view():
    env = FakeEnv([EMPTY, EMPTY, view(det("red box", 0.8))])

    result = run(env)

    assert result.found
    assert env.rotations == [45.0, 45.0]
    assert result.views_checked == 3
    assert result.candidate.view_index == 2
    assert result.candidate.heading_offset_deg == pytest.approx(90.0)
    assert not env.returned
    assert "0.80" in result.message()


def test_low_confidence_first_then_confident_later_view():
    env = FakeEnv(
        [
            view(det("red box", 0.30)),
            EMPTY,
            EMPTY,
            view(det("red box", 0.88)),
        ]
    )

    result = run(env)

    assert result.found
    assert env.rotations == [45.0, 45.0, 45.0]
    assert result.candidate.confidence == pytest.approx(0.88)


def test_best_candidate_in_a_view_is_tried_first():
    env = FakeEnv(
        [
            view(
                det("red box", 0.7, bbox=(0, 0, 50, 50)),
                det("red box", 0.95, bbox=(300, 100, 400, 200)),
            )
        ]
    )

    result = run(env)

    assert result.candidate.confidence == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# Not found -> the caller must not navigate
# ---------------------------------------------------------------------------

def test_not_visible_anywhere_scans_full_turn_and_returns():
    env = FakeEnv([EMPTY])

    result = run(env)

    assert not result.found
    assert result.reason == SearchResult.NOT_DETECTED
    assert result.grounding is None
    assert result.views_checked == 8
    assert env.rotations == [45.0] * 7
    assert env.returned
    assert "did not navigate" in result.message()


def test_only_low_confidence_is_reported_as_not_found():
    env = FakeEnv([view(det("red box", 0.40))])

    result = run(env)

    assert not result.found
    assert result.reason == SearchResult.LOW_CONFIDENCE
    assert result.best_confidence == pytest.approx(0.40)
    assert env.returned
    message = result.message()
    assert "0.40" in message and "0.60" in message


def test_confidence_exactly_at_threshold_is_accepted():
    env = FakeEnv([view(det("red box", 0.60))])

    assert run(env).found


def test_missing_confidence_is_not_trusted_by_default():
    env = FakeEnv([view(det("red box", None))])

    result = run(env)

    assert not result.found
    assert result.reason == SearchResult.LOW_CONFIDENCE


def test_detection_without_depth_is_not_navigable():
    env = FakeEnv([view(det("red box", 0.95))], ungroundable={0, 1, 2, 3, 4, 5, 6, 7})

    result = run(env)

    assert not result.found
    assert result.reason == SearchResult.UNGROUNDED
    assert env.returned


def test_ungroundable_view_does_not_stop_the_search():
    env = FakeEnv(
        [view(det("red box", 0.95)), view(det("red box", 0.90))],
        ungroundable={0},
    )

    result = run(env)

    assert result.found
    assert result.candidate.view_index == 1


def test_rotation_failure_stops_search_and_reports_it():
    env = FakeEnv([EMPTY], rotate_fails_on=3)

    result = run(env)

    assert not result.found
    assert result.reason == SearchResult.ROTATION_FAILED
    assert result.views_checked == 3
    assert env.returned


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def test_wrong_color_is_not_a_match():
    env = FakeEnv([view(det("blue box", 0.99))])

    result = run(env)

    assert not result.found
    assert result.reason == SearchResult.NOT_DETECTED


def test_color_in_attributes_matches():
    env = FakeEnv([view(det("box", 0.9, attributes={"color": "red"}))])

    assert run(env).found


def test_plural_and_articles_are_ignored():
    env = FakeEnv([view(det("Red Box", 0.9))])

    assert run(env, target="the red boxes").found


def test_detection_without_bbox_is_ignored():
    env = FakeEnv([view(det("red box", 0.99, bbox=None))])

    assert not run(env).found


def test_tiny_bbox_is_ignored():
    env = FakeEnv([view(det("red box", 0.99, bbox=(10, 10, 14, 14)))])

    assert not run(env).found


def test_single_word_target():
    env = FakeEnv([view(det("chair", 0.9))])

    assert run(env, target="chair").found


# ---------------------------------------------------------------------------
# Config / helpers
# ---------------------------------------------------------------------------

def test_view_count():
    assert ObjectSearchConfig().view_count == 8
    assert ObjectSearchConfig(rotation_step_deg=90).view_count == 4
    assert ObjectSearchConfig(rotation_step_deg=100).view_count == 4
    assert ObjectSearchConfig(rotation_step_deg=180, sweep_deg=180).view_count == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_confidence": 1.5},
        {"min_confidence": -0.1},
        {"rotation_step_deg": 0},
        {"rotation_step_deg": 200},
        {"sweep_deg": 0},
        {"sweep_deg": 400},
    ],
)
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        ObjectSearchConfig(**kwargs)


def test_empty_target_rejected():
    with pytest.raises(ValueError):
        run(FakeEnv([EMPTY]), target="   the   ")


def test_sanitize_target_strips_prompt_injection_characters():
    cleaned = sanitize_target('red "box"\n{"ignore": 1} }}\n')

    for bad in ['"', "{", "}", "\n"]:
        assert bad not in cleaned

    assert cleaned.startswith("red box")


def test_build_find_prompt_contains_target_and_resolution():
    prompt = build_find_prompt('red box"}\n', 640, 480)

    assert "red box" in prompt
    assert "640x480" in prompt
    assert '"}\n' not in prompt


def test_target_tokens_normalise():
    assert target_tokens("The Red Boxes") == frozenset({"red", "box"})


def test_summary_is_json_friendly():
    import json

    env = FakeEnv([view(det("red box", 0.9))])
    summary = run(env).to_summary()

    json.dumps(summary)

    assert summary["found"] is True
    assert summary["candidate"]["confidence"] == pytest.approx(0.9)
