"""Schema + planner tests for object navigation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from command_schema import ObjectTarget, parse_command
from execution_context import ExecutionContext
from task_planner import plan_command


def names(command):
    return [skill.name for skill in plan_command(parse_command(command))]


def test_object_target_is_accepted():
    command = parse_command(
        {"action": "navigate", "target": {"type": "object", "value": "red box"}}
    )

    assert isinstance(command.target, ObjectTarget)


def test_object_navigation_plan_searches_before_navigating():
    assert names(
        {"action": "navigate", "target": {"type": "object", "value": "red box"}}
    ) == ["find_object", "navigate_to_target", "wait_for_navigation_result"]


def test_object_plan_carries_type_and_value():
    skills = plan_command(
        parse_command(
            {"action": "navigate", "target": {"type": "object", "value": "blue box"}}
        )
    )

    for skill in skills[:2]:
        assert skill.parameters["target_type"] == "object"
        assert skill.parameters["target_value"] == "blue box"


def test_named_location_plan_is_unchanged():
    assert names(
        {
            "action": "navigate",
            "target": {"type": "named_location", "value": "Station 1"},
        }
    ) == ["resolve_target", "navigate_to_target", "wait_for_navigation_result"]


def test_sequence_with_object_and_named_location():
    assert names(
        {
            "action": "sequence",
            "commands": [
                {"action": "navigate", "target": {"type": "object", "value": "red box"}},
                {"action": "wait", "duration": 2, "unit": "s"},
                {
                    "action": "navigate",
                    "target": {"type": "named_location", "value": "Base Station"},
                },
            ],
        }
    ) == [
        "find_object",
        "navigate_to_target",
        "wait_for_navigation_result",
        "wait",
        "resolve_target",
        "navigate_to_target",
        "wait_for_navigation_result",
    ]


@pytest.mark.parametrize(
    "target",
    [
        {"type": "coordinates", "value": "1,2"},
        {"type": "object"},
        {"type": "object", "value": ""},
        {"type": "object", "value": "red box", "x": 1.0},
    ],
)
def test_invalid_targets_rejected(target):
    with pytest.raises(ValidationError):
        parse_command({"action": "navigate", "target": target})


def test_target_keys_do_not_collide():
    assert ExecutionContext.target_key("object", "box") != ExecutionContext.target_key(
        "named_location", "box"
    )
