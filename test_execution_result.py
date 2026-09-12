from __future__ import annotations

from execution_result import ExecutionResult


def test_success_result() -> None:
    result = ExecutionResult.succeeded(
        message="Test completed.",
        metadata={"test": True},
    )

    assert result.success is True
    assert result.message == "Test completed."
    assert result.error_code is None
    assert result.error_message is None
    assert result.metadata == {"test": True}


def test_failure_result() -> None:
    result = ExecutionResult.failed(
        error_code="TARGET_NOT_FOUND",
        error_message="Requested target was not found.",
    )

    assert result.success is False
    assert result.message == "Robot command execution failed."
    assert result.error_code == "TARGET_NOT_FOUND"
    assert result.error_message == (
        "Requested target was not found."
    )
    assert result.metadata is None


def test_failure_result_with_metadata() -> None:
    result = ExecutionResult.failed(
        error_code="SKILL_EXECUTION_FAILED",
        error_message="Navigation failed.",
        metadata={
            "failed_skill": "navigate_to_target",
            "completed_until_failure": 1,
        },
    )

    assert result.success is False
    assert result.error_code == "SKILL_EXECUTION_FAILED"
    assert result.error_message == "Navigation failed."

    assert result.metadata == {
        "failed_skill": "navigate_to_target",
        "completed_until_failure": 1,
    }


def test_result_is_immutable() -> None:
    result = ExecutionResult.succeeded(
        message="Immutable test."
    )

    try:
        result.success = False
    except Exception:
        pass
    else:
        raise AssertionError(
            "ExecutionResult should be immutable."
        )


def main() -> int:
    test_success_result()
    test_failure_result()
    test_failure_result_with_metadata()
    test_result_is_immutable()

    print("ExecutionResult tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
