from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExecutionResult:
    """Structured result of executing a robot command."""

    success: bool
    message: str
    error_code: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] | None = None

    @classmethod
    def succeeded(
        cls,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> "ExecutionResult":
        return cls(
            success=True,
            message=message,
            metadata=metadata,
        )

    @classmethod
    def failed(
        cls,
        error_code: str,
        error_message: str,
        metadata: dict[str, Any] | None = None,
    ) -> "ExecutionResult":
        return cls(
            success=False,
            message="Robot command execution failed.",
            error_code=error_code,
            error_message=error_message,
            metadata=metadata,
        )
