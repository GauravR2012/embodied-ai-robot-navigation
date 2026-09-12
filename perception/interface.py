from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from .models import PerceptionResult


class PerceptionBackend(ABC):
    """Interface implemented by all perception backends."""

    @abstractmethod
    def analyze(
        self,
        image_path: str | Path,
        instruction: str,
    ) -> PerceptionResult:
        """Analyze an image and return structured perception."""
        raise NotImplementedError
