"""
Object search: find a described object, rotating in place if needed.

This module contains NO ROS code. The robot-specific operations are
supplied through the :class:`SearchEnvironment` protocol, which makes the
decision logic fully unit-testable.

Behaviour
---------

    view 0 (current heading)
        -> VLM detection
        -> accept only a matching object whose confidence >= threshold
           AND whose position can be measured with depth
        -> accepted: stop searching, hand the object to the planner
        -> otherwise: rotate by ``rotation_step_deg`` and look again

    after a full sweep without an accepted object
        -> rotate back to the original heading
        -> report "not found" (the caller must NOT navigate)

A VLM / network failure is never reported as "object not found": those
exceptions propagate to the caller untouched.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from .models import BoundingBox, DetectedObject, PerceptionResult


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class GroundingError(RuntimeError):
    """The object was detected but its 3D position could not be measured."""


class RotationError(RuntimeError):
    """The robot could not rotate to the next viewpoint."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ObjectSearchConfig:
    """Tunable parameters for the rotate-and-detect search."""

    # A detection below this confidence is treated as "not seen".
    min_confidence: float = 0.60

    # Rotation between viewpoints. The simulated camera has a 60 deg
    # horizontal field of view, so 45 deg leaves a 15 deg overlap.
    rotation_step_deg: float = 45.0

    # Total angle covered by the sweep (360 = look all the way around).
    sweep_deg: float = 360.0

    # Detections smaller than this (pixels, either side) are ignored.
    min_bbox_side_px: float = 8.0

    # Confidence assigned when the VLM omits one. 0.0 = never accepted.
    missing_confidence: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be within [0, 1].")

        if not 0.0 < self.rotation_step_deg <= 180.0:
            raise ValueError("rotation_step_deg must be in (0, 180].")

        if not 0.0 < self.sweep_deg <= 360.0:
            raise ValueError("sweep_deg must be in (0, 360].")

    @property
    def view_count(self) -> int:
        """Number of viewpoints, including the starting one."""
        return max(
            1,
            int(math.ceil(self.sweep_deg / self.rotation_step_deg - 1e-9)),
        )


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ObjectCandidate:
    """A VLM detection that matches the requested target."""

    label: str
    confidence: float
    bbox: BoundingBox
    view_index: int
    heading_offset_deg: float


@dataclass(frozen=True)
class SearchView:
    """What happened at one viewpoint (kept for logs / the UI)."""

    index: int
    heading_offset_deg: float
    scene_description: str | None
    note: str


@dataclass(frozen=True)
class SearchResult:
    """Outcome of a search."""

    found: bool
    target: str
    reason: str
    views_checked: int
    min_confidence: float
    candidate: ObjectCandidate | None = None
    grounding: Any = None
    best_confidence: float | None = None
    views: tuple[SearchView, ...] = ()

    # Reasons for found == False.
    NOT_DETECTED = "not_detected"
    LOW_CONFIDENCE = "low_confidence"
    UNGROUNDED = "ungrounded"
    ROTATION_FAILED = "rotation_failed"
    FOUND = "found"

    def message(self) -> str:
        """Human-readable explanation of the outcome."""

        if self.found and self.candidate is not None:
            return (
                f"Found '{self.target}' with confidence "
                f"{self.candidate.confidence:.2f} "
                f"(view {self.candidate.view_index + 1}, "
                f"{self.candidate.heading_offset_deg:+.0f} deg from the "
                "starting heading)."
            )

        if self.reason == self.LOW_CONFIDENCE:
            return (
                f"I could not confidently find '{self.target}'. "
                f"The best match had confidence {self.best_confidence:.2f}, "
                f"below the required {self.min_confidence:.2f} "
                f"({self.views_checked} view(s) checked). "
                "The robot did not navigate."
            )

        if self.reason == self.UNGROUNDED:
            return (
                f"'{self.target}' was detected but its distance could not "
                "be measured (no valid depth, possibly out of range). "
                "The robot did not navigate."
            )

        if self.reason == self.ROTATION_FAILED:
            return (
                f"I could not finish scanning for '{self.target}' because "
                "the robot was unable to rotate "
                f"(only {self.views_checked} view(s) checked). "
                "The robot did not navigate."
            )

        return (
            f"'{self.target}' is not in view: I scanned "
            f"{self.views_checked} view(s) and did not detect it. "
            "The robot did not navigate."
        )

    def to_summary(self) -> dict[str, Any]:
        """JSON-friendly summary for execution metadata."""

        summary: dict[str, Any] = {
            "target": self.target,
            "found": self.found,
            "reason": self.reason,
            "message": self.message(),
            "views_checked": self.views_checked,
            "min_confidence": self.min_confidence,
            "best_confidence": self.best_confidence,
            "views": [
                {
                    "index": view.index,
                    "heading_offset_deg": view.heading_offset_deg,
                    "scene_description": view.scene_description,
                    "note": view.note,
                }
                for view in self.views
            ],
        }

        if self.candidate is not None:
            summary["candidate"] = {
                "label": self.candidate.label,
                "confidence": self.candidate.confidence,
                "view_index": self.candidate.view_index,
                "heading_offset_deg": self.candidate.heading_offset_deg,
                "bbox": [
                    self.candidate.bbox.x_min,
                    self.candidate.bbox.y_min,
                    self.candidate.bbox.x_max,
                    self.candidate.bbox.y_max,
                ],
            }

        return summary


# ---------------------------------------------------------------------------
# Robot-side operations
# ---------------------------------------------------------------------------

class SearchEnvironment(Protocol):
    """Robot operations the search needs. Implemented by the ROS executor."""

    def capture(self) -> Any:
        """Return a snapshot (RGB + depth) taken while the robot is still."""

    def detect(self, frame: Any, target: str) -> PerceptionResult:
        """Run the VLM on ``frame`` looking for ``target``."""

    def ground(self, frame: Any, bbox: BoundingBox) -> Any:
        """
        Measure the object's position from depth.

        Raises:
            GroundingError: if the position cannot be measured.
        """

    def rotate(self, delta_deg: float) -> None:
        """
        Rotate in place by ``delta_deg`` (positive = counter-clockwise).

        Raises:
            RotationError: if the robot cannot rotate.
        """

    def return_to_start(self) -> None:
        """Rotate back to the heading the search started from."""


# ---------------------------------------------------------------------------
# Target text handling
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    {
        "the", "a", "an", "of", "to", "that", "this", "my", "your",
        "one", "object", "thing", "near", "nearest", "closest",
    }
)


def sanitize_target(target: str) -> str:
    """
    Reduce a target description to plain words.

    The text originates from the LLM and ends up inside a VLM prompt, so
    anything other than letters, digits, spaces, '-' and '_' is dropped.
    """

    if not isinstance(target, str):
        raise ValueError("Target must be a string.")

    cleaned = re.sub(r"[^\w\s\-]", " ", target, flags=re.UNICODE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned[:80]


def _singular(token: str) -> str:
    if len(token) > 3 and token.endswith("ies"):
        return token[:-3] + "y"

    if len(token) > 3 and token.endswith(("xes", "ches", "shes", "sses")):
        return token[:-2]

    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]

    return token


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())

    return {
        _singular(word)
        for word in words
        if word not in _STOPWORDS
    }


def target_tokens(target: str) -> frozenset[str]:
    """Normalised word set of a target description."""
    return frozenset(_tokens(target))


def detection_matches(
    detected: DetectedObject,
    tokens: frozenset[str],
) -> bool:
    """
    True if every word of the target appears in the detection's label or
    attributes ("red box" matches label "red box", or label "box" with
    attributes {"color": "red"}).
    """

    if not tokens:
        return False

    haystack = _tokens(detected.label)

    for key, value in detected.attributes.items():
        haystack |= _tokens(str(key))
        haystack |= _tokens(str(value))

    return tokens <= haystack


def build_find_prompt(target: str, width: int, height: int) -> str:
    """Instruction sent to the VLM to locate ``target``."""

    target = sanitize_target(target)

    return f"""
You are an object localization system.

Find the object described below in the image.

Target:
{target}

Return ONLY valid JSON.

Required format:

{{
  "scene_description": "string",
  "objects": [
    {{
      "label": "{target}",
      "confidence": 0.0,
      "bbox": {{
        "x_min": 0,
        "y_min": 0,
        "x_max": 0,
        "y_max": 0
      }},
      "attributes": {{}}
    }}
  ]
}}

Bounding box coordinates must be pixel coordinates in the original image.

The original image resolution is {width}x{height}.

The bounding box must satisfy:

x_min < x_max
y_min < y_max

Rules:
- Only report the target if you can clearly see an object that matches
  the full description (including its color, if one is given).
- Do NOT report a different object (for example a box of another color)
  as the target.
- If you are unsure, report it with a LOW confidence (below 0.5).
- If the target is not visible, return:

{{
  "scene_description": "target not visible",
  "objects": []
}}

Do not include markdown.
Do not include explanations outside the JSON.
""".strip()


# ---------------------------------------------------------------------------
# Searcher
# ---------------------------------------------------------------------------

class ObjectSearcher:
    """Runs the rotate-and-detect search through a SearchEnvironment."""

    def __init__(
        self,
        environment: SearchEnvironment,
        config: ObjectSearchConfig | None = None,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.environment = environment
        self.config = config or ObjectSearchConfig()
        self._log = log or (lambda _message: None)

    # ------------------------------------------------------------------

    def extract_candidates(
        self,
        result: PerceptionResult,
        tokens: frozenset[str],
        view_index: int,
        heading_offset_deg: float,
    ) -> list[ObjectCandidate]:
        """Matching detections of one view, best confidence first."""

        candidates: list[ObjectCandidate] = []

        for detected in result.objects:
            if detected.bbox is None:
                continue

            if not detection_matches(detected, tokens):
                continue

            width = detected.bbox.x_max - detected.bbox.x_min
            height = detected.bbox.y_max - detected.bbox.y_min

            if (
                width < self.config.min_bbox_side_px
                or height < self.config.min_bbox_side_px
            ):
                continue

            confidence = (
                detected.confidence
                if detected.confidence is not None
                else self.config.missing_confidence
            )

            candidates.append(
                ObjectCandidate(
                    label=detected.label,
                    confidence=float(confidence),
                    bbox=detected.bbox,
                    view_index=view_index,
                    heading_offset_deg=heading_offset_deg,
                )
            )

        candidates.sort(key=lambda c: c.confidence, reverse=True)

        return candidates

    # ------------------------------------------------------------------

    def search(self, target: str) -> SearchResult:
        """Look for ``target``; rotate and repeat until found or swept."""

        clean_target = sanitize_target(target)
        tokens = target_tokens(clean_target)

        if not tokens:
            raise ValueError(
                f"Target description is empty or unusable: {target!r}"
            )

        config = self.config
        view_total = config.view_count

        views: list[SearchView] = []
        best_confidence: float | None = None
        saw_ungrounded = False
        rotation_failed = False

        for index in range(view_total):
            heading = index * config.rotation_step_deg

            frame = self.environment.capture()

            perception = self.environment.detect(frame, clean_target)

            candidates = self.extract_candidates(
                perception,
                tokens,
                index,
                heading,
            )

            note = "no matching object"

            if candidates:
                top = candidates[0]

                if (
                    best_confidence is None
                    or top.confidence > best_confidence
                ):
                    best_confidence = top.confidence

                note = (
                    f"best match '{top.label}' "
                    f"confidence {top.confidence:.2f}"
                )

            for candidate in candidates:
                if candidate.confidence < config.min_confidence:
                    note += (
                        f" (below threshold {config.min_confidence:.2f})"
                    )
                    break

                try:
                    grounding = self.environment.ground(
                        frame,
                        candidate.bbox,
                    )
                except GroundingError as exc:
                    saw_ungrounded = True
                    note = (
                        f"'{candidate.label}' confidence "
                        f"{candidate.confidence:.2f} but not measurable: "
                        f"{exc}"
                    )
                    continue

                note = (
                    f"'{candidate.label}' accepted, "
                    f"confidence {candidate.confidence:.2f}"
                )

                views.append(
                    SearchView(
                        index=index,
                        heading_offset_deg=heading,
                        scene_description=perception.scene_description,
                        note=note,
                    )
                )

                self._log(
                    f"View {index + 1}/{view_total} "
                    f"({heading:+.0f} deg): {note}"
                )

                return SearchResult(
                    found=True,
                    target=clean_target,
                    reason=SearchResult.FOUND,
                    views_checked=index + 1,
                    min_confidence=config.min_confidence,
                    candidate=candidate,
                    grounding=grounding,
                    best_confidence=candidate.confidence,
                    views=tuple(views),
                )

            views.append(
                SearchView(
                    index=index,
                    heading_offset_deg=heading,
                    scene_description=perception.scene_description,
                    note=note,
                )
            )

            self._log(
                f"View {index + 1}/{view_total} "
                f"({heading:+.0f} deg): {note}"
            )

            if index < view_total - 1:
                try:
                    self.environment.rotate(config.rotation_step_deg)
                except RotationError as exc:
                    rotation_failed = True
                    self._log(f"Rotation failed: {exc}")
                    break

        # Nothing accepted: put the robot back where it started.
        try:
            self.environment.return_to_start()
        except RotationError as exc:
            self._log(f"Could not return to the starting heading: {exc}")

        if rotation_failed:
            reason = SearchResult.ROTATION_FAILED
        elif saw_ungrounded:
            reason = SearchResult.UNGROUNDED
        elif best_confidence is not None:
            reason = SearchResult.LOW_CONFIDENCE
        else:
            reason = SearchResult.NOT_DETECTED

        return SearchResult(
            found=False,
            target=clean_target,
            reason=reason,
            views_checked=len(views),
            min_confidence=config.min_confidence,
            best_confidence=best_confidence,
            views=tuple(views),
        )
