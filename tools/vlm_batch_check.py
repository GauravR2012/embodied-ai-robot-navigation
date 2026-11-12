#!/usr/bin/env python3
"""
Run a folder of saved camera frames through the VLM gateway and report how
well the confidence threshold separates "target present" from "absent".

This is the script to run during the SHORT paid Vast.ai session: it needs
no ROS, no Gazebo and no robot, only the saved frames.

Label frames through their file names:

    pos_*.jpg   the target IS visible in the frame
    neg_*.jpg   the target is NOT visible (other objects, empty scene, ...)
    anything else is reported but not scored

(tools/preflight_check.py --collect DIR saves frames with these names.)

It uses the same prompt, parser and matching code as the robot, so the
result reflects what the robot would actually do.

Usage (from the repo root):

    python3 tools/vlm_batch_check.py --images frames/ --target "red box"
    python3 tools/vlm_batch_check.py --images frames/ --target "red box" \
        --gateway-url http://127.0.0.1:8000 --min-confidence 0.6
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

DEFAULT_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


# ---------------------------------------------------------------------------
# Scoring (pure, unit-tested)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FrameResult:
    name: str
    expected: bool | None        # True = pos_, False = neg_, None = unlabeled
    best_confidence: float | None  # best matching detection, None = no match
    other_objects: int = 0         # detections that did not match the target
    latency_sec: float = 0.0
    error: str | None = None


@dataclass(frozen=True)
class Confusion:
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int

    @property
    def precision(self) -> float | None:
        total = self.true_positive + self.false_positive
        return self.true_positive / total if total else None

    @property
    def recall(self) -> float | None:
        total = self.true_positive + self.false_negative
        return self.true_positive / total if total else None


def expected_from_name(name: str) -> bool | None:
    lowered = name.lower()

    if lowered.startswith("pos_"):
        return True

    if lowered.startswith("neg_"):
        return False

    return None


def is_accepted(result: FrameResult, threshold: float) -> bool:
    """Would the robot accept this frame's detection at ``threshold``?"""

    return (
        result.error is None
        and result.best_confidence is not None
        and result.best_confidence >= threshold
    )


def confusion(results: list[FrameResult], threshold: float) -> Confusion:
    tp = fp = fn = tn = 0

    for result in results:
        if result.expected is None or result.error is not None:
            continue

        accepted = is_accepted(result, threshold)

        if result.expected and accepted:
            tp += 1
        elif result.expected and not accepted:
            fn += 1
        elif not result.expected and accepted:
            fp += 1
        else:
            tn += 1

    return Confusion(tp, fp, fn, tn)


def lowest_safe_threshold(
    results: list[FrameResult],
    thresholds: list[float],
) -> float | None:
    """Lowest threshold that accepts no negative frame, if there is one."""

    for threshold in sorted(thresholds):
        if confusion(results, threshold).false_positive == 0:
            return threshold

    return None


# ---------------------------------------------------------------------------
# Running the VLM
# ---------------------------------------------------------------------------

def collect_images(paths: list[str]) -> list[Path]:
    images: list[Path] = []

    for raw in paths:
        path = Path(raw)

        if path.is_dir():
            images.extend(
                sorted(
                    p
                    for p in path.iterdir()
                    if p.suffix.lower() in IMAGE_SUFFIXES
                )
            )
        elif path.is_file():
            images.append(path)
        else:
            raise FileNotFoundError(f"No such file or folder: {path}")

    return images


def run_frames(
    images: list[Path],
    target: str,
    gateway_url: str,
    timeout: float,
    verbose: bool = True,
) -> list[FrameResult]:
    import cv2

    from ai_client.gateway_client import AIGatewayClient
    from ai_client.vlm_client import RemoteVLMClient
    from perception.object_search import (
        detection_matches,
        build_find_prompt,
        sanitize_target,
        target_tokens,
    )
    from perception.remote_vlm import RemoteVLMPerception

    perception = RemoteVLMPerception(
        RemoteVLMClient(AIGatewayClient(base_url=gateway_url, timeout=timeout))
    )

    clean = sanitize_target(target)
    tokens = target_tokens(clean)

    results: list[FrameResult] = []

    for path in images:
        expected = expected_from_name(path.name)

        image = cv2.imread(str(path))

        if image is None:
            results.append(
                FrameResult(path.name, expected, None, error="unreadable image")
            )
            continue

        height, width = image.shape[:2]
        prompt = build_find_prompt(clean, width, height)

        started = time.monotonic()

        try:
            perceived = perception.analyze(image_path=path, instruction=prompt)
        except Exception as exc:
            results.append(
                FrameResult(
                    path.name,
                    expected,
                    None,
                    latency_sec=time.monotonic() - started,
                    error=str(exc),
                )
            )
            continue

        latency = time.monotonic() - started

        best: float | None = None
        others = 0
        described = []

        for detected in perceived.objects:
            confidence = (
                detected.confidence if detected.confidence is not None else 0.0
            )

            matches = detected.bbox is not None and detection_matches(
                detected, tokens
            )

            described.append(
                f"{detected.label!r}={confidence:.2f}"
                + ("" if matches else " (no match)")
            )

            if matches:
                best = confidence if best is None else max(best, confidence)
            else:
                others += 1

        result = FrameResult(
            name=path.name,
            expected=expected,
            best_confidence=best,
            other_objects=others,
            latency_sec=latency,
        )

        results.append(result)

        if verbose:
            print(
                f"  {path.name:<28} "
                f"expected={_fmt_expected(expected):<8} "
                f"best={_fmt_conf(best):<5} "
                f"latency={latency:5.2f}s  "
                f"[{', '.join(described) or 'no objects'}]",
                flush=True,
            )

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt_conf(value: float | None) -> str:
    return "none" if value is None else f"{value:.2f}"


def _fmt_expected(value: bool | None) -> str:
    return {True: "present", False: "absent", None: "-"}[value]


def _fmt_ratio(value: float | None) -> str:
    return "  n/a" if value is None else f"{value:5.2f}"


def print_report(
    results: list[FrameResult],
    chosen_threshold: float,
) -> None:
    errors = [r for r in results if r.error is not None]
    scored = [r for r in results if r.expected is not None and r.error is None]

    print()
    print("=" * 66)
    print("SUMMARY")
    print("=" * 66)

    ok = [r for r in results if r.error is None]

    if ok:
        mean_latency = sum(r.latency_sec for r in ok) / len(ok)
        print(f"Frames: {len(results)}   mean VLM latency: {mean_latency:.2f} s")

    if errors:
        print(f"\n{len(errors)} frame(s) failed:")
        for r in errors:
            print(f"  {r.name}: {r.error}")

    if not scored:
        print(
            "\nNo labeled frames. Name frames pos_*.jpg / neg_*.jpg to get "
            "an accuracy table."
        )
        return

    thresholds = sorted(set(DEFAULT_THRESHOLDS) | {chosen_threshold})

    print(
        "\nthreshold   TP  FP  FN  TN   precision  recall"
        "   (robot threshold marked with *)"
    )

    for threshold in thresholds:
        c = confusion(results, threshold)

        marker = "*" if abs(threshold - chosen_threshold) < 1e-9 else " "

        print(
            f"  {threshold:4.2f}{marker}    "
            f"{c.true_positive:3d} {c.false_positive:3d} "
            f"{c.false_negative:3d} {c.true_negative:3d}   "
            f"{_fmt_ratio(c.precision)}      {_fmt_ratio(c.recall)}"
        )

    negatives = [r for r in scored if r.expected is False]

    print()

    if negatives:
        safe = lowest_safe_threshold(results, thresholds)

        if safe is None:
            print(
                "WARNING: even at the highest threshold tested, at least one "
                "'absent' frame is accepted. The robot would navigate to a "
                "hallucinated object. The prompt or confidence gate needs "
                "work (or add a colour check on the box)."
            )
        else:
            print(
                f"Lowest threshold with no false positives: {safe:.2f}"
            )

            if chosen_threshold < safe:
                print(
                    f"NOTE: the robot threshold ({chosen_threshold:.2f}) is "
                    "below that, so it can accept a wrong detection."
                )
    else:
        print(
            "No neg_* frames, so false positives could not be measured. "
            "Include frames where the target is absent."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score the VLM on saved frames (no robot needed).",
    )

    parser.add_argument(
        "--images",
        nargs="+",
        required=True,
        help="Image files or folders (pos_* / neg_* prefix = labels).",
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Object description, e.g. 'red box'.",
    )
    parser.add_argument("--gateway-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.60,
        help="The robot's acceptance threshold (default 0.60).",
    )
    parser.add_argument("--timeout", type=float, default=120.0)

    args = parser.parse_args(argv)

    try:
        images = collect_images(args.images)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2

    if not images:
        print("No images found.", file=sys.stderr)
        return 2

    from ai_client.gateway_client import AIGatewayClient

    try:
        health = AIGatewayClient(
            base_url=args.gateway_url,
            timeout=args.timeout,
        ).health()
    except Exception as exc:
        print(
            f"Gateway not reachable at {args.gateway_url}: {exc}\n"
            "Start the mock (python3 tools/mock_gateway.py) or open the "
            "SSH tunnel to the Vast instance.",
            file=sys.stderr,
        )
        return 2

    print(f"Gateway: {health}")
    print(f"Target : {args.target!r}   Frames: {len(images)}\n")

    results = run_frames(
        images,
        args.target,
        args.gateway_url,
        args.timeout,
    )

    print_report(results, args.min_confidence)

    return 1 if any(r.error for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
