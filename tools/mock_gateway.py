#!/usr/bin/env python3
"""
Local stand-in for the Vast.ai AI gateway (no GPU, no model downloads).

It speaks exactly the same HTTP contract as ai_service/server.py:

    GET  /health
    POST /v1/chat          {"instruction": str, "context": {...}}
    POST /v1/perception    {"instruction": str, "image_base64": str}

so the Gradio UI, the planner, the executor and Nav2 can be exercised
end to end on your own machine before you pay for a GPU instance.

  * "LLM"  : a small rule-based parser. It understands navigate / rotate /
             wait / stop commands and emits the same RobotCommand JSON
             that the real Qwen model is prompted to produce.
  * "VLM"  : a colour detector. If the target contains a colour word
             ("red box", "the blue box") it finds blobs of that colour in
             the image and answers in the same JSON format as Qwen2.5-VL.
             Simulation boxes are plain solid colours, so this is enough
             to test depth, TF, goal generation, spinning and Nav2.

What this CANNOT tell you: whether the real Qwen models follow the
prompts, how good their bounding boxes are, or how trustworthy their
confidence is. Those need the real GPU (see tools/vlm_batch_check.py).

Only needs the standard library + numpy + OpenCV (+ PyYAML, optional).

Usage (from the repo root):

    python3 tools/mock_gateway.py
    python3 tools/mock_gateway.py --port 8000 --force-confidence 0.3
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import cv2
import numpy as np


MODEL_LLM = "mock-rule-based-llm"
MODEL_VLM = "mock-color-vlm"

DEFAULT_LOCATIONS_FILE = "config/test_locations.yaml"


# ---------------------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s.\-°]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^(?:please|kindly|hey robot|robot)\s+", "", text)

    return text


def load_location_names(path: str | Path | None) -> dict[str, str]:
    """
    Map normalised location name -> canonical name from the locations YAML.

    Falls back to the single test location if the file or PyYAML is
    unavailable.
    """

    fallback = {"executor test": "Executor Test"}

    if not path:
        return fallback

    file = Path(path)

    if not file.is_file():
        return fallback

    try:
        import yaml

        data = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
        names = (data.get("locations") or {}).keys()
    except Exception:
        return fallback

    result = {_normalize(str(name)): str(name) for name in names}

    return result or fallback


def interpret(text: str, locations: dict[str, str]) -> dict[str, Any]:
    """
    Convert an instruction into a RobotCommand dictionary.

    Raises:
        ValueError: if the instruction is not understood.
    """

    t = _normalize(text)

    if not t:
        raise ValueError("Instruction cannot be empty.")

    if re.fullmatch(r"(?:stop|halt|cancel|freeze)(?: now)?", t):
        return {"action": "stop"}

    m = re.fullmatch(
        r"wait(?: for)? (\d+(?:\.\d+)?) ?(?:seconds?|secs?|s)",
        t,
    )

    if m:
        return {
            "action": "wait",
            "duration": float(m.group(1)),
            "unit": "s",
        }

    if re.fullmatch(r"turn around", t):
        return {
            "action": "rotate",
            "direction": "left",
            "angle": 180.0,
            "unit": "deg",
        }

    m = re.fullmatch(
        r"(?:rotate|turn|spin)(?: to the| to)? (left|right)"
        r"(?: by)?(?: (\d+(?:\.\d+)?) ?(?:degrees?|deg|°)?)?",
        t,
    )

    if m:
        return {
            "action": "rotate",
            "direction": m.group(1),
            "angle": float(m.group(2)) if m.group(2) else 90.0,
            "unit": "deg",
        }

    m = re.fullmatch(
        r"(?:go|navigate|drive|move|head|come|travel|walk)"
        r"(?: over)? to (?:the |a |an )?(.+)",
        t,
    )

    if m:
        place = m.group(1).strip()

        if place in locations:
            return {
                "action": "navigate",
                "target": {
                    "type": "named_location",
                    "value": locations[place],
                },
            }

        return {
            "action": "navigate",
            "target": {
                "type": "object",
                "value": place,
            },
        }

    raise ValueError(
        f"Mock LLM cannot interpret: {text!r}. Try 'go to the red box', "
        "'go to executor test', 'turn left 90 degrees', 'wait 3 seconds' "
        "or 'stop'."
    )


# ---------------------------------------------------------------------------
# Mock VLM (colour detector)
# ---------------------------------------------------------------------------

# OpenCV HSV: H in [0, 179], S and V in [0, 255].
COLOR_RANGES: dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]] = {
    "red": [((0, 120, 50), (10, 255, 255)), ((170, 120, 50), (179, 255, 255))],
    "orange": [((10, 120, 50), (20, 255, 255))],
    "yellow": [((20, 100, 50), (35, 255, 255))],
    "green": [((35, 80, 40), (85, 255, 255))],
    "blue": [((100, 120, 40), (130, 255, 255))],
    "purple": [((130, 80, 40), (160, 255, 255))],
}

TARGET_PATTERN = re.compile(r"Target:\s*\n\s*(.+?)\s*\n", re.IGNORECASE)


def extract_target(instruction: str) -> str:
    """Pull the target description out of the perception prompt."""

    match = TARGET_PATTERN.search(instruction)

    if match:
        return match.group(1).strip()

    return instruction.strip()


def detect_color_blobs(
    image_bgr: np.ndarray,
    target: str,
    min_area_fraction: float = 0.0004,
    max_objects: int = 5,
) -> tuple[list[dict[str, Any]], str]:
    """
    Find blobs of the colour named in ``target``.

    Returns (objects, scene_description) in the Qwen output format.
    """

    words = set(re.findall(r"[a-z]+", target.lower()))
    colors = [name for name in COLOR_RANGES if name in words]

    if len(colors) != 1:
        return [], (
            "mock VLM understands exactly one colour word "
            f"({', '.join(COLOR_RANGES)}); got {colors or 'none'}"
        )

    color = colors[0]

    height, width = image_bgr.shape[:2]

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    mask = np.zeros((height, width), dtype=np.uint8)

    for low, high in COLOR_RANGES[color]:
        mask |= cv2.inRange(hsv, np.array(low), np.array(high))

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8),
    )

    contours, _hierarchy = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    min_area = max(100.0, min_area_fraction * width * height)

    blobs = []

    for contour in contours:
        area = float(cv2.contourArea(contour))

        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        fill = area / float(max(w * h, 1))
        size_score = min(1.0, area / (0.01 * width * height))

        confidence = 0.5 + 0.25 * size_score + 0.2 * fill

        blobs.append((area, x, y, w, h, min(0.95, confidence)))

    blobs.sort(reverse=True)

    objects = [
        {
            "label": target,
            "confidence": round(conf, 2),
            "bbox": {
                "x_min": int(x),
                "y_min": int(y),
                "x_max": int(x + w),
                "y_max": int(y + h),
            },
            "attributes": {"color": color},
        }
        for _area, x, y, w, h, conf in blobs[:max_objects]
    ]

    if objects:
        scene = f"{len(objects)} {color} region(s) found by colour detection"
    else:
        scene = f"no {color} regions found"

    return objects, scene


def decode_image(image_base64: str) -> np.ndarray:
    raw = base64.b64decode(image_base64, validate=False)

    image = cv2.imdecode(
        np.frombuffer(raw, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError("Could not decode the base64 image.")

    return image


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

class GatewayConfig:
    def __init__(
        self,
        locations: dict[str, str],
        force_confidence: float | None,
        latency_sec: float,
        debug_dir: Path | None,
    ) -> None:
        self.locations = locations
        self.force_confidence = force_confidence
        self.latency_sec = latency_sec
        self.debug_dir = debug_dir


class Handler(BaseHTTPRequestHandler):
    server_version = "MockGateway/0.1"

    # We print our own, more useful log lines.
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass

    @property
    def cfg(self) -> GatewayConfig:
        return self.server.cfg  # type: ignore[attr-defined]

    # -- helpers ---------------------------------------------------------

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")

        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length else b""

        data = json.loads(raw.decode("utf-8") or "{}")

        if not isinstance(data, dict):
            raise ValueError("Request body must be a JSON object.")

        return data

    # -- routes ----------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") == "/health":
            self._send(
                200,
                {
                    "status": "ok",
                    "llm_loaded": True,
                    "vlm_loaded": True,
                    "gpu": "MOCK (no GPU)",
                },
            )
            return

        self._send(404, {"detail": "Not Found"})

    def do_POST(self) -> None:  # noqa: N802
        route = self.path.rstrip("/")

        try:
            data = self._read_json()

            if self.cfg.latency_sec > 0:
                time.sleep(self.cfg.latency_sec)

            if route == "/v1/chat":
                self._chat(data)
            elif route == "/v1/perception":
                self._perception(data)
            else:
                self._send(404, {"detail": "Not Found"})

        except Exception as exc:
            print(f"[error] {route}: {exc}", flush=True)
            self._send(500, {"detail": str(exc)})

    def _chat(self, data: dict[str, Any]) -> None:
        instruction = data.get("instruction")

        if not isinstance(instruction, str):
            raise ValueError("'instruction' must be a string.")

        command = interpret(instruction, self.cfg.locations)

        print(f"[llm] {instruction!r} -> {json.dumps(command)}", flush=True)

        self._send(
            200,
            {
                "response": json.dumps(command),
                "model": MODEL_LLM,
            },
        )

    def _perception(self, data: dict[str, Any]) -> None:
        instruction = data.get("instruction")
        image_base64 = data.get("image_base64")

        if not isinstance(instruction, str):
            raise ValueError("'instruction' must be a string.")

        if not isinstance(image_base64, str) or not image_base64:
            raise ValueError("'image_base64' must be a non-empty string.")

        image = decode_image(image_base64)
        target = extract_target(instruction)

        objects, scene = detect_color_blobs(image, target)

        if self.cfg.force_confidence is not None:
            for obj in objects:
                obj["confidence"] = self.cfg.force_confidence

        result = {
            "scene_description": scene,
            "objects": objects,
        }

        summary = ", ".join(
            f"conf={o['confidence']:.2f} bbox="
            f"({o['bbox']['x_min']},{o['bbox']['y_min']},"
            f"{o['bbox']['x_max']},{o['bbox']['y_max']})"
            for o in objects
        ) or "nothing found"

        print(
            f"[vlm] target={target!r} image={image.shape[1]}x"
            f"{image.shape[0]} -> {summary}",
            flush=True,
        )

        self._save_debug(image, target, objects)

        self._send(
            200,
            {
                "response": json.dumps(result),
                "model": MODEL_VLM,
            },
        )

    def _save_debug(
        self,
        image: np.ndarray,
        target: str,
        objects: list[dict[str, Any]],
    ) -> None:
        directory = self.cfg.debug_dir

        if directory is None:
            return

        try:
            directory.mkdir(parents=True, exist_ok=True)

            annotated = image.copy()

            for obj in objects:
                b = obj["bbox"]

                cv2.rectangle(
                    annotated,
                    (b["x_min"], b["y_min"]),
                    (b["x_max"], b["y_max"]),
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    annotated,
                    f"{target} {obj['confidence']:.2f}",
                    (b["x_min"], max(b["y_min"] - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

            name = f"mock_vlm_{int(time.time() * 1000)}.jpg"
            cv2.imwrite(str(directory / name), annotated)

        except Exception as exc:  # debugging aid only
            print(f"[warn] could not save debug image: {exc}", flush=True)


def make_server(
    host: str,
    port: int,
    config: GatewayConfig,
) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), Handler)
    server.cfg = config  # type: ignore[attr-defined]
    server.daemon_threads = True

    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Local mock of the Vast.ai AI gateway (no GPU).",
    )

    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Same port the SSH tunnel uses. Stop the mock before "
        "opening the tunnel.",
    )
    parser.add_argument(
        "--locations",
        default=DEFAULT_LOCATIONS_FILE,
        help="Location YAML used to tell named locations from objects.",
    )
    parser.add_argument(
        "--force-confidence",
        type=float,
        default=None,
        help="Report this confidence for every detection "
        "(e.g. 0.3 to test the low-confidence path).",
    )
    parser.add_argument(
        "--latency",
        type=float,
        default=0.0,
        help="Artificial delay per request in seconds (e.g. 3).",
    )
    parser.add_argument(
        "--debug-dir",
        default="/tmp/mock_gateway_debug",
        help="Annotated images of every VLM request are saved here. "
        "Use '' to disable.",
    )

    args = parser.parse_args(argv)

    config = GatewayConfig(
        locations=load_location_names(args.locations),
        force_confidence=args.force_confidence,
        latency_sec=args.latency,
        debug_dir=Path(args.debug_dir) if args.debug_dir else None,
    )

    try:
        server = make_server(args.host, args.port, config)
    except OSError as exc:
        print(
            f"Cannot listen on {args.host}:{args.port}: {exc}\n"
            "Is the SSH tunnel (or another mock) already using the port?",
            file=sys.stderr,
        )
        return 1

    print("=" * 66)
    print("MOCK AI GATEWAY (no GPU) - not the real Qwen models")
    print("=" * 66)
    print(f"Listening on http://{args.host}:{args.port}")
    print(f"Known named locations: {sorted(config.locations.values())}")
    print(f"Colours the mock VLM understands: {', '.join(COLOR_RANGES)}")

    if config.force_confidence is not None:
        print(f"Forcing detection confidence to {config.force_confidence}")

    if config.debug_dir is not None:
        print(f"Debug images: {config.debug_dir}")

    print("Ctrl+C to stop.\n", flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping mock gateway.")
    finally:
        server.server_close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
