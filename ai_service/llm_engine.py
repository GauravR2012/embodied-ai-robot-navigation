from __future__ import annotations

import time
from typing import Any

import ollama


MODEL_ID = "qwen3:8b"


class LLMEngine:
    """Ollama-backed LLM engine for robot command interpretation."""

    def __init__(self) -> None:
        self.load_time: float | None = None
        self._loaded = False

    def load(self) -> None:
        """Verify that the Ollama model is available."""

        if self._loaded:
            return

        print(f"Loading LLM: {MODEL_ID}")

        start = time.perf_counter()

        models = ollama.list()

        available_models = {
            model.model
            for model in models.models
        }

        if MODEL_ID not in available_models:
            raise RuntimeError(
                f"Ollama model '{MODEL_ID}' is not available. "
                f"Available models: {sorted(available_models)}"
            )

        self._loaded = True

        self.load_time = (
            time.perf_counter() - start
        )

        print(
            f"LLM ready in {self.load_time:.2f} s"
        )

    @property
    def loaded(self) -> bool:
        return self._loaded

    def generate(
        self,
        instruction: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Convert natural-language instruction into RobotCommand JSON."""

        if not self.loaded:
            raise RuntimeError(
                "LLM is not loaded."
            )

        if not instruction or not instruction.strip():
            raise ValueError(
                "Instruction cannot be empty."
            )

        context = context or {}

        context_text = ""

        if context:
            context_text = (
                "\nAdditional context:\n"
                f"{context}\n"
            )

        system_prompt = """
You are the command interpreter for a mobile robot.

Convert the user's instruction into EXACTLY ONE valid RobotCommand
JSON object.

Allowed actions:

1. navigate

{
  "action": "navigate",
  "target": {
    "type": "named_location",
    "value": "<location name>"
  }
}

OR, when the user refers to a physical object:

{
  "action": "navigate",
  "target": {
    "type": "object",
    "value": "<object description>"
  }
}

Examples:

"Go to the kitchen"

{
  "action": "navigate",
  "target": {
    "type": "named_location",
    "value": "kitchen"
  }
}

"Go to the red box"

{
  "action": "navigate",
  "target": {
    "type": "object",
    "value": "red box"
  }
}

"Find the blue box"

{
  "action": "navigate",
  "target": {
    "type": "object",
    "value": "blue box"
  }
}

2. move

{
  "action": "move",
  "direction": "forward|backward|left|right",
  "distance": <number>,
  "unit": "m"
}

3. rotate

{
  "action": "rotate",
  "direction": "left|right",
  "angle": <number>,
  "unit": "deg"
}

4. stop

{
  "action": "stop"
}

5. wait

{
  "action": "wait",
  "duration": <number>,
  "unit": "s"
}

6. sequence

{
  "action": "sequence",
  "commands": [
    <valid RobotCommand>,
    <valid RobotCommand>
  ]
}

STRICT RULES:

- Return ONLY valid JSON.
- Do NOT return Markdown.
- Do NOT explain your answer.
- Do NOT use a "command" field.
- Do NOT use a "steps" field.
- A sequence MUST use "commands".
- A navigation target MUST be an object.
- Never generate arbitrary coordinates.
- Never generate Python.
- Never generate ROS commands.
- Never generate shell commands.
- Never invent named locations.
- If the user refers to a physical object such as a box,
  chair, table, person, or other object, use:
  "type": "object".
- Preserve useful object descriptors such as color and type.
- Do not claim that an object exists; only express the user's
  requested target semantically.
""".strip()

        user_prompt = (
            f'User instruction:\n"{instruction}"\n'
            f"{context_text}\n"
            "Return ONLY the JSON object."
        )

        start = time.perf_counter()

        response = ollama.chat(
            model=MODEL_ID,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            options={
                "temperature": 0,
            },
        )

        elapsed = time.perf_counter() - start

        print(
            f"LLM inference completed in {elapsed:.2f} s"
        )

        content = response["message"]["content"]

        if not isinstance(content, str):
            raise RuntimeError(
                "Ollama returned an invalid response."
            )

        return content.strip()