from __future__ import annotations

from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


PERCEPTION_SYSTEM_INSTRUCTION = """
You are the visual perception module of a robot.

Analyze the provided image and return ONLY valid JSON.
Do NOT use Markdown.
Do NOT use ```json fences.
Do NOT add explanations before or after the JSON.

Use exactly this structure:

{
  "scene_description": "short description",
  "objects": [
    {
      "label": "object name",
      "confidence": 0.0,
      "bbox": {
        "x_min": 0,
        "y_min": 0,
        "x_max": 0,
        "y_max": 0
      },
      "attributes": {
        "role": "obstacle"
      }
    }
  ]
}

Rules:
- Return at most 5 objects.
- Only include clearly visible objects.
- confidence must be between 0.0 and 1.0.
- bbox coordinates are pixel coordinates in the input image.
- x_min < x_max.
- y_min < y_max.
- Keep scene_description short.
- Keep labels concise.
- Use attributes only when useful.
- Do not invent objects that are not visible.
- The response must end with a complete JSON object.
"""


class VLMEngine:
    """Qwen2.5-VL inference engine for robotic visual perception."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
    ) -> None:
        self.model_id = model_id

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor = AutoProcessor.from_pretrained(
            model_id,
        )

        if self.device == "cuda":
            self.model = (
                Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
            )
        else:
            self.model = (
                Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=torch.float32,
                ).to(self.device)
            )

        self.model.eval()

    def analyze(
        self,
        image: Image.Image,
        instruction: str,
    ) -> str:
        """
        Run VLM perception on an image.

        The returned string is intentionally kept as raw model output.
        Deterministic validation/parsing is performed by the perception
        layer on the client side.
        """

        if image is None:
            raise ValueError("image must not be None")

        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL.Image.Image")

        instruction = instruction.strip()

        prompt = (
            f"{PERCEPTION_SYSTEM_INSTRUCTION}\n\n"
            f"Additional perception instruction:\n{instruction}"
        )

        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.model.device)
            if hasattr(value, "to")
            else value
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=384,
                do_sample=False,
            )

        input_token_count = inputs["input_ids"].shape[1]

        generated_ids_trimmed = [
            output_ids[input_token_count:]
            for output_ids in generated_ids
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return output_text.strip()