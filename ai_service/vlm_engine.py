from __future__ import annotations

import base64
import io
import time

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration


MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


PERCEPTION_SYSTEM_INSTRUCTION = """
You are the perception module of an autonomous mobile robot.

Analyze the provided camera image and return ONLY valid JSON.

The JSON schema is exactly:

{
  "scene_description": "string",
  "objects": [
    {
      "label": "string",
      "confidence": number,
      "bbox": {
        "x_min": number,
        "y_min": number,
        "x_max": number,
        "y_max": number
      },
      "attributes": {
        "key": "value"
      }
    }
  ]
}

Rules:

1. Return JSON only.
2. Do not use Markdown.
3. Do not use code fences.
4. Do not provide explanations before or after the JSON.
5. "scene_description" must briefly describe the visible scene.
6. "objects" must contain only visually identifiable objects relevant to
   robot navigation.
7. "label" must be concise and descriptive.
8. "confidence" must be a number between 0.0 and 1.0.
9. Include "bbox" when a reasonably accurate 2D bounding box can be
   identified.
10. Bounding-box coordinates must be pixel coordinates relative to the
    input image.
11. x_min must be smaller than x_max.
12. y_min must be smaller than y_max.
13. "attributes" may contain semantic information such as:
    {"role": "obstacle"}
    or
    {"role": "navigation_target"}.
14. Never invent objects that are not visible.
15. Never output robot coordinates.
16. Never output map coordinates.
17. Never output robot poses.
18. Never output ROS commands.
19. Never output Python code.
20. Never output shell commands.
21. Never output navigation actions.
22. If no relevant objects are visible, return an empty "objects" array.

Example:

{
  "scene_description": "Simulated indoor environment.",
  "objects": [
    {
      "label": "gray cylinder",
      "confidence": 0.91,
      "bbox": {
        "x_min": 210,
        "y_min": 140,
        "x_max": 280,
        "y_max": 310
      },
      "attributes": {
        "role": "obstacle"
      }
    },
    {
      "label": "green structure",
      "confidence": 0.87,
      "attributes": {
        "role": "navigation_target"
      }
    }
  ]
}
""".strip()


class VLMEngine:
    def __init__(self):
        self.model = None
        self.processor = None
        self.load_time = None

    def load(self) -> None:
        print(f"Loading VLM: {MODEL_ID}")

        start = time.perf_counter()

        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID
        )

        self.model = (
            Qwen2_5_VLForConditionalGeneration.from_pretrained(
                MODEL_ID,
                torch_dtype="auto",
                device_map="auto",
            )
        )

        self.model.eval()

        torch.cuda.synchronize()

        self.load_time = time.perf_counter() - start

        print(
            f"VLM loaded in {self.load_time:.2f} s"
        )

    @property
    def loaded(self) -> bool:
        return self.model is not None

    @staticmethod
    def decode_image(image_base64: str) -> Image.Image:
        """
        Decode a base64-encoded image into an RGB PIL image.
        """

        try:
            image_bytes = base64.b64decode(
                image_base64,
                validate=True,
            )
        except Exception as exc:
            raise ValueError(
                "Invalid base64 image."
            ) from exc

        try:
            image = Image.open(
                io.BytesIO(image_bytes)
            ).convert("RGB")
        except Exception as exc:
            raise ValueError(
                "Unable to decode image."
            ) from exc

        return image

    def analyze(
        self,
        image_base64: str,
        instruction: str,
    ) -> str:
        """
        Run visual perception using Qwen2.5-VL.

        The model is explicitly instructed to return the
        backend-independent perception JSON contract.
        """

        if not self.loaded:
            raise RuntimeError(
                "VLM is not loaded."
            )

        if not instruction or not instruction.strip():
            raise ValueError(
                "Instruction cannot be empty."
            )

        image = self.decode_image(
            image_base64
        )

        user_instruction = (
            f"{PERCEPTION_SYSTEM_INSTRUCTION}\n\n"
            f"Additional perception instruction:\n"
            f"{instruction.strip()}"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text",
                        "text": user_instruction,
                    },
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = (
            process_vision_info(messages)
        )

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
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
            output = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )

        torch.cuda.synchronize()

        generated = output[
            :,
            inputs["input_ids"].shape[1]:,
        ]

        response = self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        return response