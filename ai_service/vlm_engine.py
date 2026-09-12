import base64
import io
import time

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration


MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


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
                        "text": instruction,
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
