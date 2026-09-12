import time

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"


class LLMEngine:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.load_time = None

    def load(self) -> None:
        print(f"Loading LLM: {MODEL_ID}")

        start = time.perf_counter()

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID
        )

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=quant_config,
            device_map="auto",
        )

        self.model.eval()

        torch.cuda.synchronize()

        self.load_time = time.perf_counter() - start

        print(
            f"LLM loaded in {self.load_time:.2f} s"
        )

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def generate(
        self,
        instruction: str,
        context: dict | None = None,
    ) -> str:

        if not self.loaded:
            raise RuntimeError("LLM is not loaded.")

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

        prompt = f"""
You are the command interpreter for a mobile robot.

Convert the user's instruction into EXACTLY ONE valid RobotCommand
JSON object.

Allowed actions:

1. navigate

{{
  "action": "navigate",
  "target": {{
    "type": "named_location",
    "value": "<location name>"
  }}
}}

2. move

{{
  "action": "move",
  "direction": "forward|backward|left|right",
  "distance": <number>,
  "unit": "m"
}}

3. rotate

{{
  "action": "rotate",
  "direction": "left|right",
  "angle": <number>,
  "unit": "deg"
}}

4. stop

{{
  "action": "stop"
}}

5. wait

{{
  "action": "wait",
  "duration": <number>,
  "unit": "s"
}}

6. sequence

{{
  "action": "sequence",
  "commands": [
    <valid RobotCommand>,
    <valid RobotCommand>
  ]
}}

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

User instruction:

"{instruction}"
{context_text}

Return ONLY the JSON object.
""".strip()

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a robot command interpreter. "
                    "Return only valid JSON."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.inference_mode():

            output = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
            )

        torch.cuda.synchronize()

        generated = output[0][
            inputs["input_ids"].shape[1]:
        ]

        response = self.tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ).strip()

        return response
