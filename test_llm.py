#!/usr/bin/env python3

import json
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"


def memory():
    """
    Return current CUDA memory usage in GB:
        allocated, reserved
    """
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    return allocated, reserved


def extract_json(text: str) -> str:
    """
    Extract the first JSON object from the model response.

    The model is instructed to return JSON only, but this keeps the
    benchmark robust against accidental surrounding text.
    """
    text = text.strip()

    # Direct JSON response.
    if text.startswith("{") and text.endswith("}"):
        return text

    # Remove possible Markdown code fences.
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    text = text.strip()

    if text.startswith("{") and text.endswith("}"):
        return text

    # Find the first JSON object.
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in LLM response.")

    return text[start : end + 1]


def validate_robot_command(command_data):
    """
    Validate the generated command against the project's actual
    RobotCommand Pydantic schema.

    This test is intended to run from the repository root.
    """
    try:
        from command_schema import parse_command
    except ImportError as exc:
        raise RuntimeError(
            "Could not import command_schema.py. "
            "Run this script from the repository root."
        ) from exc

    return parse_command(command_data)


def main():

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this test.")

    print("=" * 70)
    print("QWEN 2.5 3B LLM TEST")
    print("=" * 70)

    print("\nGPU:", torch.cuda.get_device_name(0))

    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    print(f"Total VRAM: {total:.2f} GB")

    # ------------------------------------------------------------------
    # Quantization
    # ------------------------------------------------------------------

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    # ------------------------------------------------------------------
    # Initial memory
    # ------------------------------------------------------------------

    before = memory()

    print("\nVRAM before LLM load:")
    print(f"  Allocated: {before[0]:.2f} GB")
    print(f"  Reserved:  {before[1]:.2f} GB")

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    print("Loading 4-bit LLM...")

    start = time.perf_counter()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        device_map="auto",
    )

    model.eval()

    torch.cuda.synchronize()

    load_time = time.perf_counter() - start

    after_load = memory()

    print(f"\nModel loaded in {load_time:.2f} s")

    print("\nVRAM after LLM load:")
    print(f"  Allocated: {after_load[0]:.2f} GB")
    print(f"  Reserved:  {after_load[1]:.2f} GB")
    print(f"  Increase:  {after_load[0] - before[0]:.2f} GB")

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    prompt = """
You are the command interpreter for a mobile robot.

Convert the user's instruction into EXACTLY ONE valid RobotCommand
JSON object.

Allowed actions and their exact schemas:

1. navigate

{
  "action": "navigate",
  "target": {
    "type": "named_location",
    "value": "<location name>"
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
- A sequence MUST use the field "commands".
- A navigate command MUST have a "target" object.
- A navigate target MUST contain "type": "named_location".
- A navigate target MUST contain "value".
- A wait command MUST contain "unit": "s".
- A move command MUST contain "unit": "m".
- A rotate command MUST contain "unit": "deg".
- Never generate Python.
- Never generate ROS commands.
- Never generate shell commands.
- Never generate arbitrary coordinates.
- Never invent locations.
- If multiple actions are requested, use "sequence".

Example:

User instruction:
"Go to Station 1 and wait there for 5 seconds."

Correct JSON:
{
  "action": "sequence",
  "commands": [
    {
      "action": "navigate",
      "target": {
        "type": "named_location",
        "value": "Station 1"
      }
    },
    {
      "action": "wait",
      "duration": 5,
      "unit": "s"
    }
  ]
}

User instruction:
"Go to Station 1."

Correct JSON:
{
  "action": "navigate",
  "target": {
    "type": "named_location",
    "value": "Station 1"
  }
}

User instruction:
"Wait for 5 seconds."

Correct JSON:
{
  "action": "wait",
  "duration": 5,
  "unit": "s"
}

Now convert this instruction:

"Go to Station 1 and wait there for 5 seconds."

Return ONLY the JSON object.
""".strip()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a robot command interpreter. "
                "Return only valid JSON. "
                "Never generate Python, ROS commands, shell commands, "
                "or arbitrary coordinates."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    # ------------------------------------------------------------------
    # Chat template
    # ------------------------------------------------------------------

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    ).to(model.device)

    # ------------------------------------------------------------------
    # Warm-up
    # ------------------------------------------------------------------

    print("\nRunning warm-up...")

    with torch.inference_mode():
        model.generate(
            **inputs,
            max_new_tokens=128,
        )

    torch.cuda.synchronize()

    print("Warm-up complete.")

    # Clear temporary generation allocations before timed run.
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # Timed inference
    # ------------------------------------------------------------------

    print("\nRunning timed inference...")

    start = time.perf_counter()

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
        )

    torch.cuda.synchronize()

    inference_time = time.perf_counter() - start

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    generated = output[0][inputs["input_ids"].shape[1] :]

    response = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()

    final_memory = memory()

    # ------------------------------------------------------------------
    # Parse JSON
    # ------------------------------------------------------------------

    parsed_command = None
    json_error = None

    try:
        json_text = extract_json(response)
        parsed_command = json.loads(json_text)
    except Exception as exc:
        json_error = str(exc)

    # ------------------------------------------------------------------
    # Validate RobotCommand
    # ------------------------------------------------------------------

    validated_command = None
    validation_error = None

    if parsed_command is not None:

        try:
            validated_command = validate_robot_command(parsed_command)
        except Exception as exc:
            validation_error = str(exc)

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("LLM TEST RESULT")
    print("=" * 70)

    print("Model:", MODEL_ID)
    print("GPU:", torch.cuda.get_device_name(0))
    print(f"Total VRAM:      {total:.2f} GB")
    print(f"Load time:       {load_time:.2f} s")
    print(f"Inference time:  {inference_time:.2f} s")
    print(f"VRAM allocated:  {final_memory[0]:.2f} GB")
    print(f"VRAM reserved:   {final_memory[1]:.2f} GB")

    # ------------------------------------------------------------------
    # Raw response
    # ------------------------------------------------------------------

    print("\nLLM RESPONSE")
    print("-" * 70)
    print(response)
    print("-" * 70)

    # ------------------------------------------------------------------
    # JSON parsing
    # ------------------------------------------------------------------

    print("\nJSON PARSING")
    print("-" * 70)

    if parsed_command is not None:
        print("PASS")
        print(json.dumps(parsed_command, indent=2))
    else:
        print("FAIL")
        print("Error:", json_error)

    # ------------------------------------------------------------------
    # RobotCommand validation
    # ------------------------------------------------------------------

    print("\nROBOT COMMAND SCHEMA VALIDATION")
    print("-" * 70)

    if validated_command is not None:

        print("PASS")
        print("\nValidated command:")

        try:
            print(validated_command.model_dump_json(indent=2))
        except AttributeError:
            print(validated_command)

    else:

        print("FAIL")

        if validation_error:
            print("Error:")
            print(validation_error)

    # ------------------------------------------------------------------
    # Final status
    # ------------------------------------------------------------------

    print("\n" + "=" * 70)

    if validated_command is not None:
        print("OVERALL RESULT: PASS")
        print("=" * 70)
        print("\nThe LLM generated a valid RobotCommand.")
    else:
        print("OVERALL RESULT: FAIL")
        print("=" * 70)
        print("\nThe LLM output did not satisfy the RobotCommand contract.")

    print("\nTEST COMPLETE")


if __name__ == "__main__":
    main()
