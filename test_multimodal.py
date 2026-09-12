#!/usr/bin/env python3

import json
import time

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)


LLM_ID = "Qwen/Qwen2.5-3B-Instruct"
VLM_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


def gpu_memory_gb():
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    return allocated, reserved


def print_memory(label):
    allocated, reserved = gpu_memory_gb()

    print(f"\n{label}")
    print("-" * 70)
    print(f"Allocated: {allocated:.2f} GB")
    print(f"Reserved:  {reserved:.2f} GB")

    return allocated, reserved


def main():

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")

    print("=" * 70)
    print("MULTIMODAL LLM + VLM GPU RESIDENCY TEST")
    print("=" * 70)

    gpu_name = torch.cuda.get_device_name(0)
    total_vram = (
        torch.cuda.get_device_properties(0).total_memory
        / (1024 ** 3)
    )

    print(f"GPU:        {gpu_name}")
    print(f"Total VRAM: {total_vram:.2f} GB")

    # ================================================================
    # INITIAL MEMORY
    # ================================================================

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    print_memory("VRAM before loading models")

    # ================================================================
    # LOAD VLM
    # ================================================================

    print("\n" + "=" * 70)
    print("LOADING VLM")
    print("=" * 70)

    print(f"Model: {VLM_ID}")

    processor_start = time.perf_counter()

    processor = AutoProcessor.from_pretrained(VLM_ID)

    processor_time = time.perf_counter() - processor_start

    print(f"Processor loaded in {processor_time:.2f} s")

    model_start = time.perf_counter()

    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VLM_ID,
        torch_dtype="auto",
        device_map="auto",
    )

    vlm.eval()

    torch.cuda.synchronize()

    vlm_load_time = time.perf_counter() - model_start

    print(f"VLM model loaded in {vlm_load_time:.2f} s")

    vlm_memory = print_memory("VRAM after VLM load")

    # ================================================================
    # LOAD LLM
    # ================================================================

    print("\n" + "=" * 70)
    print("LOADING LLM")
    print("=" * 70)

    print(f"Model: {LLM_ID}")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(LLM_ID)

    llm_start = time.perf_counter()

    llm = AutoModelForCausalLM.from_pretrained(
        LLM_ID,
        quantization_config=quant_config,
        device_map="auto",
    )

    llm.eval()

    torch.cuda.synchronize()

    llm_load_time = time.perf_counter() - llm_start

    print(f"LLM model loaded in {llm_load_time:.2f} s")

    combined_memory = print_memory("VRAM after VLM + LLM load")

    # ================================================================
    # MEMORY SUMMARY
    # ================================================================

    print("\n" + "=" * 70)
    print("MODEL RESIDENCY SUMMARY")
    print("=" * 70)

    print(f"VLM allocated:       {vlm_memory[0]:.2f} GB")
    print(f"VLM reserved:        {vlm_memory[1]:.2f} GB")
    print()
    print(f"Both allocated:      {combined_memory[0]:.2f} GB")
    print(f"Both reserved:       {combined_memory[1]:.2f} GB")
    print()
    print(
        f"Estimated free VRAM: "
        f"{total_vram - combined_memory[1]:.2f} GB"
    )

    # ================================================================
    # LLM TEST INPUT
    # ================================================================

    llm_prompt = """
You are the command interpreter for a mobile robot.

Convert the user's instruction into EXACTLY ONE valid RobotCommand
JSON object.

Allowed actions:

- navigate
- move
- rotate
- stop
- wait
- sequence

For navigation:

{
  "action": "navigate",
  "target": {
    "type": "named_location",
    "value": "<location name>"
  }
}

For waiting:

{
  "action": "wait",
  "duration": <number>,
  "unit": "s"
}

For a sequence:

{
  "action": "sequence",
  "commands": [
    <valid RobotCommand>,
    <valid RobotCommand>
  ]
}

STRICT RULES:

- Return ONLY valid JSON.
- Do NOT use a "command" field.
- Do NOT use a "steps" field.
- A sequence MUST use "commands".
- A navigation target MUST be an object.
- Never generate arbitrary coordinates.
- Never generate Python.
- Never generate ROS commands.
- Never generate shell commands.

User instruction:

"Go to Station 1 and wait there for 5 seconds."

Return ONLY the JSON object.
""".strip()

    llm_messages = [
        {
            "role": "system",
            "content": (
                "You are a robot command interpreter. "
                "Return only valid JSON."
            ),
        },
        {
            "role": "user",
            "content": llm_prompt,
        },
    ]

    llm_text = tokenizer.apply_chat_template(
        llm_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    llm_inputs = tokenizer(
        llm_text,
        return_tensors="pt",
    ).to(llm.device)

    # ================================================================
    # VLM TEST INPUT
    # ================================================================

    # Create a small synthetic image without requiring another file.
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (640, 480), "white")

    draw = ImageDraw.Draw(image)

    draw.rectangle(
        [100, 150, 220, 270],
        fill="red",
    )

    draw.rectangle(
        [400, 150, 520, 270],
        fill="green",
    )

    draw.text(
        [20, 20],
        "Synthetic robot perception test",
        fill="black",
    )

    image_path = "/tmp/multimodal_test_image.png"

    image.save(image_path)

    vlm_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {
                    "type": "text",
                    "text": (
                        "Describe the objects in this image. "
                        "Identify obstacles and possible navigation targets."
                    ),
                },
            ],
        }
    ]

    vlm_text = processor.apply_chat_template(
        vlm_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    from qwen_vl_utils import process_vision_info

    image_inputs, video_inputs = process_vision_info(
        vlm_messages
    )

    vlm_inputs = processor(
        text=[vlm_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    vlm_inputs = {
        key: value.to(vlm.device)
        if hasattr(value, "to")
        else value
        for key, value in vlm_inputs.items()
    }

    # ================================================================
    # WARM-UP LLM
    # ================================================================

    print("\n" + "=" * 70)
    print("WARMING UP LLM")
    print("=" * 70)

    with torch.inference_mode():

        llm.generate(
            **llm_inputs,
            max_new_tokens=128,
            do_sample=False,
        )

    torch.cuda.synchronize()

    print("LLM warm-up complete.")

    # ================================================================
    # WARM-UP VLM
    # ================================================================

    print("\n" + "=" * 70)
    print("WARMING UP VLM")
    print("=" * 70)

    with torch.inference_mode():

        vlm.generate(
            **vlm_inputs,
            max_new_tokens=128,
        )

    torch.cuda.synchronize()

    print("VLM warm-up complete.")

    # ================================================================
    # MEMORY AFTER WARM-UP
    # ================================================================

    warm_memory = print_memory(
        "VRAM after both model warm-ups"
    )

    # ================================================================
    # TIMED LLM INFERENCE
    # ================================================================

    print("\n" + "=" * 70)
    print("TIMED LLM INFERENCE")
    print("=" * 70)

    torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():

        llm_output = llm.generate(
            **llm_inputs,
            max_new_tokens=128,
            do_sample=False,
        )

    torch.cuda.synchronize()

    llm_inference_time = time.perf_counter() - start

    llm_generated = llm_output[0][
        llm_inputs["input_ids"].shape[1]:
    ]

    llm_response = tokenizer.decode(
        llm_generated,
        skip_special_tokens=True,
    ).strip()

    llm_after = print_memory(
        "VRAM after LLM inference"
    )

    print(f"LLM inference time: {llm_inference_time:.2f} s")

    print("\nLLM response:")
    print("-" * 70)
    print(llm_response)
    print("-" * 70)

    # ================================================================
    # TIMED VLM INFERENCE
    # ================================================================

    print("\n" + "=" * 70)
    print("TIMED VLM INFERENCE")
    print("=" * 70)

    torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():

        vlm_output = vlm.generate(
            **vlm_inputs,
            max_new_tokens=128,
        )

    torch.cuda.synchronize()

    vlm_inference_time = time.perf_counter() - start

    vlm_generated = vlm_output[
        :,
        vlm_inputs["input_ids"].shape[1]:,
    ]

    vlm_response = processor.batch_decode(
        vlm_generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    vlm_after = print_memory(
        "VRAM after VLM inference"
    )

    print(f"VLM inference time: {vlm_inference_time:.2f} s")

    print("\nVLM response:")
    print("-" * 70)
    print(vlm_response)
    print("-" * 70)

    # ================================================================
    # REVERSE ORDER TEST
    # ================================================================

    print("\n" + "=" * 70)
    print("REVERSE ORDER RESIDENCY TEST")
    print("=" * 70)

    print("\nRunning VLM again...")

    torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():

        vlm.generate(
            **vlm_inputs,
            max_new_tokens=128,
        )

    torch.cuda.synchronize()

    vlm_second_time = time.perf_counter() - start

    print(
        f"Second VLM inference: "
        f"{vlm_second_time:.2f} s"
    )

    print_memory(
        "VRAM after second VLM inference"
    )

    print("\nRunning LLM again...")

    torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():

        llm.generate(
            **llm_inputs,
            max_new_tokens=128,
            do_sample=False,
        )

    torch.cuda.synchronize()

    llm_second_time = time.perf_counter() - start

    print(
        f"Second LLM inference: "
        f"{llm_second_time:.2f} s"
    )

    final_memory = print_memory(
        "Final VRAM after alternating inference"
    )

    # ================================================================
    # FINAL RESULT
    # ================================================================

    print("\n" + "=" * 70)
    print("MULTIMODAL GPU TEST RESULT")
    print("=" * 70)

    print(f"GPU: {gpu_name}")
    print(f"Total VRAM: {total_vram:.2f} GB")

    print()
    print(f"VLM load time: {vlm_load_time:.2f} s")
    print(f"LLM load time: {llm_load_time:.2f} s")

    print()
    print(f"LLM inference: {llm_inference_time:.2f} s")
    print(f"VLM inference: {vlm_inference_time:.2f} s")

    print()
    print(
        f"Peak observed reserved memory: "
        f"{max(
            vlm_memory[1],
            combined_memory[1],
            warm_memory[1],
            llm_after[1],
            vlm_after[1],
            final_memory[1],
        ):.2f} GB"
    )

    print(
        f"Final reserved memory: "
        f"{final_memory[1]:.2f} GB"
    )

    print()

    if final_memory[1] < total_vram:
        print("MODEL RESIDENCY: PASS")
    else:
        print("MODEL RESIDENCY: FAIL")

    print()
    print("LLM + VLM alternating inference completed.")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
