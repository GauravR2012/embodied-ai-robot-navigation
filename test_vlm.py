from __future__ import annotations

import time
import gc

import torch
from PIL import Image, ImageDraw
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)


MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


def gpu_memory_gb() -> tuple[float, float]:
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    return allocated, reserved


def create_test_image() -> Image.Image:
    """
    Create a simple synthetic robot-environment image.

    This avoids depending on an external image download.
    """
    image = Image.new("RGB", (640, 480), "white")
    draw = ImageDraw.Draw(image)

    # Floor
    draw.rectangle((0, 300, 640, 480), fill="lightgray")

    # Red obstacle
    draw.rectangle((220, 220, 330, 330), fill="red")

    # Green target
    draw.rectangle((470, 230, 560, 320), fill="green")

    # Blue object
    draw.ellipse((70, 220, 150, 300), fill="blue")

    # Labels
    draw.text((235, 340), "RED OBSTACLE", fill="black")
    draw.text((465, 330), "GREEN TARGET", fill="black")
    draw.text((75, 310), "BLUE OBJECT", fill="black")

    return image


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print("=" * 70)
    print("Qwen2.5-VL-7B-Instruct — VLM Test")
    print("=" * 70)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")

    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"Total VRAM: {total_vram:.2f} GB")

    torch.cuda.empty_cache()
    gc.collect()

    before_allocated, before_reserved = gpu_memory_gb()

    print()
    print(f"VRAM before model load:")
    print(f"  Allocated: {before_allocated:.2f} GB")
    print(f"  Reserved:  {before_reserved:.2f} GB")

    # ------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------

    print()
    print("Loading processor...")

    processor_start = time.perf_counter()

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    processor_time = time.perf_counter() - processor_start

    print(f"Processor loaded in {processor_time:.2f} s")

    print()
    print("Loading VLM model...")

    model_start = time.perf_counter()

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map="auto",
    )

    model.eval()

    # Synchronize before measuring memory
    torch.cuda.synchronize()

    model_load_time = time.perf_counter() - model_start

    allocated, reserved = gpu_memory_gb()

    print(f"Model loaded in {model_load_time:.2f} s")

    print()
    print("VRAM after model load:")
    print(f"  Allocated: {allocated:.2f} GB")
    print(f"  Reserved:  {reserved:.2f} GB")
    print(f"  Model VRAM increase: {allocated - before_allocated:.2f} GB")

    # ------------------------------------------------------------
    # Create test image
    # ------------------------------------------------------------

    image = create_test_image()

    image_path = "vlm_test_image.png"
    image.save(image_path)

    print()
    print(f"Test image saved to: {image_path}")

    # ------------------------------------------------------------
    # Prepare prompt
    # ------------------------------------------------------------

    prompt = """
You are a perception system for a mobile robot.

Analyze the image and identify:
1. All visible objects.
2. Which object appears to be an obstacle.
3. Which object appears to be a possible navigation target.

Return a concise structured description.
Do not invent objects that are not visible.
""".strip()

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
                    "text": prompt,
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )

    # Move tensors to the model device.
    inputs = {
        key: value.to(model.device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }

    # ------------------------------------------------------------
    # Warm-up inference
    # ------------------------------------------------------------

    print()
    print("Running warm-up inference...")

    with torch.inference_mode():
        _ = model.generate(
            **inputs,
            max_new_tokens=64,
        )

    torch.cuda.synchronize()

    print("Warm-up complete.")

    # ------------------------------------------------------------
    # Timed inference
    # ------------------------------------------------------------

    print()
    print("Running timed VLM inference...")

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    inference_start = time.perf_counter()

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
        )

    torch.cuda.synchronize()

    inference_time = time.perf_counter() - inference_start

    # Remove input tokens from generated output.
    generated_ids_trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(
            inputs["input_ids"],
            generated_ids,
        )
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    result = output_text[0].strip()

    # ------------------------------------------------------------
    # Final metrics
    # ------------------------------------------------------------

    allocated_after, reserved_after = gpu_memory_gb()

    print()
    print("=" * 70)
    print("VLM TEST RESULT")
    print("=" * 70)

    print(f"Model:              {MODEL_ID}")
    print(f"GPU:                {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM:         {total_vram:.2f} GB")
    print(f"Model load time:    {model_load_time:.2f} s")
    print(f"Inference time:     {inference_time:.2f} s")
    print(f"VRAM allocated:     {allocated_after:.2f} GB")
    print(f"VRAM reserved:      {reserved_after:.2f} GB")

    print()
    print("VLM RESPONSE")
    print("-" * 70)
    print(result)
    print("-" * 70)

    print()
    print("TEST COMPLETE")


if __name__ == "__main__":
    main()
