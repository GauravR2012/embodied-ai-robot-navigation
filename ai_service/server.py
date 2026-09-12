import torch
from fastapi import FastAPI, HTTPException

from .llm_engine import LLMEngine
from .schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    PerceptionRequest,
    PerceptionResponse,
)
from .vlm_engine import VLMEngine


app = FastAPI(
    title="Embodied AI Gateway",
    version="0.1.0",
)


llm_engine = LLMEngine()
vlm_engine = VLMEngine()


@app.on_event("startup")
def startup_event():

    print("=" * 70)
    print("EMBODIED AI GATEWAY STARTUP")
    print("=" * 70)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required."
        )

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    llm_engine.load()
    vlm_engine.load()

    allocated = (
        torch.cuda.memory_allocated()
        / (1024 ** 3)
    )

    reserved = (
        torch.cuda.memory_reserved()
        / (1024 ** 3)
    )

    print()
    print(
        f"VRAM allocated: {allocated:.2f} GB"
    )
    print(
        f"VRAM reserved:  {reserved:.2f} GB"
    )

    print("=" * 70)
    print("GATEWAY READY")
    print("=" * 70)


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health():

    gpu = None

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)

    return HealthResponse(
        status="ok",
        llm_loaded=llm_engine.loaded,
        vlm_loaded=vlm_engine.loaded,
        gpu=gpu,
    )


@app.post(
    "/v1/chat",
    response_model=ChatResponse,
)
def chat(request: ChatRequest):

    try:

        response = llm_engine.generate(
            instruction=request.instruction,
            context=request.context,
        )

        return ChatResponse(
            response=response,
            model="Qwen/Qwen2.5-3B-Instruct",
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


@app.post(
    "/v1/perception",
    response_model=PerceptionResponse,
)
def perception(
    request: PerceptionRequest,
):

    try:

        response = vlm_engine.analyze(
            image_base64=request.image_base64,
            instruction=request.instruction,
        )

        return PerceptionResponse(
            response=response,
            model="Qwen/Qwen2.5-VL-7B-Instruct",
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc
