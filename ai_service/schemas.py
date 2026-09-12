from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    instruction: str
    context: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    response: str
    model: str


class PerceptionRequest(BaseModel):
    instruction: str
    image_base64: str


class PerceptionResponse(BaseModel):
    response: str
    model: str


class HealthResponse(BaseModel):
    status: str
    llm_loaded: bool
    vlm_loaded: bool
    gpu: str | None = None
