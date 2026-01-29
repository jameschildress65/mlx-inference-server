"""IPC message models for orchestrator ↔ worker communication."""

from pydantic import BaseModel, Field
from typing import Optional, Literal


# Worker → Orchestrator Messages

class ReadyMessage(BaseModel):
    """Worker sends this on successful model load."""
    type: Literal["ready"] = "ready"
    model_name: str
    memory_gb: float


class CompletionResponse(BaseModel):
    """Worker sends this after generating completion."""
    type: Literal["completion_response"] = "completion_response"
    text: str
    tokens: int
    finish_reason: str  # "stop", "length", "error"


class StreamChunk(BaseModel):
    """Worker sends multiple chunks for streaming responses."""
    type: Literal["stream_chunk"] = "stream_chunk"
    text: str  # Decoded text segment
    token: int  # Token ID
    done: bool = False
    finish_reason: Optional[str] = None  # "stop", "length", or None


class ErrorMessage(BaseModel):
    """Worker sends this on error."""
    type: Literal["error"] = "error"
    error: str  # Error class name
    message: str  # Human-readable description


class PongMessage(BaseModel):
    """Worker responds to health ping."""
    type: Literal["pong"] = "pong"
    status: str = "healthy"


# Orchestrator → Worker Messages

class ImageData(BaseModel):
    """Image data for vision/multimodal requests.

    Two-tier strategy:
    - Small images (<500KB): inline base64 encoding
    - Large images (≥500KB): shared memory with offset/length

    C3 fix: Added generation field for shmem mode to detect stale reads
    when image buffer is reset during concurrent requests.
    """
    type: Literal["inline", "shmem"] = "inline"
    data: Optional[str] = None  # base64 encoded image (inline mode)
    offset: Optional[int] = None  # shared memory offset (shmem mode)
    length: Optional[int] = None  # shared memory length (shmem mode)
    generation: Optional[int] = None  # C3: generation counter (shmem mode)
    format: str = "jpeg"  # Image format: jpeg, png, webp, bmp, etc.


class CompletionRequest(BaseModel):
    """Orchestrator sends this to request generation."""
    type: Literal["completion"] = "completion"
    model: str
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 1.0
    repetition_penalty: float = 1.1
    stream: bool = False
    images: Optional[list[ImageData]] = None  # Vision/multimodal support


class PingMessage(BaseModel):
    """Orchestrator sends this to check worker health."""
    type: Literal["ping"] = "ping"


class ShutdownMessage(BaseModel):
    """Orchestrator sends this to gracefully shutdown worker."""
    type: Literal["shutdown"] = "shutdown"
