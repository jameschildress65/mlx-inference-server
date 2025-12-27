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


class PingMessage(BaseModel):
    """Orchestrator sends this to check worker health."""
    type: Literal["ping"] = "ping"


class ShutdownMessage(BaseModel):
    """Orchestrator sends this to gracefully shutdown worker."""
    type: Literal["shutdown"] = "shutdown"
