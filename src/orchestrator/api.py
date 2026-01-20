"""FastAPI application for MLX Server V3."""

import asyncio
import logging
import json
import time
from asyncio import Semaphore
from functools import lru_cache
from typing import Dict, Any, Optional, Union, Literal
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from .worker_manager import WorkerManager, NoModelLoadedError, WorkerError
from ..ipc.messages import CompletionRequest as IPCCompletionRequest
from ..ipc.shared_memory_bridge import WorkerCommunicationError
# StdioBridge no longer used directly - WorkerManager handles IPC abstraction
from ..config.server_config import ServerConfig
from .image_utils import prepare_images, ImageProcessingError

logger = logging.getLogger(__name__)

# Phase 2.7: Check MLX Metal availability for GPU health checks
try:
    import mlx.core as mx
    _HAS_MLX_METAL = hasattr(mx, 'metal')
except ImportError:
    mx = None
    _HAS_MLX_METAL = False

# Phase 1 (NASA): Request queue depth control - backpressure mechanism
# Opus 4.5 recommendation: Limit concurrent requests to prevent worker overload
_request_semaphore: Optional[Semaphore] = None
_max_queue_depth = 10  # Conservative: limits concurrent requests to single worker

def _get_request_semaphore() -> Semaphore:
    """Get or create the request semaphore for backpressure control.

    Lazy initialization ensures semaphore is created in the correct event loop.
    Returns HTTP 503 when queue is full instead of crashing worker.
    """
    global _request_semaphore
    if _request_semaphore is None:
        _request_semaphore = Semaphore(_max_queue_depth)
    return _request_semaphore


# Phase 2.2: Request Metrics - Monitor Phase 1 backpressure and performance
class RequestMetrics:
    """Track request metrics for monitoring and tuning Phase 1 improvements.

    Tracks:
    - Backpressure effectiveness (503 rejects, queue depth)
    - Request performance (duration, wait times)
    - Success/failure rates

    Thread-safety: Single-threaded asyncio, no locks needed.
    """

    def __init__(self):
        # Backpressure metrics
        self.queue_full_rejects = 0  # HTTP 503 count
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0

        # Performance metrics (keep last 1000 for stats)
        self._max_samples = 1000
        self.request_duration_ms: list[float] = []
        self.queue_wait_time_ms: list[float] = []

    def record_queue_reject(self):
        """Record a 503 rejection due to full queue."""
        self.queue_full_rejects += 1

    def record_request_start(self):
        """Record request received."""
        self.total_requests += 1

    def record_request_success(self, duration_ms: float, wait_ms: float = 0.0):
        """Record successful request completion."""
        self.successful_requests += 1
        self._add_duration(duration_ms)
        if wait_ms > 0:
            self._add_wait_time(wait_ms)

    def record_request_failure(self, duration_ms: float):
        """Record failed request."""
        self.failed_requests += 1
        self._add_duration(duration_ms)

    def _add_duration(self, ms: float):
        """Add duration sample, keep last N."""
        self.request_duration_ms.append(ms)
        if len(self.request_duration_ms) > self._max_samples:
            self.request_duration_ms.pop(0)

    def _add_wait_time(self, ms: float):
        """Add wait time sample, keep last N."""
        self.queue_wait_time_ms.append(ms)
        if len(self.queue_wait_time_ms) > self._max_samples:
            self.queue_wait_time_ms.pop(0)

    def get_stats(self) -> dict:
        """Get current metrics snapshot."""
        import statistics

        # Calculate percentiles for durations
        duration_stats = {}
        if self.request_duration_ms:
            duration_stats = {
                "p50": statistics.median(self.request_duration_ms),
                "p95": self._percentile(self.request_duration_ms, 0.95),
                "p99": self._percentile(self.request_duration_ms, 0.99),
                "avg": statistics.mean(self.request_duration_ms),
                "min": min(self.request_duration_ms),
                "max": max(self.request_duration_ms),
            }

        # Calculate wait time stats
        wait_stats = {}
        if self.queue_wait_time_ms:
            wait_stats = {
                "avg": statistics.mean(self.queue_wait_time_ms),
                "p95": self._percentile(self.queue_wait_time_ms, 0.95),
            }

        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "queue_full_rejects": self.queue_full_rejects,
            "success_rate": self.successful_requests / max(1, self.total_requests),
            "request_duration_ms": duration_stats,
            "queue_wait_time_ms": wait_stats,
        }

    def _percentile(self, data: list[float], p: float) -> float:
        """Calculate percentile (p in [0, 1])."""
        import statistics
        return statistics.quantiles(data, n=100)[int(p * 100) - 1] if len(data) >= 2 else (data[0] if data else 0.0)


# Global metrics instance
_request_metrics = RequestMetrics()


def get_metrics() -> RequestMetrics:
    """Get global metrics instance."""
    return _request_metrics


# Phase 1.1: Tokenizer cache - avoid reloading tokenizer on every request
# Opus 4.5 High Priority Fix H1: Bounded cache (was unbounded dict)
@lru_cache(maxsize=5)
def get_tokenizer(model_path: str):
    """Get cached tokenizer or load if not cached.

    Security: Opus 4.5 High Priority Fix H1
    - Bounded cache prevents memory leak from many different models
    - LRU eviction: keeps last 5 tokenizers, evicts least recently used
    - Each tokenizer ~50-200MB, so 5 = ~250MB-1GB max

    Args:
        model_path: HuggingFace model path (e.g. 'mlx-community/Qwen2.5-32B-Instruct-4bit')

    Returns:
        AutoTokenizer instance
    """
    from transformers import AutoTokenizer
    logger.info(f"Loading tokenizer for {model_path}")
    return AutoTokenizer.from_pretrained(model_path)


def clear_tokenizer_cache():
    """Clear tokenizer cache (call on model unload).

    Security: Opus 4.5 High Priority Fix H1
    Allows explicit cache cleanup when models are unloaded.
    """
    get_tokenizer.cache_clear()
    logger.info("Tokenizer cache cleared")


# Request/Response Models (OpenAI-compatible)

class CompletionRequest(BaseModel):
    """OpenAI-compatible completion request."""
    model: str
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 1.0
    repetition_penalty: float = 1.1
    stream: bool = False


class CompletionResponse(BaseModel):
    """OpenAI-compatible completion response."""
    id: str = "cmpl-mlx"
    object: str = "text_completion"
    created: int
    model: str
    choices: list
    usage: Dict[str, int]


# Vision/Multimodal Support - Content Block Models

class TextContent(BaseModel):
    """Text content block for multimodal messages."""
    type: Literal["text"] = "text"
    text: str


class ImageUrlContent(BaseModel):
    """Image URL content block for multimodal messages."""
    type: Literal["image_url"] = "image_url"
    image_url: dict  # {"url": "data:image/jpeg;base64,..." or "https://..."}


# Union type for content blocks
ContentBlock = Union[TextContent, ImageUrlContent]


class ChatMessage(BaseModel):
    """Chat message with optional multimodal content.

    Supports both text-only (backward compatible) and multimodal formats:
    - Text-only: content = "Hello"
    - Multimodal: content = [{"type": "text", "text": "..."}, {"type": "image_url", ...}]
    """
    role: str
    content: Union[str, list[ContentBlock]]  # Backward compatible with text-only


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str
    messages: list[ChatMessage]
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 1.0
    repetition_penalty: float = 1.1
    stream: bool = False


# API Application

def create_app(config: ServerConfig, worker_manager: WorkerManager) -> FastAPI:
    """
    Create FastAPI application.

    Args:
        config: Server configuration
        worker_manager: WorkerManager instance

    Returns:
        FastAPI application
    """
    app = FastAPI(
        title="MLX Server V3",
        description="Production-grade LLM inference server with process isolation",
        version="3.1.1"
    )

    # OpenAI-compatible error formatting
    def format_openai_error(message: str, error_type: str = "server_error") -> Dict[str, Any]:
        """
        Format error in OpenAI-compatible structure.

        Args:
            message: Error message
            error_type: Error type (server_error, invalid_request_error, authentication_error)

        Returns:
            OpenAI-compatible error object
        """
        return {
            "error": {
                "message": message,
                "type": error_type,
                "param": None,
                "code": None
            }
        }

    # Custom exception handler for OpenAI compatibility
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request, exc: HTTPException):
        """Convert HTTPException to OpenAI-compatible error format."""
        # Map HTTP status codes to OpenAI error types
        error_type = "server_error"
        if exc.status_code == 400:
            error_type = "invalid_request_error"
        elif exc.status_code in (401, 403):
            error_type = "authentication_error"

        return JSONResponse(
            status_code=exc.status_code,
            content=format_openai_error(exc.detail, error_type)
        )

    # Phase 2.7: Health Check Helpers
    def check_gpu_health() -> dict:
        """Check GPU availability and status."""
        try:
            # Check if MLX Metal is available
            if _HAS_MLX_METAL:
                # Try to get GPU memory info (if available on this platform)
                return {
                    "available": True,
                    "backend": "mlx.metal"
                }
            else:
                return {
                    "available": False,
                    "backend": "cpu_fallback"
                }
        except Exception as e:
            logger.debug(f"GPU health check failed: {e}")
            return {
                "available": False,
                "error": str(e)
            }

    def check_memory_health() -> dict:
        """Check system memory usage."""
        import psutil
        mem = psutil.virtual_memory()
        return {
            "percent_used": mem.percent,
            "healthy": mem.percent < 90,
            "available_gb": mem.available / (1024**3),
            "total_gb": mem.total / (1024**3)
        }

    def check_disk_health() -> dict:
        """Check disk usage."""
        import psutil
        disk = psutil.disk_usage('/')
        return {
            "percent_used": disk.percent,
            "healthy": disk.percent < 90,
            "available_gb": disk.free / (1024**3),
            "total_gb": disk.total / (1024**3)
        }

    def check_worker_health() -> dict:
        """Check worker process health."""
        try:
            health = worker_manager.health_check()
            status = worker_manager.get_status()
            return {
                "alive": health["healthy"],
                "status": health["status"],
                "model_loaded": status["model_loaded"],
                "model_name": status["model_name"] if status["model_loaded"] else None
            }
        except Exception as e:
            logger.error(f"Worker health check failed: {e}")
            return {
                "alive": False,
                "status": "error",
                "error": str(e)
            }

    # Health check (Phase 2.7: Deep health check)
    @app.get("/health")
    async def health_check():
        """
        Deep health check for load balancer and monitoring.

        Checks:
        - GPU availability and status
        - System memory usage (<90%)
        - Disk usage (<90%)
        - Worker process health
        - Model loaded status

        Returns:
            200 if all checks pass, 503 if any check fails
        """
        from datetime import datetime

        # Run all health checks
        gpu = check_gpu_health()
        memory = check_memory_health()
        disk = check_disk_health()
        worker = check_worker_health()

        checks = {
            "gpu": gpu.get("available", False),
            "memory": memory["healthy"],
            "disk": disk["healthy"],
            "worker": worker["alive"],
            "model_loaded": worker["model_loaded"]
        }

        # Overall health = all checks pass
        healthy = all(checks.values())

        # Detailed response
        response = {
            "healthy": healthy,
            "checks": checks,
            "details": {
                "gpu": gpu,
                "memory": memory,
                "disk": disk,
                "worker": worker
            },
            "timestamp": datetime.utcnow().isoformat(),
            "version": "3.1.1"
        }

        status_code = 200 if healthy else 503
        return JSONResponse(status_code=status_code, content=response)

    # Phase 2.7: Kubernetes-style readiness probe
    @app.get("/ready")
    async def readiness_check():
        """
        Kubernetes-style readiness probe.

        Ready = can handle new requests right now:
        - Queue has available slots
        - Worker is alive
        - Model is loaded

        Returns:
            200 if ready to handle requests, 503 if not ready
        """
        semaphore = _get_request_semaphore()
        worker = check_worker_health()

        # Ready conditions
        queue_space = semaphore._value > 0
        worker_alive = worker["alive"]
        model_loaded = worker["model_loaded"]

        ready = queue_space and worker_alive and model_loaded

        response = {
            "ready": ready,
            "conditions": {
                "queue_available": queue_space,
                "worker_alive": worker_alive,
                "model_loaded": model_loaded
            },
            "queue_slots_available": semaphore._value
        }

        status_code = 200 if ready else 503
        return JSONResponse(status_code=status_code, content=response)

    # V1 Completions endpoint
    @app.post("/v1/completions")
    async def completions(request: CompletionRequest):
        """
        OpenAI-compatible completions endpoint.

        Args:
            request: Completion request

        Returns:
            Completion response
        """
        # Phase 1 (NASA): Backpressure control - limit concurrent requests
        semaphore = _get_request_semaphore()

        # Opus Improvement 3: Fast-fail optimization for backpressure
        # Note: This check is a fast-fail optimization, not a safety mechanism.
        # The semaphore itself enforces the limit; this check allows immediate
        # 503 response instead of brief blocking. A small race window exists
        # but is benign (worst case: 1 extra request waits instead of failing).
        if semaphore.locked() and semaphore._value == 0:
            logger.warning("Request queue full - returning 503")
            # Phase 2.2: Track queue rejection
            get_metrics().record_queue_reject()
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "message": "Server is at capacity. Please retry after a brief delay.",
                        "type": "server_overloaded",
                        "code": "queue_full"
                    }
                }
            )

        # Acquire semaphore slot (auto-releases on function exit or exception)
        async with semaphore:
            # Phase 2.2: Track request timing
            request_start_time = time.time()
            get_metrics().record_request_start()

            try:
                # Check if model is loaded, if not load it
                status = worker_manager.get_status()
                if not status["model_loaded"] or status["model_name"] != request.model:
                    logger.info(f"Loading model on-demand: {request.model}")
                    load_result = worker_manager.load_model(request.model)
                    logger.info(f"Model loaded: {load_result.model_name} ({load_result.memory_gb:.2f} GB)")

                # Create IPC request
                ipc_request = IPCCompletionRequest(
                    model=request.model,
                    prompt=request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    repetition_penalty=request.repetition_penalty,
                    stream=request.stream
                )

                # Generate
                if request.stream:
                    # Streaming
                    async def generate_stream():
                        """Generate SSE stream for text completions."""
                        try:
                            # Use WorkerManager's streaming method (handles SharedMemoryBridge/StdioBridge)
                            for chunk in worker_manager.generate_stream(ipc_request):
                                if chunk.type == "stream_chunk":
                                    # Format as SSE for text completions (not chat)
                                    data = {
                                        "id": "cmpl-mlx-v3",
                                        "object": "text_completion",
                                        "created": int(__import__('time').time()),
                                        "model": request.model,
                                        "choices": [{
                                            "index": 0,
                                            "text": chunk.text if chunk.text else "",
                                            "finish_reason": chunk.finish_reason
                                        }]
                                    }
                                    yield f"data: {json.dumps(data)}\n\n"

                                    if chunk.done:
                                        yield "data: [DONE]\n\n"
                                        break
                                elif chunk.type == "error":
                                    yield f"data: {json.dumps(format_openai_error(chunk.message))}\n\n"
                                    break
                        except Exception as e:
                            # M2: Don't expose internal details in streaming errors
                            logger.error(f"Streaming error: {e}", exc_info=True)
                            yield f"data: {json.dumps(format_openai_error('Internal server error'))}\n\n"

                    return StreamingResponse(generate_stream(), media_type="text/event-stream")
                else:
                    # Non-streaming
                    result = worker_manager.generate(ipc_request)

                    # Count prompt tokens using tokenizer
                    try:
                        tokenizer = get_tokenizer(request.model)
                        prompt_tokens = len(tokenizer.encode(request.prompt))
                    except Exception as e:
                        # Can't load tokenizer (e.g., test mock), default to 0
                        logger.warning(f"Could not count prompt tokens: {e}")
                        prompt_tokens = 0
                    completion_tokens = result["tokens"]
                    total_tokens = prompt_tokens + completion_tokens

                    # Phase 2.2: Track successful request
                    duration_ms = (time.time() - request_start_time) * 1000
                    get_metrics().record_request_success(duration_ms)

                    # Format response (OpenAI-compatible)
                    return {
                        "id": "cmpl-mlx-v3",
                        "object": "text_completion",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [{
                            "text": result["text"],
                            "index": 0,
                            "logprobs": None,
                            "finish_reason": result["finish_reason"]
                        }],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": total_tokens
                        }
                    }

            except HTTPException:
                # Phase 2.2: Track failure
                duration_ms = (time.time() - request_start_time) * 1000
                get_metrics().record_request_failure(duration_ms)
                # Re-raise HTTP exceptions (like 501 for streaming)
                raise
            except ValueError as e:
                # Phase 2.2: Track failure
                duration_ms = (time.time() - request_start_time) * 1000
                get_metrics().record_request_failure(duration_ms)
                # Validation errors (e.g., "Vision model requires at least one image")
                raise HTTPException(status_code=400, detail=str(e))
            except NoModelLoadedError as e:
                # Phase 2.2: Track failure
                duration_ms = (time.time() - request_start_time) * 1000
                get_metrics().record_request_failure(duration_ms)
                raise HTTPException(status_code=503, detail=str(e))
            except WorkerCommunicationError as e:
                # Phase 2.2: Track failure
                duration_ms = (time.time() - request_start_time) * 1000
                get_metrics().record_request_failure(duration_ms)
                # Buffer full - system overloaded (backpressure)
                logger.warning(f"Backpressure triggered: {e}")
                raise HTTPException(status_code=503, detail=str(e))
            except WorkerError as e:
                # Phase 2.2: Track failure
                duration_ms = (time.time() - request_start_time) * 1000
                get_metrics().record_request_failure(duration_ms)
                logger.error(f"Worker error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
            except Exception as e:
                # Phase 2.2: Track failure
                duration_ms = (time.time() - request_start_time) * 1000
                get_metrics().record_request_failure(duration_ms)
                # M2: Don't expose internal details to client
                logger.error(f"Unexpected error: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail="Internal server error")

    # V1 Chat completions endpoint
    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        """
        OpenAI-compatible chat completions endpoint.

        Args:
            request: Chat completion request

        Returns:
            Chat completion response
        """
        # Phase 1 (NASA): Backpressure control - limit concurrent requests
        semaphore = _get_request_semaphore()

        # Opus Improvement 3: Fast-fail optimization for backpressure
        # Note: This check is a fast-fail optimization, not a safety mechanism.
        # The semaphore itself enforces the limit; this check allows immediate
        # 503 response instead of brief blocking. A small race window exists
        # but is benign (worst case: 1 extra request waits instead of failing).
        if semaphore.locked() and semaphore._value == 0:
            logger.warning("Request queue full - returning 503")
            # Phase 2.2: Track queue rejection
            get_metrics().record_queue_reject()
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "message": "Server is at capacity. Please retry after a brief delay.",
                        "type": "server_overloaded",
                        "code": "queue_full"
                    }
                }
            )

        # Acquire semaphore slot (auto-releases on function exit or exception)
        async with semaphore:
            # Phase 2.2: Track request timing
            request_start_time = time.time()
            get_metrics().record_request_start()

            # Convert chat messages to prompt using model's chat template
            logger.info(f"Chat completion request: stream={request.stream}, messages={len(request.messages)}, "
                       f"max_tokens={request.max_tokens}, temp={request.temperature}, "
                       f"top_p={request.top_p}, rep_penalty={request.repetition_penalty}")

            try:
                # Get cached tokenizer (Phase 1.1 optimization)
                tokenizer = get_tokenizer(request.model)

                # Convert messages to dicts for apply_chat_template
                # Handle multimodal content (list of blocks) by extracting text only
                messages_dict = []
                for msg in request.messages:
                    if isinstance(msg.content, list):
                        # Multimodal: extract text blocks only (skip images for prompt)
                        text_parts = [
                            block.text for block in msg.content
                            if hasattr(block, 'type') and block.type == 'text'
                        ]
                        content_str = " ".join(text_parts)
                    else:
                        # Text-only (backward compatible)
                        content_str = msg.content
                    messages_dict.append({"role": msg.role, "content": content_str})

                # Apply chat template (tokenize=False returns string, not tokens)
                prompt = tokenizer.apply_chat_template(
                    messages_dict,
                    tokenize=False,
                    add_generation_prompt=True
                )
                logger.info(f"Chat template applied, prompt length: {len(prompt)} chars, stream={request.stream}")
                logger.debug(f"Generated prompt: {prompt[:200]}...")
            except Exception as e:
                logger.warning(f"Failed to apply chat template for {request.model}: {e}")
                logger.warning("Falling back to simple concatenation")
                # Fallback to simple concatenation if template fails
                # Handle multimodal content (list of content blocks)
                prompt_parts = []
                for msg in request.messages:
                    if isinstance(msg.content, list):
                        # Multimodal: extract text blocks only (skip images)
                        text_parts = [
                            block.text for block in msg.content
                            if hasattr(block, 'type') and block.type == 'text'
                        ]
                        content_str = " ".join(text_parts)
                    else:
                        # Text-only
                        content_str = msg.content
                    prompt_parts.append(f"{msg.role}: {content_str}")
                prompt = "\n".join(prompt_parts)

            try:
                # Check if model is loaded, if not load it
                status = worker_manager.get_status()
                if not status["model_loaded"] or status["model_name"] != request.model:
                    logger.info(f"Loading model on-demand: {request.model}")
                    load_result = worker_manager.load_model(request.model)
                    logger.info(f"Model loaded: {load_result.model_name} ({load_result.memory_gb:.2f} GB)")

                # Phase 2: Image preprocessing for vision/multimodal requests
                images = None
                has_images = any(
                    isinstance(msg.content, list) and
                    any(hasattr(block, 'type') and block.type == 'image_url' for block in msg.content)
                    for msg in request.messages
                )

                if has_images:
                    logger.info("Vision/multimodal request detected - preprocessing images")
                    try:
                        # Collect all content blocks from all messages
                        all_content_blocks = []
                        for msg in request.messages:
                            if isinstance(msg.content, list):
                                all_content_blocks.extend(msg.content)

                        # Get bridge for large images (if using shared memory)
                        bridge = worker_manager.bridge if hasattr(worker_manager, 'bridge') else None

                        # Preprocess images
                        images = await prepare_images(all_content_blocks, bridge=bridge)
                        logger.info(f"Preprocessed {len(images)} images for request")
                    except ImportError as e:
                        # Pillow/mlx-vlm not installed
                        logger.warning(f"Missing dependency for vision: {e}")
                        raise HTTPException(status_code=400, detail=str(e))
                    except ImageProcessingError as e:
                        logger.warning(f"Image preprocessing failed: {e}")
                        raise HTTPException(status_code=400, detail=f"Image processing error: {e}")

                # Create IPC request
                ipc_request = IPCCompletionRequest(
                    model=request.model,
                    prompt=prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    repetition_penalty=request.repetition_penalty,
                    stream=request.stream,
                    images=images  # Pass preprocessed images (None for text-only)
                )

                # Generate
                if request.stream:
                    # Streaming - chat format
                    async def generate_stream():
                        """Generate SSE stream for chat completions."""
                        try:
                            # Use WorkerManager's streaming method (handles SharedMemoryBridge/StdioBridge)
                            for chunk in worker_manager.generate_stream(ipc_request):
                                if chunk.type == "stream_chunk":
                                    # Format as SSE for chat completions (use delta.content)
                                    data = {
                                        "id": "chatcmpl-mlx-v3",
                                        "object": "chat.completion.chunk",
                                        "created": int(__import__('time').time()),
                                        "model": request.model,
                                        "choices": [{
                                            "index": 0,
                                            "delta": {"content": chunk.text} if chunk.text else {},
                                            "finish_reason": chunk.finish_reason
                                        }]
                                    }
                                    yield f"data: {json.dumps(data)}\n\n"

                                    if chunk.done:
                                        yield "data: [DONE]\n\n"
                                        break
                                elif chunk.type == "error":
                                    yield f"data: {json.dumps(format_openai_error(chunk.message))}\n\n"
                                    break
                        except Exception as e:
                            # M2: Don't expose internal details in streaming errors
                            logger.error(f"Streaming error: {e}", exc_info=True)
                            yield f"data: {json.dumps(format_openai_error('Internal server error'))}\n\n"

                    return StreamingResponse(generate_stream(), media_type="text/event-stream")
                else:
                    # Non-streaming
                    import time
                    start_time = time.time()
                    result = worker_manager.generate(ipc_request)
                    generation_time = time.time() - start_time

                    # Convert to chat format
                    response_text = result["text"]
                    completion_tokens = result["tokens"]

                    # Count prompt tokens using tokenizer (reload if template failed)
                    try:
                        prompt_tokens = len(tokenizer.encode(prompt))
                    except NameError:
                        # Tokenizer not defined (template failed), try to reload it
                        try:
                            tokenizer = get_tokenizer(request.model)
                            prompt_tokens = len(tokenizer.encode(prompt))
                        except Exception as e:
                            # Can't load tokenizer (e.g., test mock), default to 0
                            logger.warning(f"Could not count prompt tokens: {e}")
                            prompt_tokens = 0
                    total_tokens = prompt_tokens + completion_tokens

                    tokens_per_sec = completion_tokens / generation_time if generation_time > 0 else 0
                    logger.info(f"Chat completion: {len(response_text)} chars, {completion_tokens} tokens in {generation_time:.1f}s ({tokens_per_sec:.1f} tok/s)")
                    logger.debug(f"Response content: {response_text[:200]}...")

                    # Phase 2.2: Track successful request
                    duration_ms = (time.time() - request_start_time) * 1000
                    get_metrics().record_request_success(duration_ms)

                    return {
                        "id": "chatcmpl-mlx-v3",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": response_text
                            },
                            "finish_reason": result["finish_reason"]
                        }],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": total_tokens,
                            "tokens_per_sec": round(tokens_per_sec, 2)
                        }
                    }

            except HTTPException:
                # Phase 2.2: Track failure
                duration_ms = (time.time() - request_start_time) * 1000
                get_metrics().record_request_failure(duration_ms)
                # Re-raise HTTP exceptions
                raise
            except ValueError as e:
                # Phase 2.2: Track failure
                duration_ms = (time.time() - request_start_time) * 1000
                get_metrics().record_request_failure(duration_ms)
                # Validation errors (e.g., "Vision model requires at least one image")
                raise HTTPException(status_code=400, detail=str(e))
            except NoModelLoadedError as e:
                # Phase 2.2: Track failure
                duration_ms = (time.time() - request_start_time) * 1000
                get_metrics().record_request_failure(duration_ms)
                raise HTTPException(status_code=503, detail=str(e))
            except WorkerCommunicationError as e:
                # Phase 2.2: Track failure
                duration_ms = (time.time() - request_start_time) * 1000
                get_metrics().record_request_failure(duration_ms)
                # Buffer full - system overloaded (backpressure)
                logger.warning(f"Backpressure triggered: {e}")
                raise HTTPException(status_code=503, detail=str(e))
            except WorkerError as e:
                # Phase 2.2: Track failure
                duration_ms = (time.time() - request_start_time) * 1000
                get_metrics().record_request_failure(duration_ms)
                logger.error(f"Worker error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
            except Exception as e:
                # Phase 2.2: Track failure
                duration_ms = (time.time() - request_start_time) * 1000
                get_metrics().record_request_failure(duration_ms)
                # M2: Don't expose internal details to client
                logger.error(f"Unexpected error: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail="Internal server error")

    # Models list endpoint
    @app.get("/v1/models")
    async def list_models():
        """List available models from HuggingFace cache."""
        import os
        from pathlib import Path

        models = []

        # Check HF_HOME or default cache location
        hf_home = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        hub_path = Path(hf_home) / "hub"

        if hub_path.exists():
            # Scan for model directories (format: models--org--model-name)
            for item in hub_path.iterdir():
                if item.is_dir() and item.name.startswith("models--"):
                    # Convert models--mlx-community--Model-Name to mlx-community/Model-Name
                    parts = item.name.split("--", 2)
                    if len(parts) == 3:
                        model_id = f"{parts[1]}/{parts[2]}"
                        # Only include mlx-community models
                        if parts[1] == "mlx-community":
                            models.append({
                                "id": model_id,
                                "object": "model",
                                "created": int(item.stat().st_mtime),
                                "owned_by": parts[1]
                            })

        # Sort by creation time (most recent first)
        models.sort(key=lambda x: x["created"], reverse=True)

        return {
            "object": "list",
            "data": models
        }

    return app


def create_admin_app(config: ServerConfig, worker_manager: WorkerManager) -> FastAPI:
    """
    Create admin API application.

    Args:
        config: Server configuration
        worker_manager: WorkerManager instance

    Returns:
        FastAPI application for admin endpoints
    """
    app = FastAPI(
        title="MLX Server V3 - Admin API",
        description="Administrative endpoints for model management",
        version="3.1.1"
    )

    @app.get("/admin/health")
    async def admin_health():
        """Admin health check with worker status."""
        health = worker_manager.health_check()
        return {
            "status": "healthy" if health["healthy"] else "degraded",
            "worker_status": health["status"],
            "version": "3.1.1"
        }

    @app.get("/admin/status")
    async def admin_status():
        """Get server and worker status."""
        status = worker_manager.get_status()
        health = worker_manager.health_check()

        return {
            "status": "running",
            "version": "3.1.1",
            "ports": {
                "main": config.main_port,
                "admin": config.admin_port
            },
            "model": {
                "loaded": status["model_loaded"],
                "name": status["model_name"],
                "memory_gb": status["memory_gb"]
            },
            "worker": {
                "healthy": health["healthy"],
                "status": health["status"]
            },
            "config": {
                "machine_type": config.machine_type,
                "total_ram_gb": config.total_ram_gb,
                "idle_timeout_seconds": config.idle_timeout_seconds
            }
        }

    @app.get("/admin/metrics")
    async def admin_metrics():
        """
        Get request metrics for monitoring Phase 1 backpressure and performance.

        Returns:
            Metrics including:
            - Request counts (total, success, failures, 503 rejects)
            - Success rate
            - Request duration statistics (p50, p95, p99, avg, min, max)
            - Queue wait time statistics
            - Current queue state
        """
        metrics = get_metrics()
        semaphore = _get_request_semaphore()

        # Get metrics snapshot
        stats = metrics.get_stats()

        # Add current queue state
        stats["queue"] = {
            "max_depth": _max_queue_depth,
            "active_requests": _max_queue_depth - semaphore._value,
            "available_slots": semaphore._value,
            "utilization": (_max_queue_depth - semaphore._value) / _max_queue_depth
        }

        return stats

    @app.get("/admin/capabilities")
    async def admin_capabilities():
        """
        Get model capabilities (text/vision support).

        Returns:
            Model capabilities and mlx-vlm availability
        """
        status = worker_manager.get_status()
        model_name = status["model_name"]

        # Detect vision capability from model name
        vision_patterns = ["qwen2-vl", "qwen2.5-vl", "-vl-", "llava", "idefics"]
        is_vision = False
        detection_method = "model_name"

        if model_name:
            model_lower = model_name.lower()
            is_vision = any(pattern in model_lower for pattern in vision_patterns)
        else:
            detection_method = "no_model_loaded"

        # Check mlx-vlm availability
        mlx_vlm_available = False
        try:
            import mlx_vlm
            mlx_vlm_available = True
        except ImportError:
            pass

        return {
            "model": model_name,
            "capabilities": {
                "text": True,  # All models support text
                "vision": is_vision
            },
            "detection_method": detection_method,
            "mlx_vlm_available": mlx_vlm_available,
            "notes": (
                "Vision models require mlx-vlm package. "
                "Install: pip install mlx-vlm pillow"
            ) if is_vision and not mlx_vlm_available else None
        }

    @app.post("/admin/load")
    async def admin_load(model_path: str):
        """
        Load a model.

        Args:
            model_path: HuggingFace model path

        Returns:
            Load result
        """
        try:
            result = worker_manager.load_model(model_path)
            return {
                "status": "success",
                "model_name": result.model_name,
                "memory_gb": result.memory_gb,
                "load_time": result.load_time
            }
        except Exception as e:
            # M2: Don't expose internal details to client
            logger.error(f"Load failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to load model")

    @app.post("/admin/unload")
    async def admin_unload():
        """
        Unload current model (terminate worker).

        Returns:
            Unload result with memory freed
        """
        try:
            result = worker_manager.unload_model()
            return result.to_dict()
        except NoModelLoadedError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            # M2: Don't expose internal details to client
            logger.error(f"Unload failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to unload model")

    return app
