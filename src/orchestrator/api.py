"""FastAPI application for MLX Server V3."""

import asyncio
import logging
import json
import time
import uuid
from asyncio import Semaphore
from functools import lru_cache
from typing import Dict, Any, Optional, Union, Literal
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from .worker_manager import WorkerManager, NoModelLoadedError, WorkerError
from ..ipc.messages import CompletionRequest as IPCCompletionRequest
from ..ipc.shared_memory_bridge import WorkerCommunicationError
# StdioBridge no longer used directly - WorkerManager handles IPC abstraction
from ..config.server_config import ServerConfig
from .image_utils import prepare_images, ImageProcessingError
# P0 bloat remediation: Shared helpers for endpoints
from .request_handlers import (
    BackpressureGuard, RequestTimer, ensure_model_loaded, RateLimitGuard,
    QueueGuard, parse_priority
)
from .rate_limiter import RateLimiter, RateLimitConfig
from .response_formatters import CompletionFormatter, ChatCompletionFormatter
from .health_checks import check_gpu_health, check_memory_health, check_disk_health, check_worker_health
from .prometheus_metrics import metrics_router, metrics_collector
from .priority_queue import PriorityRequestQueue, QueueConfig, Priority

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
_max_queue_depth = 10  # Default fallback (Phase 2.1: Overridden by config.max_concurrent_requests)

# P1: Global rate limiter instance (disabled by default for home lab)
_rate_limiter: Optional[RateLimiter] = None

# Priority request queue (replaces semaphore when enabled)
_request_queue: Optional[PriorityRequestQueue] = None
_queue_enabled: bool = False


def _generate_request_id() -> str:
    """
    6.2: Generate unique request ID for tracing/correlation.

    Format: req-<uuid4> (e.g., req-550e8400-e29b-41d4-a716-446655440000)
    Used in responses and logs for debugging/tracing.
    """
    return f"req-{uuid.uuid4()}"

def _get_request_semaphore() -> Semaphore:
    """Get or create the request semaphore for backpressure control.

    Lazy initialization ensures semaphore is created in the correct event loop.
    Returns HTTP 503 when queue is full instead of crashing worker.
    """
    global _request_semaphore
    if _request_semaphore is None:
        _request_semaphore = Semaphore(_max_queue_depth)
    return _request_semaphore


def _get_rate_limiter() -> RateLimiter:
    """Get global rate limiter instance.

    P1: Returns the rate limiter configured at app startup.
    Disabled by default (config.rate_limit_enabled=False).
    """
    global _rate_limiter
    if _rate_limiter is None:
        # Fallback if not initialized (shouldn't happen in normal operation)
        _rate_limiter = RateLimiter(RateLimitConfig(enabled=False))
    return _rate_limiter


def _get_request_queue() -> Optional[PriorityRequestQueue]:
    """Get global priority request queue.

    Returns None if queue not enabled (uses semaphore fallback).
    """
    return _request_queue


def _is_queue_enabled() -> bool:
    """Check if priority queue is enabled."""
    return _queue_enabled and _request_queue is not None


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

# H3: Schema size limits to prevent DoS (module-level constants)
RESPONSE_FORMAT_MAX_SCHEMA_SIZE_BYTES = 64 * 1024  # 64KB max schema size
RESPONSE_FORMAT_MAX_SCHEMA_DEPTH = 20  # Maximum nesting depth


def _check_schema_depth(obj: Any, current_depth: int = 0) -> int:
    """Check maximum nesting depth of schema object."""
    if current_depth > RESPONSE_FORMAT_MAX_SCHEMA_DEPTH:
        return current_depth
    if isinstance(obj, dict):
        if not obj:
            return current_depth
        return max(_check_schema_depth(v, current_depth + 1) for v in obj.values())
    if isinstance(obj, list):
        if not obj:
            return current_depth
        return max(_check_schema_depth(v, current_depth + 1) for v in obj)
    return current_depth


class ResponseFormat(BaseModel):
    """OpenAI-compatible response_format field for structured output.

    Supports:
    - type="text": Default, unstructured text output
    - type="json_object": Constrain output to valid JSON object
    - type="json_schema": Constrain output to match provided JSON schema

    Requires outlines package for json_object/json_schema modes.

    Security (Opus H2/H3):
    - Validates json_schema is provided when type="json_schema"
    - Limits schema size to prevent DoS via complex regex compilation
    """
    type: Literal["text", "json_object", "json_schema"] = "text"
    json_schema: Optional[Dict[str, Any]] = None  # Required when type="json_schema"

    def model_post_init(self, __context: Any) -> None:
        """Validate response_format after model initialization.

        Opus H2: Validate json_schema is provided when type="json_schema"
        Opus H3: Validate schema size/complexity limits
        """
        # H2: Require schema when type is json_schema
        if self.type == "json_schema" and self.json_schema is None:
            raise ValueError(
                "json_schema is required when type='json_schema'. "
                "Provide a JSON schema or use type='json_object' for generic JSON."
            )

        # H3: Validate schema size limits
        if self.json_schema is not None:
            schema_str = json.dumps(self.json_schema)

            # Check size
            if len(schema_str) > RESPONSE_FORMAT_MAX_SCHEMA_SIZE_BYTES:
                raise ValueError(
                    f"JSON schema too large: {len(schema_str)} bytes exceeds "
                    f"{RESPONSE_FORMAT_MAX_SCHEMA_SIZE_BYTES} byte limit"
                )

            # Check depth
            depth = _check_schema_depth(self.json_schema)
            if depth > RESPONSE_FORMAT_MAX_SCHEMA_DEPTH:
                raise ValueError(
                    f"JSON schema too deeply nested: depth {depth} exceeds "
                    f"{RESPONSE_FORMAT_MAX_SCHEMA_DEPTH} level limit"
                )


class CompletionRequest(BaseModel):
    """OpenAI-compatible completion request.

    S4 Security Fix: All generation parameters have bounds validation
    to prevent resource exhaustion and invalid model behavior.
    """
    model: str
    prompt: str = Field(..., min_length=1, max_length=100000)  # S4: Limit prompt size
    max_tokens: int = Field(default=100, ge=1, le=32768)  # S4: Reasonable token limit
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)  # S4: Valid temp range
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)  # S4: Valid probability
    repetition_penalty: float = Field(default=1.1, ge=0.0, le=10.0)  # S4: Sane range
    stream: bool = False
    response_format: Optional[ResponseFormat] = None  # JSON mode support


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
    """OpenAI-compatible chat completion request.

    S4 Security Fix: All generation parameters have bounds validation
    to prevent resource exhaustion and invalid model behavior.
    """
    model: str
    messages: list[ChatMessage] = Field(..., min_length=1, max_length=100)  # S4: Limit message count
    max_tokens: int = Field(default=100, ge=1, le=32768)  # S4: Reasonable token limit
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)  # S4: Valid temp range
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)  # S4: Valid probability
    repetition_penalty: float = Field(default=1.1, ge=0.0, le=10.0)  # S4: Sane range
    stream: bool = False
    response_format: Optional[ResponseFormat] = None  # JSON mode support


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
    # Phase 2.1: Initialize queue depth from configuration
    global _max_queue_depth, _rate_limiter, _request_queue, _queue_enabled
    _max_queue_depth = config.max_concurrent_requests
    logger.info(f"Request queue depth set to {_max_queue_depth} (from config)")

    # P1: Initialize rate limiter from configuration
    _rate_limiter = RateLimiter(RateLimitConfig(
        requests_per_minute=config.rate_limit_rpm,
        burst_size=config.rate_limit_burst,
        enabled=config.rate_limit_enabled
    ))
    if config.rate_limit_enabled:
        logger.info(f"Rate limiting enabled: {config.rate_limit_rpm} RPM, burst={config.rate_limit_burst}")
    else:
        logger.info("Rate limiting disabled (enable with MLX_RATE_LIMIT_ENABLED=1)")

    # Initialize priority request queue
    _queue_enabled = config.queue_enabled
    if _queue_enabled:
        _request_queue = PriorityRequestQueue(QueueConfig(
            max_slots=config.max_concurrent_requests,
            max_queue_depth=config.queue_max_depth,
            reject_threshold=config.queue_reject_threshold,
            timeout_high=float(config.queue_timeout_high),
            timeout_normal=float(config.queue_timeout_normal),
            timeout_low=float(config.queue_timeout_low),
            enabled=True
        ))
        logger.info(
            f"Priority queue enabled: max_depth={config.queue_max_depth}, "
            f"timeouts=({config.queue_timeout_high}/{config.queue_timeout_normal}/{config.queue_timeout_low}s)"
        )
    else:
        logger.info("Priority queue disabled (using semaphore backpressure)")

    app = FastAPI(
        title="MLX Server V3",
        description="Production-grade LLM inference server with process isolation",
        version="3.2.0"
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
    # P0: Health check functions now in health_checks module
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

        # P0: Run all health checks using shared helpers
        gpu = check_gpu_health()
        memory = check_memory_health()
        disk = check_disk_health()
        worker = check_worker_health(worker_manager)

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
            "version": "3.2.0"
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
        worker = check_worker_health(worker_manager)

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
    async def completions(request: CompletionRequest, http_request: Request):
        """
        OpenAI-compatible completions endpoint.

        Args:
            request: Completion request
            http_request: FastAPI Request for headers

        Returns:
            Completion response
        """
        # P1: Rate limit check (before backpressure - fail fast if rate limited)
        rate_guard = RateLimitGuard(_get_rate_limiter())
        rate_response = rate_guard.check_rate_limit()
        if rate_response:
            return rate_response

        # 6.2: Generate unique request ID for tracing/correlation
        request_id = _generate_request_id()
        logger.debug(f"[{request_id}] Processing completion request")

        # Use priority queue or semaphore fallback
        if _is_queue_enabled():
            return await _completions_with_queue(request, http_request, request_id, worker_manager, format_openai_error)
        else:
            return await _completions_with_semaphore(request, request_id, worker_manager, format_openai_error)

    async def _completions_with_queue(request, http_request, request_id, worker_manager, format_openai_error):
        """Process completion with priority queue."""
        priority = parse_priority(http_request)
        queue_guard = QueueGuard(_get_request_queue(), get_metrics())

        # Enqueue request
        error_response, queue_position = await queue_guard.enqueue(request_id, priority)
        if error_response:
            return error_response

        # Wait for slot
        try:
            timeout_response = await queue_guard.wait_for_slot()
            if timeout_response:
                return timeout_response

            # Record queue wait time
            wait_ms = queue_guard.wait_time_ms
            if wait_ms > 0:
                metrics_collector.record_queue_wait(wait_ms / 1000, priority.name.lower())

            # Process request
            response = await _process_completion(request, request_id, worker_manager, format_openai_error)

            # Add queue headers to response if it's a JSONResponse
            if hasattr(response, 'headers'):
                response.headers["X-Queue-Wait-Ms"] = str(int(wait_ms))
                if queue_position and queue_position > 0:
                    response.headers["X-Queue-Position"] = str(queue_position)

            return response
        finally:
            await queue_guard.release()

    async def _completions_with_semaphore(request, request_id, worker_manager, format_openai_error):
        """Process completion with semaphore backpressure (fallback)."""
        guard = BackpressureGuard(_get_request_semaphore(), get_metrics())
        capacity_response = guard.check_capacity()
        if capacity_response:
            return capacity_response

        async with guard.semaphore:
            return await _process_completion(request, request_id, worker_manager, format_openai_error)

    async def _process_completion(request, request_id, worker_manager, format_openai_error):
        """Core completion processing logic."""
        # P0: Track request timing using shared timer
        timer = RequestTimer(get_metrics())
        timer.start()

        try:
            # P0: Ensure model loaded using shared helper
            ensure_model_loaded(worker_manager, request.model)

            # Create IPC request
            # Extract JSON mode settings
            response_format_type = None
            json_schema = None
            if request.response_format:
                response_format_type = request.response_format.type
                if response_format_type == "json_schema":
                    json_schema = request.response_format.json_schema

            ipc_request = IPCCompletionRequest(
                model=request.model,
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
                stream=request.stream,
                response_format_type=response_format_type,
                json_schema=json_schema
            )

            # Generate
            if request.stream:
                # P0: Streaming using shared formatter
                # 6.2: Capture request_id in closure for streaming
                stream_request_id = request_id

                async def generate_stream():
                    """Generate SSE stream for text completions."""
                    try:
                        for chunk in worker_manager.generate_stream(ipc_request):
                            if chunk.type == "stream_chunk":
                                # P0: Use CompletionFormatter for SSE formatting
                                # 6.2: Pass request_id for tracing
                                yield CompletionFormatter.format_stream_chunk(
                                    request.model, chunk, request_id=stream_request_id
                                )
                                if chunk.done:
                                    yield "data: [DONE]\n\n"
                                    break
                            elif chunk.type == "error":
                                yield f"data: {json.dumps(format_openai_error(chunk.message))}\n\n"
                                break
                    except Exception as e:
                        logger.error(f"[{stream_request_id}] Streaming error: {e}", exc_info=True)
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
                    logger.warning(f"[{request_id}] Could not count prompt tokens: {e}")
                    prompt_tokens = 0

                # P0: Track success and format using shared helpers
                # 6.2: Pass request_id for tracing
                timer.record_success()
                logger.debug(f"[{request_id}] Completion successful: {result['tokens']} tokens")
                return CompletionFormatter.format_non_streaming(
                    request.model, result, prompt_tokens, request_id=request_id
                )

        except HTTPException:
            timer.record_failure()
            raise
        except ValueError as e:
            timer.record_failure()
            raise HTTPException(status_code=400, detail=str(e))
        except NoModelLoadedError as e:
            timer.record_failure()
            raise HTTPException(status_code=503, detail=str(e))
        except WorkerCommunicationError as e:
            timer.record_failure()
            logger.warning(f"Backpressure triggered: {e}")
            raise HTTPException(status_code=503, detail=str(e))
        except WorkerError as e:
            timer.record_failure()
            logger.error(f"Worker error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            timer.record_failure()
            logger.error(f"Unexpected error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error")

    # V1 Chat completions endpoint
    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest, http_request: Request):
        """
        OpenAI-compatible chat completions endpoint.

        Args:
            request: Chat completion request
            http_request: FastAPI Request for headers

        Returns:
            Chat completion response
        """
        # P1: Rate limit check (before backpressure - fail fast if rate limited)
        rate_guard = RateLimitGuard(_get_rate_limiter())
        rate_response = rate_guard.check_rate_limit()
        if rate_response:
            return rate_response

        # 6.2: Generate unique request ID for tracing/correlation
        request_id = _generate_request_id()

        # Use priority queue or semaphore fallback
        if _is_queue_enabled():
            return await _chat_completions_with_queue(request, http_request, request_id, worker_manager, format_openai_error)
        else:
            return await _chat_completions_with_semaphore(request, request_id, worker_manager, format_openai_error)

    async def _chat_completions_with_queue(request, http_request, request_id, worker_manager, format_openai_error):
        """Process chat completion with priority queue."""
        priority = parse_priority(http_request)
        queue_guard = QueueGuard(_get_request_queue(), get_metrics())

        # Enqueue request
        error_response, queue_position = await queue_guard.enqueue(request_id, priority)
        if error_response:
            return error_response

        # Wait for slot
        try:
            timeout_response = await queue_guard.wait_for_slot()
            if timeout_response:
                return timeout_response

            # Record queue wait time
            wait_ms = queue_guard.wait_time_ms
            if wait_ms > 0:
                metrics_collector.record_queue_wait(wait_ms / 1000, priority.name.lower())

            # Process request
            response = await _process_chat_completion(request, request_id, worker_manager, format_openai_error)

            # Add queue headers to response if it's a JSONResponse
            if hasattr(response, 'headers'):
                response.headers["X-Queue-Wait-Ms"] = str(int(wait_ms))
                if queue_position and queue_position > 0:
                    response.headers["X-Queue-Position"] = str(queue_position)

            return response
        finally:
            await queue_guard.release()

    async def _chat_completions_with_semaphore(request, request_id, worker_manager, format_openai_error):
        """Process chat completion with semaphore backpressure (fallback)."""
        guard = BackpressureGuard(_get_request_semaphore(), get_metrics())
        capacity_response = guard.check_capacity()
        if capacity_response:
            return capacity_response

        async with guard.semaphore:
            return await _process_chat_completion(request, request_id, worker_manager, format_openai_error)

    async def _process_chat_completion(request, request_id, worker_manager, format_openai_error):
        """Core chat completion processing logic."""
        # P0: Track request timing using shared timer
        timer = RequestTimer(get_metrics())
        timer.start()

        # Convert chat messages to prompt using model's chat template
        logger.info(f"[{request_id}] Chat completion request: stream={request.stream}, messages={len(request.messages)}, "
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
            tokenizer = None  # Mark that we need to reload

        try:
            # P0: Ensure model loaded using shared helper
            ensure_model_loaded(worker_manager, request.model)

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
            # Extract JSON mode settings
            response_format_type = None
            json_schema = None
            if request.response_format:
                response_format_type = request.response_format.type
                if response_format_type == "json_schema":
                    json_schema = request.response_format.json_schema

            ipc_request = IPCCompletionRequest(
                model=request.model,
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
                stream=request.stream,
                images=images,  # Pass preprocessed images (None for text-only)
                response_format_type=response_format_type,
                json_schema=json_schema
            )

            # Generate
            if request.stream:
                # P0: Streaming using shared formatter
                # 6.2: Capture request_id in closure for streaming
                stream_request_id = request_id

                async def generate_stream():
                    """Generate SSE stream for chat completions."""
                    try:
                        for chunk in worker_manager.generate_stream(ipc_request):
                            if chunk.type == "stream_chunk":
                                # P0: Use ChatCompletionFormatter for SSE formatting
                                # 6.2: Pass request_id for tracing
                                yield ChatCompletionFormatter.format_stream_chunk(
                                    request.model, chunk, request_id=stream_request_id
                                )
                                if chunk.done:
                                    yield "data: [DONE]\n\n"
                                    break
                            elif chunk.type == "error":
                                yield f"data: {json.dumps(format_openai_error(chunk.message))}\n\n"
                                break
                    except Exception as e:
                        logger.error(f"[{stream_request_id}] Streaming error: {e}", exc_info=True)
                        yield f"data: {json.dumps(format_openai_error('Internal server error'))}\n\n"

                return StreamingResponse(generate_stream(), media_type="text/event-stream")
            else:
                # Non-streaming
                start_time = time.time()
                result = worker_manager.generate(ipc_request)
                generation_time = time.time() - start_time

                # Count prompt tokens using tokenizer (reload if template failed)
                try:
                    if tokenizer is None:
                        tokenizer = get_tokenizer(request.model)
                    prompt_tokens = len(tokenizer.encode(prompt))
                except Exception as e:
                    logger.warning(f"[{request_id}] Could not count prompt tokens: {e}")
                    prompt_tokens = 0

                # Log performance metrics
                completion_tokens = result["tokens"]
                tokens_per_sec = completion_tokens / generation_time if generation_time > 0 else 0
                logger.info(f"[{request_id}] Chat completion: {len(result['text'])} chars, {completion_tokens} tokens in {generation_time:.1f}s ({tokens_per_sec:.1f} tok/s)")
                logger.debug(f"[{request_id}] Response content: {result['text'][:200]}...")

                # P0: Track success and format using shared helpers
                # 6.2: Pass request_id for tracing
                timer.record_success()
                return ChatCompletionFormatter.format_non_streaming(
                    request.model, result, prompt_tokens, generation_time, request_id=request_id
                )

        except HTTPException:
            timer.record_failure()
            raise
        except ValueError as e:
            timer.record_failure()
            raise HTTPException(status_code=400, detail=str(e))
        except NoModelLoadedError as e:
            timer.record_failure()
            raise HTTPException(status_code=503, detail=str(e))
        except WorkerCommunicationError as e:
            timer.record_failure()
            logger.warning(f"Backpressure triggered: {e}")
            raise HTTPException(status_code=503, detail=str(e))
        except WorkerError as e:
            timer.record_failure()
            logger.error(f"Worker error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            timer.record_failure()
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
        version="3.2.0"
    )

    # 6.3: Include Prometheus metrics router
    app.include_router(metrics_router)

    # Initialize server info metric
    metrics_collector.set_server_info(version="3.2.0")

    @app.get("/admin/health")
    async def admin_health():
        """Admin health check with worker status."""
        health = worker_manager.health_check()
        return {
            "status": "healthy" if health["healthy"] else "degraded",
            "worker_status": health["status"],
            "version": "3.2.0"
        }

    @app.get("/admin/status")
    async def admin_status():
        """Get server and worker status."""
        status = worker_manager.get_status()
        health = worker_manager.health_check()

        return {
            "status": "running",
            "version": "3.2.0",
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

        # P1: Add rate limiter status
        stats["rate_limiter"] = _get_rate_limiter().get_status()

        # Add priority queue status if enabled
        if _is_queue_enabled():
            queue = _get_request_queue()
            queue_stats = queue.get_stats()
            stats["priority_queue"] = {
                "enabled": True,
                "active_slots": queue_stats.active_slots,
                "max_slots": queue_stats.max_slots,
                "queue_depth": queue_stats.queue_depth,
                "max_queue_depth": queue_stats.max_queue_depth,
                "waiting_by_priority": {
                    "high": queue_stats.waiting_high,
                    "normal": queue_stats.waiting_normal,
                    "low": queue_stats.waiting_low
                },
                "total_enqueued": queue_stats.total_enqueued,
                "total_timeouts": queue_stats.total_timeouts,
                "total_completed": queue_stats.total_completed,
                "avg_wait_time_ms": round(queue_stats.avg_wait_time_ms, 2)
            }
        else:
            stats["priority_queue"] = {"enabled": False}

        return stats

    @app.get("/admin/queue")
    async def admin_queue():
        """
        Get detailed priority queue status.

        Returns:
            Queue statistics including depth, wait times, and priority breakdown
        """
        if not _is_queue_enabled():
            return JSONResponse(
                status_code=200,
                content={
                    "enabled": False,
                    "message": "Priority queue is disabled. Using semaphore backpressure."
                }
            )

        queue = _get_request_queue()
        stats = queue.get_stats()

        return {
            "enabled": True,
            "active_slots": stats.active_slots,
            "max_slots": stats.max_slots,
            "queue_depth": stats.queue_depth,
            "max_queue_depth": stats.max_queue_depth,
            "utilization": stats.active_slots / stats.max_slots if stats.max_slots > 0 else 0,
            "waiting_by_priority": {
                "high": stats.waiting_high,
                "normal": stats.waiting_normal,
                "low": stats.waiting_low
            },
            "totals": {
                "enqueued": stats.total_enqueued,
                "completed": stats.total_completed,
                "timeouts": stats.total_timeouts
            },
            "avg_wait_time_ms": round(stats.avg_wait_time_ms, 2)
        }

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
