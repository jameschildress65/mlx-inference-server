"""Request handling infrastructure for MLX Server API endpoints.

Provides shared utilities for backpressure control, metrics tracking,
and model loading across completion endpoints.
"""

import time
import logging
from typing import Optional, Tuple
from asyncio import Semaphore
from fastapi import Request
from fastapi.responses import JSONResponse

from .worker_manager import WorkerManager
from .rate_limiter import RateLimiter
from .priority_queue import (
    PriorityRequestQueue, Priority, QueuedRequest,
    QueueTimeoutError, QueueFullError
)

logger = logging.getLogger(__name__)


class RateLimitGuard:
    """Handles rate limiting for incoming requests.

    Provides HTTP 429 response with Retry-After header when rate limit exceeded.
    Disabled by default for home lab use - enable via MLX_RATE_LIMIT_ENABLED=1.

    Args:
        rate_limiter: RateLimiter instance for token bucket checks
    """

    def __init__(self, rate_limiter: RateLimiter):
        self.rate_limiter = rate_limiter

    def check_rate_limit(self) -> Optional[JSONResponse]:
        """Check if request is allowed under rate limit.

        Returns:
            JSONResponse with 429 if rate limited, None if allowed
        """
        allowed, retry_after = self.rate_limiter.check()
        if not allowed:
            return JSONResponse(
                status_code=429,
                headers={"Retry-After": str(int(retry_after) + 1)},
                content={
                    "error": {
                        "message": f"Rate limit exceeded. Please retry after {retry_after:.1f} seconds.",
                        "type": "rate_limit_exceeded",
                        "code": "rate_limited"
                    }
                }
            )
        return None


class BackpressureGuard:
    """Handles backpressure control and queue management.

    Provides fast-fail optimization for request queue capacity checks.
    Returns HTTP 503 when queue is full instead of blocking/crashing.

    Args:
        semaphore: AsyncIO semaphore controlling concurrent request limit
        metrics: RequestMetrics instance for tracking queue rejections
    """

    def __init__(self, semaphore: Semaphore, metrics):
        self.semaphore = semaphore
        self.metrics = metrics

    def check_capacity(self) -> Optional[JSONResponse]:
        """Check if server has capacity for new request.

        Fast-fail optimization: Returns 503 immediately if queue full
        instead of blocking. Small race window exists but is benign
        (worst case: 1 extra request waits instead of failing).

        Returns:
            JSONResponse with 503 if queue full, None if capacity available
        """
        if self.semaphore.locked() and self.semaphore._value == 0:
            logger.warning("Request queue full - returning 503")
            self.metrics.record_queue_reject()
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
        return None


def parse_priority(http_request: Request) -> Priority:
    """Parse priority from X-Priority header.

    Args:
        http_request: FastAPI Request object

    Returns:
        Priority enum value (defaults to NORMAL)
    """
    header_value = http_request.headers.get("X-Priority", "normal").lower()
    if header_value == "high":
        return Priority.HIGH
    elif header_value == "low":
        return Priority.LOW
    else:
        return Priority.NORMAL


class QueueGuard:
    """Handles priority queue management for incoming requests.

    Replaces BackpressureGuard with proper queuing instead of immediate 503.
    Provides wait-with-timeout semantics and queue position tracking.

    Args:
        queue: PriorityRequestQueue instance
        metrics: RequestMetrics instance for tracking
    """

    def __init__(self, queue: PriorityRequestQueue, metrics):
        self.queue = queue
        self.metrics = metrics
        self._entry: Optional[QueuedRequest] = None
        self._request_id: Optional[str] = None
        self._priority: Priority = Priority.NORMAL

    async def enqueue(
        self,
        request_id: str,
        priority: Priority = Priority.NORMAL
    ) -> Tuple[Optional[JSONResponse], Optional[int]]:
        """Enqueue request and return error response if queue full.

        Args:
            request_id: Unique request identifier
            priority: Request priority level

        Returns:
            Tuple of (error_response, queue_position)
            - error_response: JSONResponse if rejected, None if queued
            - queue_position: Position in queue (0 = immediate, None if rejected)
        """
        self._request_id = request_id
        self._priority = priority

        try:
            self._entry = await self.queue.enqueue(request_id, priority)

            # Get position (0 = got slot immediately, 1+ = waiting)
            position = await self.queue.get_position(request_id)
            return None, position

        except QueueFullError as e:
            logger.warning(f"[{request_id}] Queue full: {e.queue_depth} >= {e.threshold}")
            self.metrics.record_queue_reject()
            return JSONResponse(
                status_code=503,
                headers={"Retry-After": "30"},
                content={
                    "error": {
                        "message": "Server queue is full. Please retry after a brief delay.",
                        "type": "server_overloaded",
                        "code": "queue_full"
                    }
                }
            ), None

    async def wait_for_slot(self) -> Optional[JSONResponse]:
        """Wait for processing slot to become available.

        Returns:
            JSONResponse with 503 if timeout, None if slot acquired
        """
        if self._entry is None:
            return None

        try:
            await self.queue.wait_for_slot(self._entry)
            return None

        except QueueTimeoutError as e:
            logger.warning(f"[{self._request_id}] Queue timeout: {e.wait_time_ms:.0f}ms")
            from .prometheus_metrics import metrics_collector
            metrics_collector.record_queue_timeout(self._priority.name.lower())

            return JSONResponse(
                status_code=503,
                headers={
                    "Retry-After": "30",
                    "X-Queue-Position": str(e.position),
                    "X-Queue-Wait-Ms": str(int(e.wait_time_ms))
                },
                content={
                    "error": {
                        "message": f"Request timed out waiting in queue after {e.wait_time_ms:.0f}ms",
                        "type": "queue_timeout",
                        "code": "queue_timeout"
                    }
                }
            )

    async def release(self) -> None:
        """Release slot on request completion."""
        if self._request_id:
            await self.queue.release_slot(self._request_id)

    @property
    def wait_time_ms(self) -> float:
        """Get time spent waiting in queue (milliseconds)."""
        if self._entry:
            return self._entry.wait_time_ms
        return 0.0


class RequestTimer:
    """Tracks request timing for metrics.

    Provides centralized timing and metrics recording for request lifecycle.
    Ensures metrics are properly tracked even on error paths.

    Args:
        metrics: RequestMetrics instance for recording timing data
    """

    def __init__(self, metrics):
        self.metrics = metrics
        self.start_time = None

    def start(self):
        """Start timing a request and record start event."""
        self.start_time = time.time()
        self.metrics.record_request_start()

    def record_success(self):
        """Record successful completion with duration."""
        if self.start_time:
            duration_ms = (time.time() - self.start_time) * 1000
            self.metrics.record_request_success(duration_ms)

    def record_failure(self):
        """Record failure with duration."""
        if self.start_time:
            duration_ms = (time.time() - self.start_time) * 1000
            self.metrics.record_request_failure(duration_ms)


def ensure_model_loaded(worker_manager: WorkerManager, model_name: str) -> None:
    """Ensure requested model is loaded, load on-demand if needed.

    Checks if the requested model is currently loaded and active.
    If not, triggers model loading via worker manager. Logs model
    load operations for observability.

    Args:
        worker_manager: WorkerManager instance
        model_name: Model to ensure is loaded

    Raises:
        Same exceptions as worker_manager.load_model() - typically
        WorkerError, NoModelLoadedError, or model loading failures
    """
    status = worker_manager.get_status()
    if not status["model_loaded"] or status["model_name"] != model_name:
        logger.info(f"Loading model on-demand: {model_name}")
        load_result = worker_manager.load_model(model_name)
        logger.info(f"Model loaded: {load_result.model_name} ({load_result.memory_gb:.2f} GB)")
