"""Request handling infrastructure for MLX Server API endpoints.

Provides shared utilities for backpressure control, metrics tracking,
and model loading across completion endpoints.
"""

import time
import logging
from typing import Optional
from asyncio import Semaphore
from fastapi.responses import JSONResponse

from .worker_manager import WorkerManager
from .rate_limiter import RateLimiter

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
