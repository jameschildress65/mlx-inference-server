"""
6.3: Prometheus metrics endpoint for MLX Server V3.

Provides standard Prometheus metrics for monitoring:
- Request counts by endpoint and status
- Request latency histograms
- Token generation counters
- Model and worker status gauges
- Queue depth and rate limiting status

Usage:
    from prometheus_metrics import metrics_router, record_request
    app.include_router(metrics_router)
"""

from prometheus_client import (
    Counter, Gauge, Histogram, Info,
    generate_latest, CONTENT_TYPE_LATEST,
    CollectorRegistry
)
from fastapi import Response, APIRouter
from typing import Optional
import time

# Create a custom registry to avoid conflicts with default registry
REGISTRY = CollectorRegistry()

# ============ Server Info ============
SERVER_INFO = Info(
    'mlx_server',
    'MLX inference server information',
    registry=REGISTRY
)

# ============ Request Metrics ============
REQUEST_COUNT = Counter(
    'mlx_requests_total',
    'Total requests processed',
    ['endpoint', 'status'],
    registry=REGISTRY
)

REQUEST_LATENCY = Histogram(
    'mlx_request_latency_seconds',
    'Request latency in seconds',
    ['endpoint'],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
    registry=REGISTRY
)

REQUEST_IN_PROGRESS = Gauge(
    'mlx_requests_in_progress',
    'Number of requests currently being processed',
    ['endpoint'],
    registry=REGISTRY
)

# ============ Token Metrics ============
TOKENS_GENERATED = Counter(
    'mlx_tokens_generated_total',
    'Total tokens generated',
    ['model'],
    registry=REGISTRY
)

TOKENS_PER_SECOND = Gauge(
    'mlx_tokens_per_second',
    'Current token generation rate',
    ['model'],
    registry=REGISTRY
)

# ============ Model Metrics ============
MODEL_LOADED = Gauge(
    'mlx_model_loaded',
    'Whether a model is currently loaded (1=yes, 0=no)',
    registry=REGISTRY
)

MODEL_MEMORY_GB = Gauge(
    'mlx_model_memory_gb',
    'Memory used by loaded model in GB',
    ['model'],
    registry=REGISTRY
)

# ============ Worker Metrics ============
WORKER_ALIVE = Gauge(
    'mlx_worker_alive',
    'Whether worker process is alive (1=yes, 0=no)',
    registry=REGISTRY
)

ACTIVE_REQUESTS = Gauge(
    'mlx_active_requests',
    'Number of active requests in queue',
    registry=REGISTRY
)

# ============ Queue/Backpressure Metrics ============
QUEUE_FULL_REJECTS = Counter(
    'mlx_queue_full_rejects_total',
    'Total requests rejected due to full queue (503)',
    registry=REGISTRY
)

QUEUE_DEPTH = Gauge(
    'mlx_queue_depth',
    'Current queue depth',
    registry=REGISTRY
)

QUEUE_MAX_DEPTH = Gauge(
    'mlx_queue_max_depth',
    'Maximum queue depth',
    registry=REGISTRY
)

# Priority queue specific metrics
QUEUE_WAIT_TIME = Histogram(
    'mlx_queue_wait_seconds',
    'Time spent waiting in queue',
    ['priority'],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0),
    registry=REGISTRY
)

QUEUE_TIMEOUTS = Counter(
    'mlx_queue_timeouts_total',
    'Requests that timed out while waiting in queue',
    ['priority'],
    registry=REGISTRY
)

QUEUE_WAITING_BY_PRIORITY = Gauge(
    'mlx_queue_waiting',
    'Requests waiting in queue by priority',
    ['priority'],
    registry=REGISTRY
)

# ============ Rate Limiting Metrics ============
RATE_LIMIT_REJECTS = Counter(
    'mlx_rate_limit_rejects_total',
    'Total requests rejected due to rate limiting (429)',
    registry=REGISTRY
)

RATE_LIMIT_TOKENS = Gauge(
    'mlx_rate_limit_tokens',
    'Current token bucket level',
    registry=REGISTRY
)


class MetricsCollector:
    """
    Centralized metrics collection for MLX Server.

    Thread-safe via atomic operations on prometheus_client metrics.
    Uses lock for _start_times dict to handle concurrent async requests.
    """

    def __init__(self):
        import threading
        self._start_times: dict[str, float] = {}
        self._lock = threading.Lock()  # Protects _start_times dict

    def set_server_info(self, version: str, model: Optional[str] = None):
        """Set server info metric."""
        SERVER_INFO.info({
            'version': version,
            'model': model or 'none'
        })

    def record_request_start(self, endpoint: str, request_id: str) -> None:
        """Record request start with provided request_id for timing.

        Args:
            endpoint: API endpoint (e.g., 'completions', 'chat')
            request_id: Unique request ID from api.py (req-<uuid4> format)
        """
        with self._lock:
            self._start_times[request_id] = time.time()
        REQUEST_IN_PROGRESS.labels(endpoint=endpoint).inc()

    def record_request_end(
        self,
        request_id: str,
        endpoint: str,
        status: str,
        tokens: int = 0,
        model: Optional[str] = None
    ):
        """Record request completion with timing and token count."""
        REQUEST_IN_PROGRESS.labels(endpoint=endpoint).dec()
        REQUEST_COUNT.labels(endpoint=endpoint, status=status).inc()

        # Record latency
        with self._lock:
            start_time = self._start_times.pop(request_id, None)
        if start_time:
            latency = time.time() - start_time
            REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)

            # Record tokens per second if applicable
            if tokens > 0 and model and latency > 0:
                TOKENS_GENERATED.labels(model=model).inc(tokens)
                TOKENS_PER_SECOND.labels(model=model).set(tokens / latency)

    def record_queue_reject(self):
        """Record a 503 rejection due to full queue."""
        QUEUE_FULL_REJECTS.inc()

    def record_rate_limit_reject(self):
        """Record a 429 rejection due to rate limiting."""
        RATE_LIMIT_REJECTS.inc()

    def update_model_status(
        self,
        loaded: bool,
        model_name: Optional[str] = None,
        memory_gb: float = 0.0
    ):
        """Update model loaded status."""
        MODEL_LOADED.set(1 if loaded else 0)
        if model_name:
            MODEL_MEMORY_GB.labels(model=model_name).set(memory_gb)

    def update_worker_status(self, alive: bool, active_requests: int = 0):
        """Update worker status gauges."""
        WORKER_ALIVE.set(1 if alive else 0)
        ACTIVE_REQUESTS.set(active_requests)

    def update_queue_status(self, depth: int, max_depth: int):
        """Update queue depth gauges."""
        QUEUE_DEPTH.set(depth)
        QUEUE_MAX_DEPTH.set(max_depth)

    def update_rate_limit_tokens(self, tokens: float):
        """Update rate limiter token bucket level."""
        RATE_LIMIT_TOKENS.set(tokens)

    def record_queue_wait(self, wait_seconds: float, priority: str):
        """Record queue wait time for a request."""
        QUEUE_WAIT_TIME.labels(priority=priority).observe(wait_seconds)

    def record_queue_timeout(self, priority: str):
        """Record a queue timeout."""
        QUEUE_TIMEOUTS.labels(priority=priority).inc()

    def update_queue_waiting_by_priority(self, high: int, normal: int, low: int):
        """Update queue waiting counts by priority."""
        QUEUE_WAITING_BY_PRIORITY.labels(priority='high').set(high)
        QUEUE_WAITING_BY_PRIORITY.labels(priority='normal').set(normal)
        QUEUE_WAITING_BY_PRIORITY.labels(priority='low').set(low)


# Global collector instance
metrics_collector = MetricsCollector()


# ============ FastAPI Router ============
metrics_router = APIRouter()


@metrics_router.get("/metrics")
async def prometheus_metrics():
    """
    6.3: Prometheus metrics endpoint.

    Returns metrics in Prometheus text exposition format.
    Compatible with Prometheus, Grafana, and other monitoring tools.
    """
    return Response(
        content=generate_latest(REGISTRY),
        media_type=CONTENT_TYPE_LATEST
    )
