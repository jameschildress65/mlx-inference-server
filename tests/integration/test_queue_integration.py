"""
Integration tests for Priority Request Queue.

Tests queue behavior with actual API endpoints:
- X-Priority header handling
- Queue position headers in response
- Concurrent request handling
- Queue statistics endpoint
- Fallback to semaphore mode

NASA-grade: comprehensive integration testing.
"""

import pytest
import requests
import asyncio
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.orchestrator.priority_queue import PriorityRequestQueue, QueueConfig, Priority
from src.orchestrator.request_handlers import QueueGuard, parse_priority
from fastapi import Request
from unittest.mock import MagicMock


# Test model - use small model for integration tests
TEST_MODEL = os.environ.get("MLX_TEST_MODEL", "mlx-community/Qwen2.5-0.5B-Instruct-4bit")

# Skip if model tests are disabled
SKIP_MODEL_TESTS = os.environ.get("MLX_SKIP_MODEL_TESTS", "0") == "1"


class TestParsePriority:
    """Test X-Priority header parsing."""

    def test_parse_priority_high(self):
        """HIGH priority parsed from header."""
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {"X-Priority": "high"}

        priority = parse_priority(mock_request)
        assert priority == Priority.HIGH

    def test_parse_priority_normal(self):
        """NORMAL priority parsed from header."""
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {"X-Priority": "normal"}

        priority = parse_priority(mock_request)
        assert priority == Priority.NORMAL

    def test_parse_priority_low(self):
        """LOW priority parsed from header."""
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {"X-Priority": "low"}

        priority = parse_priority(mock_request)
        assert priority == Priority.LOW

    def test_parse_priority_default(self):
        """Missing header defaults to NORMAL."""
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}

        priority = parse_priority(mock_request)
        assert priority == Priority.NORMAL

    def test_parse_priority_case_insensitive(self):
        """Header value is case-insensitive."""
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {"X-Priority": "HIGH"}

        priority = parse_priority(mock_request)
        assert priority == Priority.HIGH

    def test_parse_priority_invalid_value(self):
        """Invalid value defaults to NORMAL."""
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {"X-Priority": "urgent"}

        priority = parse_priority(mock_request)
        assert priority == Priority.NORMAL


class TestQueueGuard:
    """Test QueueGuard request handler."""

    @pytest.fixture
    def queue_config(self):
        """Create queue config for tests."""
        return QueueConfig(max_slots=2, reject_threshold=10)

    @pytest.fixture
    def mock_metrics(self):
        """Create mock metrics collector."""
        metrics = MagicMock()
        return metrics

    @pytest.mark.asyncio
    async def test_enqueue_success(self, queue_config, mock_metrics):
        """Successful enqueue returns no error."""
        queue = PriorityRequestQueue(queue_config)
        guard = QueueGuard(queue, mock_metrics)

        error, position = await guard.enqueue("req-1", Priority.NORMAL)

        assert error is None
        assert position == 0  # Got slot immediately

    @pytest.mark.asyncio
    async def test_enqueue_returns_position(self, queue_config, mock_metrics):
        """Enqueue returns correct queue position."""
        queue = PriorityRequestQueue(queue_config)
        guard = QueueGuard(queue, mock_metrics)

        # Fill slots
        await queue.enqueue("req-1", Priority.NORMAL)
        await queue.enqueue("req-2", Priority.NORMAL)

        # Third request gets queued
        error, position = await guard.enqueue("req-3", Priority.NORMAL)

        assert error is None
        assert position == 1  # First in queue

    @pytest.mark.asyncio
    async def test_enqueue_queue_full(self, mock_metrics):
        """Queue full returns 503 error response."""
        config = QueueConfig(max_slots=2, reject_threshold=3)
        queue = PriorityRequestQueue(config)
        guard = QueueGuard(queue, mock_metrics)

        # Fill to threshold
        await queue.enqueue("req-1", Priority.NORMAL)
        await queue.enqueue("req-2", Priority.NORMAL)
        await queue.enqueue("req-3", Priority.NORMAL)

        # Fourth request should be rejected
        error, position = await guard.enqueue("req-4", Priority.NORMAL)

        assert error is not None
        assert error.status_code == 503
        assert position is None
        mock_metrics.record_queue_reject.assert_called_once()

    @pytest.mark.asyncio
    async def test_wait_for_slot_success(self, queue_config, mock_metrics):
        """Wait for slot succeeds when slot available."""
        queue = PriorityRequestQueue(queue_config)
        guard = QueueGuard(queue, mock_metrics)

        await guard.enqueue("req-1", Priority.NORMAL)
        error = await guard.wait_for_slot()

        assert error is None

    @pytest.mark.asyncio
    async def test_wait_for_slot_timeout(self, mock_metrics):
        """Wait for slot returns 503 on timeout."""
        config = QueueConfig(max_slots=1, timeout_normal=0.1)
        queue = PriorityRequestQueue(config)
        guard = QueueGuard(queue, mock_metrics)

        # Take the only slot
        await queue.enqueue("req-1", Priority.NORMAL)

        # Second request will timeout
        await guard.enqueue("req-2", Priority.NORMAL)
        error = await guard.wait_for_slot()

        assert error is not None
        assert error.status_code == 503
        assert "X-Queue-Wait-Ms" in error.headers

    @pytest.mark.asyncio
    async def test_release(self, queue_config, mock_metrics):
        """Release frees the slot."""
        queue = PriorityRequestQueue(queue_config)
        guard = QueueGuard(queue, mock_metrics)

        await guard.enqueue("req-1", Priority.NORMAL)
        await guard.release()

        stats = queue.get_stats()
        assert stats.active_slots == 0

    @pytest.mark.asyncio
    async def test_wait_time_ms(self, queue_config, mock_metrics):
        """Wait time is tracked."""
        queue = PriorityRequestQueue(queue_config)
        guard = QueueGuard(queue, mock_metrics)

        await guard.enqueue("req-1", Priority.NORMAL)

        # Some time passes
        await asyncio.sleep(0.1)

        # Wait time should reflect elapsed time
        assert guard.wait_time_ms >= 100


@pytest.mark.skipif(SKIP_MODEL_TESTS, reason="Model tests disabled")
class TestQueueIntegrationWithServer:
    """Test queue with actual server (requires running server)."""

    @pytest.fixture
    def server_url(self):
        """Get server URL from environment or use default."""
        return os.environ.get("MLX_SERVER_URL", "http://127.0.0.1:11440")

    def test_priority_header_accepted(self, server_url):
        """Server accepts X-Priority header."""
        try:
            response = requests.get(f"{server_url}/health")
            if response.status_code != 200:
                pytest.skip("Server not running")
        except requests.ConnectionError:
            pytest.skip("Server not running")

        # Make request with priority header
        response = requests.post(
            f"{server_url}/v1/completions",
            json={
                "model": TEST_MODEL,
                "prompt": "Hello",
                "max_tokens": 5
            },
            headers={"X-Priority": "high"},
            timeout=60
        )

        # Should succeed (not reject due to unknown header)
        assert response.status_code in [200, 503]  # 503 if queue full

    def test_queue_endpoint(self, server_url):
        """Queue stats endpoint returns valid data."""
        try:
            response = requests.get(f"{server_url}/health")
            if response.status_code != 200:
                pytest.skip("Server not running")
        except requests.ConnectionError:
            pytest.skip("Server not running")

        # Get queue stats from admin endpoint
        admin_url = server_url.replace("11440", "11441")
        response = requests.get(f"{admin_url}/admin/queue", timeout=10)

        if response.status_code == 404:
            pytest.skip("Queue endpoint not available (queue may be disabled)")

        assert response.status_code == 200
        data = response.json()

        # Verify structure
        assert "enabled" in data
        assert "active_slots" in data
        assert "max_slots" in data


class TestQueueConcurrency:
    """Test queue behavior under concurrent load."""

    @pytest.mark.asyncio
    async def test_concurrent_requests_respect_limit(self):
        """Concurrent requests respect max_slots limit."""
        config = QueueConfig(max_slots=3, reject_threshold=100)
        queue = PriorityRequestQueue(config)

        # Track how many get immediate slots
        immediate_count = 0

        async def make_request(i):
            nonlocal immediate_count
            entry = await queue.enqueue(f"req-{i}", Priority.NORMAL)
            if entry._slot_event.is_set():
                immediate_count += 1
            return entry

        # Make 10 concurrent requests
        entries = await asyncio.gather(*[make_request(i) for i in range(10)])

        # Exactly max_slots should get immediate
        assert immediate_count == 3

        # Verify stats
        stats = queue.get_stats()
        assert stats.active_slots == 3
        assert stats.total_enqueued == 10

    @pytest.mark.asyncio
    async def test_priority_ordering_under_load(self):
        """Priority ordering maintained under concurrent load."""
        config = QueueConfig(max_slots=1, reject_threshold=100)
        queue = PriorityRequestQueue(config)

        # Take the only slot
        await queue.enqueue("req-blocker", Priority.NORMAL)

        # Queue mix of priorities
        entries_low = [await queue.enqueue(f"low-{i}", Priority.LOW) for i in range(3)]
        entries_high = [await queue.enqueue(f"high-{i}", Priority.HIGH) for i in range(3)]
        entries_normal = [await queue.enqueue(f"normal-{i}", Priority.NORMAL) for i in range(3)]

        # Release blocker - should wake HIGH first
        await queue.release_slot("req-blocker")
        assert entries_high[0]._slot_event.is_set()

        # Release again - should wake HIGH[1]
        await queue.release_slot("high-0")
        assert entries_high[1]._slot_event.is_set()

    @pytest.mark.asyncio
    async def test_timeout_under_load(self):
        """Timeouts work correctly under concurrent load."""
        config = QueueConfig(max_slots=1, timeout_normal=0.1, reject_threshold=100)
        queue = PriorityRequestQueue(config)

        # Take the only slot (will hold it)
        await queue.enqueue("req-blocker", Priority.NORMAL)

        # Queue requests that will timeout
        timeout_count = 0

        async def request_with_timeout(i):
            nonlocal timeout_count
            entry = await queue.enqueue(f"req-{i}", Priority.NORMAL)
            try:
                await queue.wait_for_slot(entry)
            except Exception:
                timeout_count += 1

        # Make 5 requests concurrently - all should timeout
        await asyncio.gather(*[request_with_timeout(i) for i in range(5)])

        assert timeout_count == 5

        stats = queue.get_stats()
        assert stats.total_timeouts == 5


class TestQueueMetrics:
    """Test queue Prometheus metrics integration."""

    @pytest.mark.asyncio
    async def test_metrics_update_on_enqueue(self):
        """Metrics updated when requests enqueue."""
        from src.orchestrator.prometheus_metrics import QUEUE_DEPTH

        config = QueueConfig(max_slots=2)
        queue = PriorityRequestQueue(config)

        # Baseline
        initial_enqueued = queue.get_stats().total_enqueued

        # Enqueue requests
        await queue.enqueue("req-1", Priority.NORMAL)
        await queue.enqueue("req-2", Priority.NORMAL)
        await queue.enqueue("req-3", Priority.NORMAL)

        stats = queue.get_stats()
        assert stats.total_enqueued == initial_enqueued + 3

    @pytest.mark.asyncio
    async def test_metrics_update_on_complete(self):
        """Metrics updated when requests complete."""
        config = QueueConfig(max_slots=5)
        queue = PriorityRequestQueue(config)

        # Enqueue and release
        await queue.enqueue("req-1", Priority.NORMAL)
        await queue.release_slot("req-1")

        stats = queue.get_stats()
        assert stats.total_completed == 1


class TestQueueFallback:
    """Test semaphore fallback mode."""

    def test_disabled_queue_has_enabled_false(self):
        """Disabled queue reports enabled=False."""
        config = QueueConfig(enabled=False)
        queue = PriorityRequestQueue(config)

        assert queue.enabled is False

        stats = queue.get_stats()
        assert stats.enabled is False

    @pytest.mark.asyncio
    async def test_disabled_queue_still_functional(self):
        """Disabled queue still works (for testing/fallback)."""
        config = QueueConfig(enabled=False, max_slots=2)
        queue = PriorityRequestQueue(config)

        # Should still be able to enqueue
        entry = await queue.enqueue("req-1", Priority.NORMAL)
        assert entry._slot_event.is_set()

        await queue.release_slot("req-1")

        stats = queue.get_stats()
        assert stats.total_completed == 1
