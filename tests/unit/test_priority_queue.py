"""
Unit tests for Priority Request Queue.

Tests queue operations, priority ordering, timeout handling, and drain logic.
NASA-grade: comprehensive coverage of all edge cases.
"""

import pytest
import asyncio
import time
from src.orchestrator.priority_queue import (
    Priority, QueueConfig, QueuedRequest, QueueStats,
    PriorityRequestQueue, QueueTimeoutError, QueueFullError
)


class TestPriorityEnum:
    """Test Priority enum ordering."""

    def test_priority_ordering(self):
        """Lower numeric value = higher priority."""
        assert Priority.HIGH < Priority.NORMAL < Priority.LOW
        assert Priority.HIGH == 0
        assert Priority.NORMAL == 1
        assert Priority.LOW == 2

    def test_priority_comparable(self):
        """Priority values are comparable."""
        priorities = [Priority.LOW, Priority.HIGH, Priority.NORMAL]
        sorted_priorities = sorted(priorities)
        assert sorted_priorities == [Priority.HIGH, Priority.NORMAL, Priority.LOW]


class TestQueueConfig:
    """Test QueueConfig defaults and validation."""

    def test_default_values(self):
        """Default config has sensible values."""
        config = QueueConfig()
        assert config.max_slots == 10
        assert config.max_queue_depth == 50
        assert config.reject_threshold == 100
        assert config.timeout_high == 120.0
        assert config.timeout_normal == 60.0
        assert config.timeout_low == 30.0
        assert config.enabled is True

    def test_custom_values(self):
        """Custom config values are respected."""
        config = QueueConfig(
            max_slots=5,
            max_queue_depth=20,
            reject_threshold=50,
            timeout_high=180.0,
            timeout_normal=90.0,
            timeout_low=45.0,
            enabled=False
        )
        assert config.max_slots == 5
        assert config.max_queue_depth == 20
        assert config.reject_threshold == 50
        assert config.timeout_high == 180.0
        assert config.timeout_normal == 90.0
        assert config.timeout_low == 45.0
        assert config.enabled is False


class TestQueuedRequest:
    """Test QueuedRequest ordering and properties."""

    def test_ordering_by_priority(self):
        """Requests are ordered by priority first."""
        req_high = QueuedRequest(
            priority=Priority.HIGH,
            insertion_order=10,
            request_id="high"
        )
        req_low = QueuedRequest(
            priority=Priority.LOW,
            insertion_order=1,
            request_id="low"
        )

        # High priority (0) < Low priority (2), regardless of insertion order
        assert req_high < req_low

    def test_ordering_within_same_priority(self):
        """Within same priority, FIFO by insertion order."""
        req1 = QueuedRequest(
            priority=Priority.NORMAL,
            insertion_order=1,
            request_id="first"
        )
        req2 = QueuedRequest(
            priority=Priority.NORMAL,
            insertion_order=2,
            request_id="second"
        )

        assert req1 < req2

    def test_wait_time_ms(self):
        """Wait time is calculated correctly."""
        req = QueuedRequest(
            priority=Priority.NORMAL,
            insertion_order=0,
            request_id="test"
        )

        # Wait a short time
        time.sleep(0.1)
        wait_ms = req.wait_time_ms

        assert wait_ms >= 100  # At least 100ms
        assert wait_ms < 200   # Less than 200ms

    def test_request_id_not_in_comparison(self):
        """Request ID does not affect ordering."""
        req_a = QueuedRequest(
            priority=Priority.NORMAL,
            insertion_order=1,
            request_id="zzz"
        )
        req_b = QueuedRequest(
            priority=Priority.NORMAL,
            insertion_order=2,
            request_id="aaa"
        )

        # Same ordering despite different request_ids
        assert req_a < req_b


class TestPriorityRequestQueueBasic:
    """Test basic queue operations."""

    @pytest.mark.asyncio
    async def test_immediate_slot_granted(self):
        """Request gets immediate slot when available."""
        config = QueueConfig(max_slots=5)
        queue = PriorityRequestQueue(config)

        entry = await queue.enqueue("req-1", Priority.NORMAL)

        # Should have slot immediately
        assert entry._slot_event.is_set()
        position = await queue.get_position("req-1")
        assert position == 0  # 0 = processing

    @pytest.mark.asyncio
    async def test_queued_when_slots_full(self):
        """Request is queued when all slots are taken."""
        config = QueueConfig(max_slots=2)
        queue = PriorityRequestQueue(config)

        # Fill slots
        entry1 = await queue.enqueue("req-1", Priority.NORMAL)
        entry2 = await queue.enqueue("req-2", Priority.NORMAL)

        # Third request should be queued
        entry3 = await queue.enqueue("req-3", Priority.NORMAL)

        assert entry1._slot_event.is_set()
        assert entry2._slot_event.is_set()
        assert not entry3._slot_event.is_set()

        position = await queue.get_position("req-3")
        assert position == 1  # First in queue

    @pytest.mark.asyncio
    async def test_wait_for_slot_returns_immediately_if_set(self):
        """wait_for_slot returns immediately when slot is already granted."""
        config = QueueConfig(max_slots=5)
        queue = PriorityRequestQueue(config)

        entry = await queue.enqueue("req-1", Priority.NORMAL)
        result = await queue.wait_for_slot(entry)

        assert result is True

    @pytest.mark.asyncio
    async def test_release_slot_wakes_waiter(self):
        """Releasing a slot wakes the highest-priority waiter."""
        config = QueueConfig(max_slots=1)
        queue = PriorityRequestQueue(config)

        # Take the only slot
        entry1 = await queue.enqueue("req-1", Priority.NORMAL)
        assert entry1._slot_event.is_set()

        # Queue a waiter
        entry2 = await queue.enqueue("req-2", Priority.NORMAL)
        assert not entry2._slot_event.is_set()

        # Release the slot
        await queue.release_slot("req-1")

        # Waiter should now have slot
        assert entry2._slot_event.is_set()


class TestPriorityRequestQueuePriorities:
    """Test priority ordering in queue."""

    @pytest.mark.asyncio
    async def test_high_priority_served_first(self):
        """HIGH priority requests are served before NORMAL."""
        config = QueueConfig(max_slots=1)
        queue = PriorityRequestQueue(config)

        # Take the only slot
        entry1 = await queue.enqueue("req-1", Priority.NORMAL)

        # Queue in reverse priority order
        entry_low = await queue.enqueue("req-low", Priority.LOW)
        entry_normal = await queue.enqueue("req-normal", Priority.NORMAL)
        entry_high = await queue.enqueue("req-high", Priority.HIGH)

        # Release the slot
        await queue.release_slot("req-1")

        # HIGH should get the slot
        assert entry_high._slot_event.is_set()
        assert not entry_normal._slot_event.is_set()
        assert not entry_low._slot_event.is_set()

    @pytest.mark.asyncio
    async def test_fifo_within_same_priority(self):
        """FIFO ordering within same priority level."""
        config = QueueConfig(max_slots=1)
        queue = PriorityRequestQueue(config)

        # Take the only slot
        entry0 = await queue.enqueue("req-0", Priority.NORMAL)

        # Queue several NORMAL priority requests
        entry1 = await queue.enqueue("req-1", Priority.NORMAL)
        entry2 = await queue.enqueue("req-2", Priority.NORMAL)
        entry3 = await queue.enqueue("req-3", Priority.NORMAL)

        # Release slots in order - should be FIFO
        await queue.release_slot("req-0")
        assert entry1._slot_event.is_set()
        assert not entry2._slot_event.is_set()

        await queue.release_slot("req-1")
        assert entry2._slot_event.is_set()
        assert not entry3._slot_event.is_set()


class TestPriorityRequestQueueTimeout:
    """Test queue timeout behavior."""

    @pytest.mark.asyncio
    async def test_timeout_raises_queue_timeout_error(self):
        """Timeout raises QueueTimeoutError."""
        config = QueueConfig(max_slots=1, timeout_normal=0.1)
        queue = PriorityRequestQueue(config)

        # Take the only slot
        entry1 = await queue.enqueue("req-1", Priority.NORMAL)

        # Queue a waiter with short timeout
        entry2 = await queue.enqueue("req-2", Priority.NORMAL)

        with pytest.raises(QueueTimeoutError) as exc_info:
            await queue.wait_for_slot(entry2)

        assert exc_info.value.request_id == "req-2"
        assert exc_info.value.wait_time_ms >= 100

    @pytest.mark.asyncio
    async def test_timeout_uses_priority_default(self):
        """Timeout uses correct priority-specific default."""
        config = QueueConfig(
            max_slots=1,
            timeout_high=0.3,
            timeout_normal=0.2,
            timeout_low=0.1
        )
        queue = PriorityRequestQueue(config)

        # Take the only slot
        await queue.enqueue("req-1", Priority.NORMAL)

        # Queue LOW priority - should use 0.1s timeout
        entry_low = await queue.enqueue("req-low", Priority.LOW)

        start = time.time()
        with pytest.raises(QueueTimeoutError):
            await queue.wait_for_slot(entry_low)
        elapsed = time.time() - start

        assert elapsed >= 0.1
        assert elapsed < 0.2  # Should not use NORMAL timeout

    @pytest.mark.asyncio
    async def test_custom_timeout_override(self):
        """Custom timeout overrides priority default."""
        config = QueueConfig(max_slots=1, timeout_normal=10.0)
        queue = PriorityRequestQueue(config)

        # Take the only slot
        await queue.enqueue("req-1", Priority.NORMAL)

        # Queue with custom timeout
        entry2 = await queue.enqueue("req-2", Priority.NORMAL, timeout=0.1)

        with pytest.raises(QueueTimeoutError):
            await queue.wait_for_slot(entry2)


class TestPriorityRequestQueueFull:
    """Test queue full behavior."""

    @pytest.mark.asyncio
    async def test_reject_threshold_raises_error(self):
        """Exceeding reject threshold raises QueueFullError."""
        config = QueueConfig(max_slots=2, reject_threshold=5)
        queue = PriorityRequestQueue(config)

        # Fill to threshold
        for i in range(5):
            await queue.enqueue(f"req-{i}", Priority.NORMAL)

        # Next request should be rejected
        with pytest.raises(QueueFullError) as exc_info:
            await queue.enqueue("req-overflow", Priority.NORMAL)

        assert exc_info.value.queue_depth >= 5
        assert exc_info.value.threshold == 5

    @pytest.mark.asyncio
    async def test_reject_during_draining(self):
        """Requests rejected during drain."""
        config = QueueConfig(max_slots=2)
        queue = PriorityRequestQueue(config)

        # Start draining
        queue._draining = True

        # New requests should be rejected
        with pytest.raises(QueueFullError):
            await queue.enqueue("req-new", Priority.NORMAL)


class TestPriorityRequestQueueCancel:
    """Test request cancellation."""

    @pytest.mark.asyncio
    async def test_cancel_waiting_request(self):
        """Cancelled request is removed from consideration."""
        config = QueueConfig(max_slots=1)
        queue = PriorityRequestQueue(config)

        # Take the only slot
        await queue.enqueue("req-1", Priority.NORMAL)

        # Queue two waiters
        entry2 = await queue.enqueue("req-2", Priority.NORMAL)
        entry3 = await queue.enqueue("req-3", Priority.NORMAL)

        # Cancel the first waiter
        result = await queue.cancel("req-2")
        assert result is True
        assert entry2._cancelled is True

        # Release slot - should wake req-3, not req-2
        await queue.release_slot("req-1")
        assert entry3._slot_event.is_set()

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_request(self):
        """Cancelling nonexistent request returns False."""
        config = QueueConfig(max_slots=5)
        queue = PriorityRequestQueue(config)

        result = await queue.cancel("nonexistent")
        assert result is False


class TestPriorityRequestQueueDrain:
    """Test queue drain for graceful shutdown."""

    @pytest.mark.asyncio
    async def test_drain_cancels_waiting_requests(self):
        """Drain cancels all waiting requests."""
        config = QueueConfig(max_slots=1)
        queue = PriorityRequestQueue(config)

        # Take the only slot
        await queue.enqueue("req-1", Priority.NORMAL)

        # Queue several waiters
        entry2 = await queue.enqueue("req-2", Priority.NORMAL)
        entry3 = await queue.enqueue("req-3", Priority.NORMAL)
        entry4 = await queue.enqueue("req-4", Priority.NORMAL)

        # Drain
        cancelled = await queue.drain(timeout=0.1)

        assert cancelled == 3
        assert entry2._cancelled is True
        assert entry3._cancelled is True
        assert entry4._cancelled is True

    @pytest.mark.asyncio
    async def test_drain_waits_for_active(self):
        """Drain waits for active requests to complete."""
        config = QueueConfig(max_slots=1)
        queue = PriorityRequestQueue(config)

        # Take a slot
        await queue.enqueue("req-1", Priority.NORMAL)

        # Start drain in background
        drain_task = asyncio.create_task(queue.drain(timeout=1.0))

        # Release the slot after a short delay
        await asyncio.sleep(0.1)
        await queue.release_slot("req-1")

        # Drain should complete
        cancelled = await asyncio.wait_for(drain_task, timeout=2.0)
        assert cancelled == 0

    @pytest.mark.asyncio
    async def test_drain_timeout(self):
        """Drain times out if active requests don't complete."""
        config = QueueConfig(max_slots=1)
        queue = PriorityRequestQueue(config)

        # Take a slot but never release
        await queue.enqueue("req-1", Priority.NORMAL)

        # Drain should timeout
        start = time.time()
        cancelled = await queue.drain(timeout=0.2)
        elapsed = time.time() - start

        assert elapsed >= 0.2
        assert elapsed < 0.5


class TestPriorityRequestQueueStats:
    """Test queue statistics."""

    @pytest.mark.asyncio
    async def test_stats_initial(self):
        """Initial stats are zeroed."""
        config = QueueConfig(max_slots=5)
        queue = PriorityRequestQueue(config)

        stats = queue.get_stats()

        assert stats.active_slots == 0
        assert stats.max_slots == 5
        assert stats.queue_depth == 0
        assert stats.waiting_high == 0
        assert stats.waiting_normal == 0
        assert stats.waiting_low == 0
        assert stats.total_enqueued == 0
        assert stats.total_timeouts == 0
        assert stats.total_completed == 0
        assert stats.avg_wait_time_ms == 0.0
        assert stats.enabled is True

    @pytest.mark.asyncio
    async def test_stats_track_enqueued(self):
        """Stats track enqueued count."""
        config = QueueConfig(max_slots=5)
        queue = PriorityRequestQueue(config)

        await queue.enqueue("req-1", Priority.NORMAL)
        await queue.enqueue("req-2", Priority.HIGH)

        stats = queue.get_stats()
        assert stats.total_enqueued == 2
        assert stats.active_slots == 2

    @pytest.mark.asyncio
    async def test_stats_track_completed(self):
        """Stats track completed count."""
        config = QueueConfig(max_slots=5)
        queue = PriorityRequestQueue(config)

        await queue.enqueue("req-1", Priority.NORMAL)
        await queue.release_slot("req-1")

        stats = queue.get_stats()
        assert stats.total_completed == 1

    @pytest.mark.asyncio
    async def test_stats_track_timeouts(self):
        """Stats track timeout count."""
        config = QueueConfig(max_slots=1, timeout_normal=0.05)
        queue = PriorityRequestQueue(config)

        await queue.enqueue("req-1", Priority.NORMAL)
        entry2 = await queue.enqueue("req-2", Priority.NORMAL)

        with pytest.raises(QueueTimeoutError):
            await queue.wait_for_slot(entry2)

        stats = queue.get_stats()
        assert stats.total_timeouts == 1

    @pytest.mark.asyncio
    async def test_stats_track_waiting_by_priority(self):
        """Stats track waiting counts by priority."""
        config = QueueConfig(max_slots=1)
        queue = PriorityRequestQueue(config)

        # Take the only slot
        await queue.enqueue("req-1", Priority.NORMAL)

        # Queue by priority
        await queue.enqueue("req-h1", Priority.HIGH)
        await queue.enqueue("req-h2", Priority.HIGH)
        await queue.enqueue("req-n1", Priority.NORMAL)
        await queue.enqueue("req-l1", Priority.LOW)
        await queue.enqueue("req-l2", Priority.LOW)
        await queue.enqueue("req-l3", Priority.LOW)

        stats = queue.get_stats()
        assert stats.waiting_high == 2
        assert stats.waiting_normal == 1
        assert stats.waiting_low == 3


class TestPriorityRequestQueuePosition:
    """Test queue position tracking."""

    @pytest.mark.asyncio
    async def test_position_processing(self):
        """Processing request returns position 0."""
        config = QueueConfig(max_slots=5)
        queue = PriorityRequestQueue(config)

        await queue.enqueue("req-1", Priority.NORMAL)
        position = await queue.get_position("req-1")

        assert position == 0

    @pytest.mark.asyncio
    async def test_position_queued(self):
        """Queued request returns correct position."""
        config = QueueConfig(max_slots=1)
        queue = PriorityRequestQueue(config)

        await queue.enqueue("req-1", Priority.NORMAL)  # Gets slot
        await queue.enqueue("req-2", Priority.NORMAL)  # Position 1
        await queue.enqueue("req-3", Priority.NORMAL)  # Position 2

        assert await queue.get_position("req-1") == 0
        assert await queue.get_position("req-2") == 1
        assert await queue.get_position("req-3") == 2

    @pytest.mark.asyncio
    async def test_position_nonexistent(self):
        """Nonexistent request returns None."""
        config = QueueConfig(max_slots=5)
        queue = PriorityRequestQueue(config)

        position = await queue.get_position("nonexistent")
        assert position is None


class TestPriorityRequestQueueEnabled:
    """Test enabled flag behavior."""

    def test_enabled_property(self):
        """Enabled property reflects config."""
        config_enabled = QueueConfig(enabled=True)
        config_disabled = QueueConfig(enabled=False)

        queue_enabled = PriorityRequestQueue(config_enabled)
        queue_disabled = PriorityRequestQueue(config_disabled)

        assert queue_enabled.enabled is True
        assert queue_disabled.enabled is False


class TestPriorityRequestQueueConcurrency:
    """Test concurrent access patterns."""

    @pytest.mark.asyncio
    async def test_concurrent_enqueue(self):
        """Multiple concurrent enqueues work correctly."""
        config = QueueConfig(max_slots=5, reject_threshold=100)
        queue = PriorityRequestQueue(config)

        async def enqueue_request(i):
            return await queue.enqueue(f"req-{i}", Priority.NORMAL)

        # Enqueue 20 requests concurrently
        tasks = [enqueue_request(i) for i in range(20)]
        entries = await asyncio.gather(*tasks)

        assert len(entries) == 20

        # 5 should have slots, 15 should be waiting
        with_slots = sum(1 for e in entries if e._slot_event.is_set())
        assert with_slots == 5

        stats = queue.get_stats()
        assert stats.total_enqueued == 20
        assert stats.active_slots == 5

    @pytest.mark.asyncio
    async def test_concurrent_release(self):
        """Multiple concurrent releases work correctly."""
        config = QueueConfig(max_slots=5)
        queue = PriorityRequestQueue(config)

        # Enqueue 5 requests
        for i in range(5):
            await queue.enqueue(f"req-{i}", Priority.NORMAL)

        # Release all concurrently
        tasks = [queue.release_slot(f"req-{i}") for i in range(5)]
        await asyncio.gather(*tasks)

        stats = queue.get_stats()
        assert stats.active_slots == 0
        assert stats.total_completed == 5
