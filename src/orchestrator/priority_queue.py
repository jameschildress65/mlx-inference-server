"""
Priority Request Queue for MLX Inference Server.

Provides queued waiting instead of immediate 503 rejection, with:
- Priority levels (HIGH, NORMAL, LOW)
- Queue position tracking
- Per-priority timeout handling
- Graceful drain for shutdown

Architecture:
- Uses asyncio.PriorityQueue for ordering by (priority, insertion_order)
- asyncio.Lock protects shared state
- asyncio.Event per request for slot signaling
- Insertion counter ensures FIFO within same priority

Thread Safety:
- All public methods are async and use asyncio.Lock
- Safe for concurrent async access within single event loop
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, Dict, Deque

logger = logging.getLogger(__name__)


class Priority(IntEnum):
    """Request priority levels.

    Lower numeric value = higher priority (for heap ordering).
    """
    HIGH = 0    # Admin/critical requests
    NORMAL = 1  # Standard API requests (default)
    LOW = 2     # Batch/background requests


@dataclass
class QueueConfig:
    """Configuration for priority queue behavior."""
    max_slots: int = 10           # Max concurrent processing (replaces semaphore)
    max_queue_depth: int = 50     # Max pending requests in queue
    reject_threshold: int = 100   # Fast-reject when total exceeds this
    timeout_high: float = 120.0   # Timeout for HIGH priority (seconds)
    timeout_normal: float = 60.0  # Timeout for NORMAL priority (seconds)
    timeout_low: float = 30.0     # Timeout for LOW priority (seconds)
    enabled: bool = True          # Feature flag (False = use semaphore fallback)


@dataclass(order=True)
class QueuedRequest:
    """A request waiting in the priority queue.

    Ordering: (priority, insertion_order) for min-heap.
    Lower priority value = higher priority.
    FIFO within same priority via insertion_order.
    """
    priority: Priority
    insertion_order: int
    request_id: str = field(compare=False)
    enqueue_time: float = field(compare=False, default_factory=time.time)
    timeout: float = field(compare=False, default=60.0)
    _slot_event: asyncio.Event = field(compare=False, default_factory=asyncio.Event)
    _cancelled: bool = field(compare=False, default=False)

    @property
    def wait_time_ms(self) -> float:
        """Time spent waiting in queue (milliseconds)."""
        return (time.time() - self.enqueue_time) * 1000


@dataclass
class QueueStats:
    """Queue statistics snapshot."""
    active_slots: int
    max_slots: int
    queue_depth: int
    max_queue_depth: int
    waiting_high: int
    waiting_normal: int
    waiting_low: int
    total_enqueued: int
    total_timeouts: int
    total_completed: int
    avg_wait_time_ms: float
    enabled: bool


class QueueTimeoutError(Exception):
    """Raised when a request times out waiting in queue."""
    def __init__(self, request_id: str, wait_time_ms: float, position: int):
        self.request_id = request_id
        self.wait_time_ms = wait_time_ms
        self.position = position
        super().__init__(f"Request {request_id} timed out after {wait_time_ms:.0f}ms at position {position}")


class QueueFullError(Exception):
    """Raised when queue is at reject threshold."""
    def __init__(self, queue_depth: int, threshold: int):
        self.queue_depth = queue_depth
        self.threshold = threshold
        super().__init__(f"Queue full: {queue_depth} >= threshold {threshold}")


class PriorityRequestQueue:
    """
    Thread-safe priority request queue with slot management.

    Replaces simple semaphore backpressure with:
    - Priority ordering (HIGH > NORMAL > LOW)
    - Queued waiting instead of immediate rejection
    - Position tracking for client feedback
    - Per-priority timeouts
    - Graceful drain for shutdown

    Usage:
        queue = PriorityRequestQueue(config)

        # In endpoint:
        entry = await queue.enqueue(request_id, Priority.NORMAL)
        try:
            await queue.wait_for_slot(entry)
            # Process request...
        finally:
            queue.release_slot(request_id)
    """

    def __init__(self, config: QueueConfig):
        self._config = config
        self._lock = asyncio.Lock()

        # Slot management
        self._active_slots = 0

        # Queue storage
        self._queue: asyncio.PriorityQueue[QueuedRequest] = asyncio.PriorityQueue()
        self._request_map: Dict[str, QueuedRequest] = {}

        # Counters for ordering and stats
        self._insertion_counter = 0
        self._total_enqueued = 0
        self._total_timeouts = 0
        self._total_completed = 0
        self._max_wait_samples = 1000
        self._wait_times: Deque[float] = deque(maxlen=self._max_wait_samples)

        # Shutdown state
        self._draining = False
        self._drain_event = asyncio.Event()

        logger.info(
            f"PriorityRequestQueue initialized: max_slots={config.max_slots}, "
            f"max_queue_depth={config.max_queue_depth}, enabled={config.enabled}"
        )

    @property
    def enabled(self) -> bool:
        """Whether queue is enabled (vs semaphore fallback)."""
        return self._config.enabled

    async def enqueue(
        self,
        request_id: str,
        priority: Priority = Priority.NORMAL,
        timeout: Optional[float] = None
    ) -> QueuedRequest:
        """
        Enqueue a request for processing.

        If a slot is immediately available, the request's event is set.
        Otherwise, request waits in queue until a slot is released.

        Args:
            request_id: Unique request identifier
            priority: Request priority level
            timeout: Override timeout (uses config default if None)

        Returns:
            QueuedRequest entry (use with wait_for_slot())

        Raises:
            QueueFullError: If queue exceeds reject threshold
        """
        if self._draining:
            raise QueueFullError(
                len(self._request_map),
                self._config.reject_threshold
            )

        # Determine timeout
        if timeout is None:
            timeout = self._get_priority_timeout(priority)

        async with self._lock:
            # Check reject threshold
            total_pending = self._active_slots + self._queue.qsize()
            if total_pending >= self._config.reject_threshold:
                logger.warning(
                    f"Queue reject threshold reached: {total_pending} >= {self._config.reject_threshold}"
                )
                raise QueueFullError(total_pending, self._config.reject_threshold)

            # Create entry
            entry = QueuedRequest(
                priority=priority,
                insertion_order=self._insertion_counter,
                request_id=request_id,
                timeout=timeout
            )
            self._insertion_counter += 1
            self._total_enqueued += 1
            self._request_map[request_id] = entry

            # Check for immediate slot
            if self._active_slots < self._config.max_slots:
                self._active_slots += 1
                entry._slot_event.set()
                logger.debug(f"[{request_id}] Immediate slot granted (active={self._active_slots})")
            else:
                # Queue for later
                await self._queue.put(entry)
                position = self._queue.qsize()
                logger.debug(
                    f"[{request_id}] Queued at position {position} "
                    f"(priority={priority.name}, active={self._active_slots})"
                )

            return entry

    async def wait_for_slot(self, entry: QueuedRequest) -> bool:
        """
        Wait for a processing slot to become available.

        Args:
            entry: QueuedRequest from enqueue()

        Returns:
            True if slot acquired

        Raises:
            QueueTimeoutError: If timeout expires before slot available
        """
        try:
            await asyncio.wait_for(entry._slot_event.wait(), timeout=entry.timeout)

            # Record wait time
            wait_ms = entry.wait_time_ms
            self._record_wait_time(wait_ms)

            logger.debug(f"[{entry.request_id}] Slot acquired after {wait_ms:.0f}ms")
            return True

        except asyncio.TimeoutError:
            # Timeout - cleanup and raise
            position = await self._remove_from_queue(entry.request_id)
            self._total_timeouts += 1

            wait_ms = entry.wait_time_ms
            logger.warning(
                f"[{entry.request_id}] Queue timeout after {wait_ms:.0f}ms at position {position}"
            )
            raise QueueTimeoutError(entry.request_id, wait_ms, position)

    async def release_slot(self, request_id: str) -> None:
        """
        Release a processing slot after request completion.

        Wakes the highest-priority waiting request, if any.

        Args:
            request_id: Request ID to release
        """
        async with self._lock:
            # Remove from tracking
            entry = self._request_map.pop(request_id, None)
            if entry:
                self._total_completed += 1

            # Release slot
            if self._active_slots > 0:
                self._active_slots -= 1

            # Wake next waiter if any (use get_nowait to avoid blocking inside lock)
            while not self._queue.empty():
                try:
                    next_entry = self._queue.get_nowait()

                    # Skip cancelled entries
                    if next_entry._cancelled:
                        continue

                    self._active_slots += 1
                    next_entry._slot_event.set()
                    logger.debug(
                        f"[{next_entry.request_id}] Woken from queue "
                        f"(priority={next_entry.priority.name})"
                    )
                    break  # Only wake one
                except asyncio.QueueEmpty:
                    break

            # Check drain completion
            if self._draining and self._active_slots == 0 and self._queue.empty():
                self._drain_event.set()

            logger.debug(f"[{request_id}] Slot released (active={self._active_slots})")

    async def get_position(self, request_id: str) -> Optional[int]:
        """
        Get current queue position for a request.

        Args:
            request_id: Request ID to check

        Returns:
            Position (0 = processing, 1+ = queue position), or None if not found
        """
        async with self._lock:
            entry = self._request_map.get(request_id)
            if entry is None:
                return None

            if entry._slot_event.is_set():
                return 0  # Currently processing

            # Count requests ahead in queue
            # Note: This is O(n) but queue is typically small
            position = 1
            temp_list = []
            while not self._queue.empty():
                try:
                    queued = self._queue.get_nowait()
                    temp_list.append(queued)
                    if queued.request_id == request_id:
                        break
                    if not queued._cancelled:
                        position += 1
                except asyncio.QueueEmpty:
                    break

            # Restore queue (use put_nowait - we know capacity exists)
            for item in temp_list:
                self._queue.put_nowait(item)

            return position

    def get_stats(self) -> QueueStats:
        """
        Get current queue statistics.

        Returns:
            QueueStats snapshot
        """
        # Count by priority (approximate - doesn't lock)
        waiting_high = 0
        waiting_normal = 0
        waiting_low = 0

        for entry in self._request_map.values():
            if not entry._slot_event.is_set():
                if entry.priority == Priority.HIGH:
                    waiting_high += 1
                elif entry.priority == Priority.NORMAL:
                    waiting_normal += 1
                else:
                    waiting_low += 1

        avg_wait = sum(self._wait_times) / len(self._wait_times) if self._wait_times else 0.0

        return QueueStats(
            active_slots=self._active_slots,
            max_slots=self._config.max_slots,
            queue_depth=self._queue.qsize(),
            max_queue_depth=self._config.max_queue_depth,
            waiting_high=waiting_high,
            waiting_normal=waiting_normal,
            waiting_low=waiting_low,
            total_enqueued=self._total_enqueued,
            total_timeouts=self._total_timeouts,
            total_completed=self._total_completed,
            avg_wait_time_ms=avg_wait,
            enabled=self._config.enabled
        )

    async def cancel(self, request_id: str) -> bool:
        """
        Cancel a waiting request.

        Args:
            request_id: Request ID to cancel

        Returns:
            True if request was found and cancelled
        """
        async with self._lock:
            entry = self._request_map.get(request_id)
            if entry is None:
                return False

            entry._cancelled = True
            entry._slot_event.set()  # Wake up waiter so it can check cancelled

            logger.debug(f"[{request_id}] Cancelled")
            return True

    async def drain(self, timeout: float = 60.0) -> int:
        """
        Drain queue for graceful shutdown.

        Stops accepting new requests and waits for active requests to complete.

        Args:
            timeout: Maximum time to wait for drain (seconds)

        Returns:
            Number of requests that were cancelled due to timeout
        """
        self._draining = True
        logger.info(f"Queue draining initiated (active={self._active_slots}, queued={self._queue.qsize()})")

        # Cancel all waiting requests
        cancelled_count = 0
        async with self._lock:
            while not self._queue.empty():
                try:
                    entry = await self._queue.get()
                    entry._cancelled = True
                    entry._slot_event.set()
                    cancelled_count += 1
                except asyncio.QueueEmpty:
                    break

        if cancelled_count > 0:
            logger.info(f"Cancelled {cancelled_count} queued requests during drain")

        # Wait for active requests to complete
        if self._active_slots > 0:
            try:
                await asyncio.wait_for(self._drain_event.wait(), timeout=timeout)
                logger.info("Queue drain completed")
            except asyncio.TimeoutError:
                logger.warning(f"Queue drain timeout after {timeout}s with {self._active_slots} active")
        else:
            logger.info("Queue drain completed (no active requests)")

        return cancelled_count

    async def _remove_from_queue(self, request_id: str) -> int:
        """Remove a request from queue and return its position."""
        async with self._lock:
            entry = self._request_map.pop(request_id, None)
            if entry:
                entry._cancelled = True

            # Find position (queue doesn't support removal, so mark cancelled)
            position = 0
            temp_list = []
            while not self._queue.empty():
                try:
                    queued = self._queue.get_nowait()
                    temp_list.append(queued)
                    if queued.request_id == request_id:
                        position = len(temp_list)
                except asyncio.QueueEmpty:
                    break

            # Restore queue (excluding cancelled, use put_nowait - we know capacity exists)
            for item in temp_list:
                if item.request_id != request_id:
                    self._queue.put_nowait(item)

            return position

    def _get_priority_timeout(self, priority: Priority) -> float:
        """Get timeout for a priority level."""
        if priority == Priority.HIGH:
            return self._config.timeout_high
        elif priority == Priority.NORMAL:
            return self._config.timeout_normal
        else:
            return self._config.timeout_low

    def _record_wait_time(self, wait_ms: float) -> None:
        """Record wait time sample for stats (O(1) with deque maxlen)."""
        self._wait_times.append(wait_ms)
