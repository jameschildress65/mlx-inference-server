"""
Graceful shutdown management for MLX Inference Server.

P2 Implementation: Kubernetes-style graceful shutdown.
- Drains in-flight requests before terminating
- Drains priority queue (cancels waiting requests)
- Clean worker process cleanup
- Configurable timeout for home lab use

Haiku validation: Simplified approach - just stop idle monitor,
drain via polling active_requests, force-kill on timeout.
"""

import asyncio
import signal
import logging
import time
import sys
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .worker_manager import WorkerManager
    from .idle_monitor import IdleMonitor
    from .priority_queue import PriorityRequestQueue

logger = logging.getLogger(__name__)


class ShutdownManager:
    """
    Manages graceful shutdown sequence.

    Kubernetes-style shutdown:
    1. Receive SIGTERM/SIGINT
    2. Stop idle monitor (prevent auto-unload interference)
    3. Wait for in-flight requests to drain (with timeout)
    4. Shutdown worker process
    5. Exit

    For home lab: 60-second default timeout (streaming can be slow).
    """

    def __init__(
        self,
        worker_manager: 'WorkerManager',
        idle_monitor: 'IdleMonitor',
        drain_timeout: int = 60,
        request_queue: Optional['PriorityRequestQueue'] = None
    ):
        """
        Initialize shutdown manager.

        Args:
            worker_manager: WorkerManager instance
            idle_monitor: IdleMonitor instance
            drain_timeout: Seconds to wait for request drain (default 60)
            request_queue: Optional PriorityRequestQueue instance for draining
        """
        self.worker_manager = worker_manager
        self.idle_monitor = idle_monitor
        self.drain_timeout = drain_timeout
        self.request_queue = request_queue
        self._shutdown_in_progress = False

    def install_handlers(self):
        """Install signal handlers for graceful shutdown."""
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)
        logger.info(f"Graceful shutdown handlers installed (drain timeout: {self.drain_timeout}s)")

    def _handle_signal(self, signum, frame):
        """Handle shutdown signal."""
        signal_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"

        # Double-signal = force exit
        if self._shutdown_in_progress:
            logger.warning(f"Received {signal_name} during shutdown - forcing exit")
            sys.exit(1)

        logger.info(f"Received {signal_name}, initiating graceful shutdown...")
        self._shutdown_in_progress = True
        self._graceful_shutdown()

    def _graceful_shutdown(self):
        """Execute graceful shutdown sequence."""
        # Step 1: Stop idle monitor (prevent interference during drain)
        logger.info("Step 1/5: Stopping idle monitor...")
        try:
            self.idle_monitor.stop()
        except Exception as e:
            logger.warning(f"Idle monitor stop error (continuing): {e}")

        # Step 2: Drain priority queue (cancel waiting requests)
        if self.request_queue is not None:
            logger.info("Step 2/5: Draining priority queue...")
            try:
                # Run drain in event loop
                cancelled = self._drain_queue()
                if cancelled > 0:
                    logger.info(f"Cancelled {cancelled} queued requests")
            except Exception as e:
                logger.warning(f"Queue drain error (continuing): {e}")
        else:
            logger.info("Step 2/5: No priority queue to drain")

        # Step 3: Drain in-flight requests
        logger.info(f"Step 3/5: Draining in-flight requests (timeout: {self.drain_timeout}s)...")
        drain_success = self._wait_for_drain()

        if drain_success:
            logger.info("All requests drained successfully")
        else:
            active = self.worker_manager.active_requests
            logger.warning(f"Drain timeout - {active} requests still active, proceeding with shutdown")

        # Step 4: Shutdown worker
        logger.info("Step 4/5: Shutting down worker...")
        try:
            self.worker_manager.shutdown()
        except Exception as e:
            logger.error(f"Worker shutdown error: {e}")

        # Step 5: Exit
        logger.info("Step 5/5: Graceful shutdown complete")
        sys.exit(0)

    def _drain_queue(self) -> int:
        """
        Drain the priority queue synchronously.

        Signal handlers run in main thread but outside async context.
        We must handle both cases:
        1. Event loop running (FastAPI/uvicorn active) - use new thread
        2. No event loop - use asyncio.run()

        Returns:
            Number of cancelled requests
        """
        if self.request_queue is None:
            return 0

        drain_timeout = min(self.drain_timeout, 30.0)  # Cap at 30s for queue drain

        try:
            # Check if there's a running loop
            loop = asyncio.get_running_loop()
            # Loop is running - we cannot use run_until_complete
            # Use threadsafe scheduling with a completion event
            import threading
            result = [0]
            done_event = threading.Event()

            def run_drain():
                try:
                    # Create new loop in this thread
                    result[0] = asyncio.run(
                        self.request_queue.drain(timeout=drain_timeout)
                    )
                except Exception as e:
                    logger.warning(f"Queue drain in thread failed: {e}")
                finally:
                    done_event.set()

            drain_thread = threading.Thread(target=run_drain, daemon=True)
            drain_thread.start()
            # Wait with timeout
            done_event.wait(timeout=drain_timeout + 5.0)
            return result[0]

        except RuntimeError:
            # No running loop - we can use asyncio.run directly
            return asyncio.run(self.request_queue.drain(timeout=drain_timeout))

    def _wait_for_drain(self) -> bool:
        """
        Wait for all in-flight requests to complete.

        Polls worker_manager.active_requests until 0 or timeout.
        Uses existing thread-safe counter (no new locks needed).

        Returns:
            True if all requests drained, False if timeout
        """
        start_time = time.time()
        check_interval = 0.5  # Poll every 500ms
        last_log_time = start_time

        while time.time() - start_time < self.drain_timeout:
            active = self.worker_manager.active_requests

            if active == 0:
                elapsed = time.time() - start_time
                logger.info(f"Drain complete in {elapsed:.1f}s")
                return True

            # Log progress every 5 seconds
            now = time.time()
            if now - last_log_time >= 5.0:
                remaining = self.drain_timeout - (now - start_time)
                logger.info(f"Draining: {active} active request(s), {remaining:.0f}s remaining")
                last_log_time = now

            time.sleep(check_interval)

        return False
