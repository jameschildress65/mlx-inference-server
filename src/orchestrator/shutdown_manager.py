"""
Graceful shutdown management for MLX Inference Server.

P2 Implementation: Kubernetes-style graceful shutdown.
- Drains in-flight requests before terminating
- Clean worker process cleanup
- Configurable timeout for home lab use

Haiku validation: Simplified approach - just stop idle monitor,
drain via polling active_requests, force-kill on timeout.
"""

import signal
import logging
import time
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .worker_manager import WorkerManager
    from .idle_monitor import IdleMonitor

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
        drain_timeout: int = 60
    ):
        """
        Initialize shutdown manager.

        Args:
            worker_manager: WorkerManager instance
            idle_monitor: IdleMonitor instance
            drain_timeout: Seconds to wait for request drain (default 60)
        """
        self.worker_manager = worker_manager
        self.idle_monitor = idle_monitor
        self.drain_timeout = drain_timeout
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
        logger.info("Step 1/4: Stopping idle monitor...")
        try:
            self.idle_monitor.stop()
        except Exception as e:
            logger.warning(f"Idle monitor stop error (continuing): {e}")

        # Step 2: Drain in-flight requests
        logger.info(f"Step 2/4: Draining in-flight requests (timeout: {self.drain_timeout}s)...")
        drain_success = self._wait_for_drain()

        if drain_success:
            logger.info("All requests drained successfully")
        else:
            active = self.worker_manager.active_requests
            logger.warning(f"Drain timeout - {active} requests still active, proceeding with shutdown")

        # Step 3: Shutdown worker
        logger.info("Step 3/4: Shutting down worker...")
        try:
            self.worker_manager.shutdown()
        except Exception as e:
            logger.error(f"Worker shutdown error: {e}")

        # Step 4: Exit
        logger.info("Step 4/4: Graceful shutdown complete")
        sys.exit(0)

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
