"""IdleMonitor for V3 - Auto-unloads idle workers.

Monitors WorkerManager activity and automatically unloads (kills) worker
subprocess after configured idle period to free memory.

V3 Adaptation:
- Works with WorkerManager instead of ModelProvider
- Tracks worker activity instead of in-process model
- Kills worker subprocess on timeout (instant cleanup)
"""

import threading
import time
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .worker_manager import WorkerManager


class IdleMonitor(threading.Thread):
    """
    Background daemon thread for idle timeout monitoring.

    Automatically unloads worker after configured idle period to free memory.
    V3-specific: Monitors worker subprocess activity.
    """

    def __init__(
        self,
        worker_manager: 'WorkerManager',
        idle_timeout_seconds: int,
        check_interval_seconds: int = 30
    ):
        """
        Initialize idle monitor for V3.

        Args:
            worker_manager: WorkerManager instance to monitor
            idle_timeout_seconds: Idle time before auto-unload
            check_interval_seconds: How often to check (default 30s)
        """
        super().__init__(daemon=True, name="V3-IdleMonitor")

        self.worker_manager = worker_manager
        self.idle_timeout = idle_timeout_seconds
        self.check_interval = check_interval_seconds
        self._stop_event = threading.Event()

        self.logger = logging.getLogger(__name__)

    def run(self) -> None:
        """
        Main monitor loop.

        Runs continuously in background, checking worker idle time at regular intervals.
        When idle time exceeds threshold and worker is running, triggers auto-unload.
        """
        self.logger.info(
            f"V3 IdleMonitor started (timeout={self.idle_timeout}s, "
            f"check_interval={self.check_interval}s)"
        )

        while not self._stop_event.is_set():
            try:
                # Get worker activity status
                idle_time = self.worker_manager.get_idle_time()
                has_active = self.worker_manager.has_active_requests()
                is_loaded = self.worker_manager.get_status()["model_loaded"]

                # Only unload if:
                # 1. Worker is loaded
                # 2. No active requests
                # 3. Idle time exceeded timeout
                if is_loaded and not has_active and idle_time >= self.idle_timeout:
                    self.logger.info(
                        f"Idle timeout reached ({idle_time:.1f}s), "
                        "attempting atomic unload..."
                    )

                    try:
                        # Atomic check-and-unload to prevent TOCTOU race
                        # Returns None if worker becomes active between check and unload
                        result = self.worker_manager.unload_model_if_idle()

                        if result is not None:
                            self.logger.info(
                                f"Auto-unloaded {result.model_name}, "
                                f"freed {result.memory_freed_gb:.2f} GB"
                            )
                        else:
                            self.logger.debug(
                                "Unload skipped - worker became active or was already unloaded"
                            )
                    except Exception as e:
                        self.logger.error(f"Auto-unload failed: {e}", exc_info=True)

                # Sleep until next check (interruptible for clean shutdown)
                self._stop_event.wait(self.check_interval)

            except Exception as e:
                # Log error but keep running
                self.logger.error(
                    f"Error in idle monitor: {e}",
                    exc_info=True
                )
                # Wait before retrying
                self._stop_event.wait(self.check_interval)

        self.logger.info("V3 IdleMonitor stopped")

    def stop(self) -> None:
        """
        Signal monitor to stop gracefully.

        Sets stop event and waits for thread to complete.
        """
        self.logger.info("V3 IdleMonitor stopping...")
        self._stop_event.set()

    def update_timeout(self, new_timeout_seconds: int) -> None:
        """
        Update idle timeout dynamically.

        Args:
            new_timeout_seconds: New timeout value in seconds

        Raises:
            ValueError: If timeout is not positive
        """
        if new_timeout_seconds <= 0:
            raise ValueError("Timeout must be positive")

        old_timeout = self.idle_timeout
        self.idle_timeout = new_timeout_seconds

        self.logger.info(
            f"Idle timeout updated: {old_timeout}s → {new_timeout_seconds}s"
        )
