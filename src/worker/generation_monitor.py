"""
Generation Monitoring - Watchdog for detecting and preventing worker hangs.

Implements:
1. Generation timeout protection (signal-based)
2. Heartbeat mechanism (periodic alive signals)
3. Detailed generation logging

Created after worker hang incident on 2026-01-05.
See: docs/worker-hang-analysis-and-diagnostic-guide.md
"""

import signal
import time
import logging
import threading
from typing import Optional, Callable
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class GenerationTimeout(Exception):
    """Raised when generation exceeds maximum allowed time."""
    pass


class GenerationMonitor:
    """
    Monitors generation requests to detect and prevent hangs.

    Features:
    - Timeout protection: Kills worker if generation exceeds max_seconds
    - Heartbeat: Sends periodic alive signals during generation
    - Detailed logging: Tracks generation start/end/duration
    """

    def __init__(self, max_seconds: int = 300, heartbeat_interval: int = 10):
        """
        Initialize generation monitor.

        Args:
            max_seconds: Maximum seconds allowed for generation (default: 5 minutes)
            heartbeat_interval: Seconds between heartbeat signals (default: 10s)
        """
        self.max_seconds = max_seconds
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_callback: Optional[Callable] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._stop_heartbeat = threading.Event()
        self._generation_active = False

        logger.info(f"Generation monitor initialized: timeout={max_seconds}s, heartbeat={heartbeat_interval}s")

    def set_heartbeat_callback(self, callback: Callable):
        """
        Set callback function for heartbeat signals.

        Args:
            callback: Function to call periodically (e.g., send_heartbeat_to_orchestrator)
        """
        self.heartbeat_callback = callback
        logger.debug("Heartbeat callback registered")

    def _timeout_handler(self, signum, frame):
        """Signal handler for generation timeout."""
        logger.error(
            f"GENERATION TIMEOUT: Worker hung for {self.max_seconds}s without completing. "
            f"This indicates a deadlock in MLX generation code. Forcing worker exit."
        )
        raise GenerationTimeout(
            f"Generation exceeded {self.max_seconds}s timeout - worker terminating"
        )

    def _heartbeat_loop(self):
        """Background thread that sends periodic heartbeat signals."""
        logger.debug("Heartbeat thread started")
        while not self._stop_heartbeat.is_set():
            if self._generation_active and self.heartbeat_callback:
                try:
                    self.heartbeat_callback()
                    logger.debug("Heartbeat sent")
                except Exception as e:
                    logger.warning(f"Heartbeat callback failed: {e}")

            # Wait for interval or stop signal
            self._stop_heartbeat.wait(self.heartbeat_interval)

        logger.debug("Heartbeat thread stopped")

    def start_heartbeat(self):
        """Start heartbeat background thread."""
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            logger.warning("Heartbeat already running")
            return

        self._stop_heartbeat.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name="generation-heartbeat"
        )
        self._heartbeat_thread.start()
        logger.info("Heartbeat monitoring started")

    def stop_heartbeat(self):
        """Stop heartbeat background thread."""
        if self._heartbeat_thread:
            self._stop_heartbeat.set()
            self._heartbeat_thread.join(timeout=2.0)
            if self._heartbeat_thread.is_alive():
                logger.warning("Heartbeat thread did not stop cleanly")
            self._heartbeat_thread = None
        logger.info("Heartbeat monitoring stopped")

    @contextmanager
    def monitor_generation(
        self,
        prompt_length: int,
        max_tokens: int,
        temperature: float
    ):
        """
        Context manager for monitored generation.

        Usage:
            with monitor.monitor_generation(len(prompt), max_tokens, temp):
                result = model.generate(...)

        Args:
            prompt_length: Length of prompt in characters
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Raises:
            GenerationTimeout: If generation exceeds max_seconds
        """
        start_time = time.time()

        # Log generation start
        logger.info(
            f"Generation START: prompt={prompt_length} chars, "
            f"max_tokens={max_tokens}, temp={temperature}"
        )

        # Set timeout alarm
        old_handler = signal.signal(signal.SIGALRM, self._timeout_handler)
        signal.alarm(self.max_seconds)
        logger.debug(f"Timeout alarm set for {self.max_seconds}s")

        # Mark generation as active (for heartbeat)
        self._generation_active = True

        try:
            yield  # Execute generation

            # Generation completed successfully
            duration = time.time() - start_time
            logger.info(
                f"Generation COMPLETE: duration={duration:.1f}s "
                f"(timeout was {self.max_seconds}s)"
            )

            # Check if we're approaching timeout (warning threshold: 80%)
            if duration > (self.max_seconds * 0.8):
                logger.warning(
                    f"Generation took {duration:.1f}s, approaching timeout limit "
                    f"of {self.max_seconds}s. Consider increasing timeout or "
                    f"reducing prompt size."
                )

        except GenerationTimeout:
            # Timeout occurred
            duration = time.time() - start_time
            logger.error(
                f"Generation TIMEOUT after {duration:.1f}s. "
                f"Last known state: prompt={prompt_length} chars, max_tokens={max_tokens}"
            )
            raise

        except Exception as e:
            # Other error during generation
            duration = time.time() - start_time
            logger.error(
                f"Generation FAILED after {duration:.1f}s: {type(e).__name__}: {e}"
            )
            raise

        finally:
            # Always clean up
            self._generation_active = False
            signal.alarm(0)  # Cancel alarm
            signal.signal(signal.SIGALRM, old_handler)  # Restore old handler
            logger.debug("Timeout alarm cancelled")


# Global singleton instance
_monitor_instance: Optional[GenerationMonitor] = None


def get_monitor(max_seconds: int = 300, heartbeat_interval: int = 10) -> GenerationMonitor:
    """
    Get global generation monitor instance.

    Args:
        max_seconds: Timeout in seconds (default: 5 minutes)
        heartbeat_interval: Heartbeat interval in seconds (default: 10s)

    Returns:
        GenerationMonitor instance
    """
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = GenerationMonitor(max_seconds, heartbeat_interval)
    return _monitor_instance
