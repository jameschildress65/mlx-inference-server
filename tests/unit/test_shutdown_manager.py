"""
Unit tests for P2 Graceful Shutdown Manager.

Tests drain logic, signal handling, and timeout behavior.
"""

import pytest
import time
import signal
from unittest.mock import Mock, patch, MagicMock
from src.orchestrator.shutdown_manager import ShutdownManager


class TestShutdownManagerDrain:
    """Test request drain logic."""

    def test_drain_completes_when_no_requests(self):
        """Drain completes immediately when no active requests."""
        mock_worker = Mock()
        mock_worker.active_requests = 0
        mock_idle = Mock()

        manager = ShutdownManager(
            worker_manager=mock_worker,
            idle_monitor=mock_idle,
            drain_timeout=5
        )

        result = manager._wait_for_drain()
        assert result is True

    def test_drain_waits_for_requests_to_complete(self):
        """Drain waits for active requests to complete."""
        mock_worker = Mock()
        mock_idle = Mock()

        # Simulate requests completing over time
        call_count = [0]
        def get_active():
            call_count[0] += 1
            if call_count[0] < 3:
                return 2  # Still active
            return 0  # Done

        type(mock_worker).active_requests = property(lambda self: get_active())

        manager = ShutdownManager(
            worker_manager=mock_worker,
            idle_monitor=mock_idle,
            drain_timeout=5
        )

        result = manager._wait_for_drain()
        assert result is True
        assert call_count[0] >= 3

    def test_drain_timeout_returns_false(self):
        """Drain returns False on timeout."""
        mock_worker = Mock()
        mock_worker.active_requests = 5  # Always active
        mock_idle = Mock()

        manager = ShutdownManager(
            worker_manager=mock_worker,
            idle_monitor=mock_idle,
            drain_timeout=1  # Short timeout for test
        )

        start = time.time()
        result = manager._wait_for_drain()
        elapsed = time.time() - start

        assert result is False
        assert elapsed >= 1.0
        assert elapsed < 2.0  # Should not exceed timeout by much


class TestShutdownManagerSignals:
    """Test signal handling."""

    def test_install_handlers(self):
        """Signal handlers are installed correctly."""
        mock_worker = Mock()
        mock_idle = Mock()

        manager = ShutdownManager(
            worker_manager=mock_worker,
            idle_monitor=mock_idle,
            drain_timeout=30
        )

        with patch('signal.signal') as mock_signal:
            manager.install_handlers()

            # Should install both SIGTERM and SIGINT handlers
            assert mock_signal.call_count == 2
            calls = [c[0] for c in mock_signal.call_args_list]
            assert (signal.SIGTERM, manager._handle_signal) in calls
            assert (signal.SIGINT, manager._handle_signal) in calls

    def test_double_signal_forces_exit(self):
        """Second signal during shutdown forces immediate exit."""
        mock_worker = Mock()
        mock_worker.active_requests = 5
        mock_idle = Mock()

        manager = ShutdownManager(
            worker_manager=mock_worker,
            idle_monitor=mock_idle,
            drain_timeout=30
        )

        # Simulate shutdown already in progress
        manager._shutdown_in_progress = True

        with pytest.raises(SystemExit) as exc_info:
            manager._handle_signal(signal.SIGTERM, None)

        assert exc_info.value.code == 1  # Force exit code


class TestShutdownManagerSequence:
    """Test shutdown sequence."""

    def test_graceful_shutdown_calls_in_order(self):
        """Shutdown sequence executes in correct order."""
        mock_worker = Mock()
        mock_worker.active_requests = 0
        mock_idle = Mock()

        manager = ShutdownManager(
            worker_manager=mock_worker,
            idle_monitor=mock_idle,
            drain_timeout=5
        )

        call_order = []

        def track_idle_stop():
            call_order.append('idle_stop')

        def track_worker_shutdown():
            call_order.append('worker_shutdown')

        mock_idle.stop = track_idle_stop
        mock_worker.shutdown = track_worker_shutdown

        with pytest.raises(SystemExit) as exc_info:
            manager._graceful_shutdown()

        # Verify order: idle_stop -> worker_shutdown
        assert call_order == ['idle_stop', 'worker_shutdown']
        assert exc_info.value.code == 0  # Clean exit

    def test_shutdown_continues_on_idle_monitor_error(self):
        """Shutdown continues even if idle monitor fails."""
        mock_worker = Mock()
        mock_worker.active_requests = 0
        mock_idle = Mock()
        mock_idle.stop.side_effect = Exception("Idle monitor error")

        manager = ShutdownManager(
            worker_manager=mock_worker,
            idle_monitor=mock_idle,
            drain_timeout=5
        )

        with pytest.raises(SystemExit) as exc_info:
            manager._graceful_shutdown()

        # Should still call worker shutdown despite idle monitor error
        mock_worker.shutdown.assert_called_once()
        assert exc_info.value.code == 0

    def test_shutdown_continues_on_worker_error(self):
        """Shutdown completes even if worker shutdown fails."""
        mock_worker = Mock()
        mock_worker.active_requests = 0
        mock_worker.shutdown.side_effect = Exception("Worker error")
        mock_idle = Mock()

        manager = ShutdownManager(
            worker_manager=mock_worker,
            idle_monitor=mock_idle,
            drain_timeout=5
        )

        with pytest.raises(SystemExit) as exc_info:
            manager._graceful_shutdown()

        # Should still exit cleanly
        assert exc_info.value.code == 0


class TestShutdownManagerConfig:
    """Test configuration options."""

    def test_default_timeout(self):
        """Default drain timeout is 60 seconds."""
        mock_worker = Mock()
        mock_idle = Mock()

        manager = ShutdownManager(
            worker_manager=mock_worker,
            idle_monitor=mock_idle
        )

        assert manager.drain_timeout == 60

    def test_custom_timeout(self):
        """Custom drain timeout is respected."""
        mock_worker = Mock()
        mock_idle = Mock()

        manager = ShutdownManager(
            worker_manager=mock_worker,
            idle_monitor=mock_idle,
            drain_timeout=120
        )

        assert manager.drain_timeout == 120
