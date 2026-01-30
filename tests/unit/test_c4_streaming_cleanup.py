"""
Unit tests for C4 fix: Streaming cleanup on worker death.

Tests that generate_stream() detects worker death early and cleans up properly.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import threading

from src.config.server_config import ServerConfig


@pytest.fixture
def mock_config():
    """Create mock ServerConfig for tests."""
    return ServerConfig(
        main_port=11440,
        admin_port=11441,
        host="0.0.0.0",
        idle_timeout_seconds=180,
        request_timeout_seconds=300,
        model_load_timeout_seconds=300,
        max_concurrent_requests=10,
        memory_threshold_gb=28,
        cache_dir="/test/cache",
        log_dir="/test/logs",
        machine_type="test-machine",
        total_ram_gb=32,
        chip_model="Test Chip",
        model_name="Test Model",
        use_shared_memory=True,
        rate_limit_enabled=False,  # P1: Rate limiting disabled for tests
        rate_limit_rpm=60,
        rate_limit_burst=10,
        graceful_shutdown_timeout=60  # P2: Graceful shutdown timeout
    )


class TestStreamingWorkerDeathDetection:
    """Tests for C4 fix: Early worker death detection in streaming."""

    def test_worker_death_detected_before_receive(self, mock_config):
        """Worker death is detected before blocking on receive."""
        from src.orchestrator.worker_manager import WorkerManager, WorkerError

        # Create manager with mocked components
        wm = WorkerManager(mock_config)

        # Setup: Mock a dead worker
        mock_worker = Mock()
        mock_worker.poll.return_value = -9  # SIGKILL exit code
        mock_worker.returncode = -9
        wm.active_worker = mock_worker
        wm.active_model_name = "test-model"

        # Mock the IPC components
        wm._shmem_bridge = Mock()
        wm._stdio_handler = Mock()

        # Mock verify to not raise (pretend model matches)
        with patch.object(wm, '_verify_worker_alive_locked'):
            with patch.object(wm, '_send_message'):
                # Attempt to stream - should detect death before receive
                request = Mock()
                request.model = "test-model"
                request.stream = True

                gen = wm.generate_stream(request)

                with pytest.raises(WorkerError) as exc_info:
                    next(gen)

                assert "Worker died during streaming" in str(exc_info.value)
                assert "exit code: -9" in str(exc_info.value)

    def test_cleanup_called_on_worker_death(self, mock_config):
        """_cleanup_dead_worker is called when worker dies during streaming."""
        from src.orchestrator.worker_manager import WorkerManager, WorkerError

        wm = WorkerManager(mock_config)

        # Setup: Mock worker that reports dead
        mock_worker = Mock()
        mock_worker.poll.return_value = 1  # Non-zero exit
        mock_worker.returncode = 1
        wm.active_worker = mock_worker
        wm.active_model_name = "test-model"
        wm._shmem_bridge = Mock()

        cleanup_called = []

        def track_cleanup():
            cleanup_called.append(True)

        with patch.object(wm, '_verify_worker_alive_locked'):
            with patch.object(wm, '_send_message'):
                with patch.object(wm, '_cleanup_dead_worker', side_effect=track_cleanup):
                    request = Mock(model="test-model", stream=True)
                    gen = wm.generate_stream(request)

                    with pytest.raises(WorkerError):
                        next(gen)

                    assert len(cleanup_called) == 1, "Cleanup should be called once"

    def test_poll_called_before_each_receive(self, mock_config):
        """poll() is called before each _receive_message()."""
        from src.orchestrator.worker_manager import WorkerManager, WorkerError
        from src.ipc.messages import StreamChunk

        wm = WorkerManager(mock_config)

        # Track poll calls
        poll_calls = []
        receive_calls = []

        mock_worker = Mock()
        def track_poll():
            poll_calls.append(len(receive_calls))  # Record how many receives happened
            return None  # Worker alive

        mock_worker.poll.side_effect = track_poll
        wm.active_worker = mock_worker
        wm.active_model_name = "test-model"

        # Return 3 chunks then done
        chunks = [
            StreamChunk(text="a", token=1, done=False),
            StreamChunk(text="b", token=2, done=False),
            StreamChunk(text="c", token=3, done=True),
        ]

        def mock_receive(timeout):
            receive_calls.append(True)
            return chunks[len(receive_calls) - 1]

        with patch.object(wm, '_verify_worker_alive_locked'):
            with patch.object(wm, '_send_message'):
                with patch.object(wm, '_receive_message', side_effect=mock_receive):
                    request = Mock(model="test-model", stream=True)
                    gen = wm.generate_stream(request)

                    results = list(gen)

                    # poll should be called before each receive
                    assert len(poll_calls) == 3
                    # poll call N should happen when receive count is N-1
                    assert poll_calls[0] == 0  # First poll before first receive
                    assert poll_calls[1] == 1  # Second poll after first receive
                    assert poll_calls[2] == 2  # Third poll after second receive

    def test_exception_propagates_after_cleanup(self, mock_config):
        """Original exception is re-raised after cleanup."""
        from src.orchestrator.worker_manager import WorkerManager, WorkerTimeoutError

        wm = WorkerManager(mock_config)

        mock_worker = Mock()
        # First poll: alive (for death check), then dead after timeout
        mock_worker.poll.side_effect = [None, 1]
        mock_worker.returncode = 1
        wm.active_worker = mock_worker
        wm.active_model_name = "test-model"

        with patch.object(wm, '_verify_worker_alive_locked'):
            with patch.object(wm, '_send_message'):
                with patch.object(wm, '_receive_message', side_effect=WorkerTimeoutError("timeout")):
                    with patch.object(wm, '_cleanup_dead_worker'):
                        request = Mock(model="test-model", stream=True)
                        gen = wm.generate_stream(request)

                        # Should raise the original WorkerTimeoutError, not a different exception
                        with pytest.raises(WorkerTimeoutError):
                            next(gen)


class TestStreamingLockBehavior:
    """Tests for lock behavior during streaming."""

    def test_lock_released_on_worker_death(self, mock_config):
        """Lock is properly released when worker dies."""
        from src.orchestrator.worker_manager import WorkerManager, WorkerError

        wm = WorkerManager(mock_config)

        mock_worker = Mock()
        mock_worker.poll.return_value = -9
        mock_worker.returncode = -9
        wm.active_worker = mock_worker
        wm.active_model_name = "test-model"

        with patch.object(wm, '_verify_worker_alive_locked'):
            with patch.object(wm, '_send_message'):
                with patch.object(wm, '_cleanup_dead_worker'):
                    request = Mock(model="test-model", stream=True)
                    gen = wm.generate_stream(request)

                    with pytest.raises(WorkerError):
                        next(gen)

        # Lock should be released - verify we can acquire it
        acquired = wm.lock.acquire(timeout=0.1)
        assert acquired, "Lock should be released after worker death"
        wm.lock.release()

    def test_activity_tracking_decremented_on_error(self, mock_config):
        """Activity counter is decremented even on error."""
        from src.orchestrator.worker_manager import WorkerManager, WorkerError

        wm = WorkerManager(mock_config)

        mock_worker = Mock()
        mock_worker.poll.return_value = -9
        mock_worker.returncode = -9
        wm.active_worker = mock_worker
        wm.active_model_name = "test-model"

        # Record initial activity count
        initial_count = wm.active_requests

        with patch.object(wm, '_verify_worker_alive_locked'):
            with patch.object(wm, '_send_message'):
                with patch.object(wm, '_cleanup_dead_worker'):
                    request = Mock(model="test-model", stream=True)
                    gen = wm.generate_stream(request)

                    with pytest.raises(WorkerError):
                        next(gen)

        # Activity count should be back to initial (context manager handles this)
        assert wm.active_requests == initial_count
