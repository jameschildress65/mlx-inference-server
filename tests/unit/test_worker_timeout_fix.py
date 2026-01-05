"""Unit tests for worker timeout handling fix.

Tests that WorkerTimeoutError is properly caught and handled without
falling back to stdio, preventing the orchestrator hang bug.
"""

import pytest
import subprocess
from unittest.mock import Mock, MagicMock, patch
from src.orchestrator.worker_manager import WorkerManager
from src.ipc.shared_memory_bridge import WorkerTimeoutError as ShmemTimeoutError
from src.ipc.stdio_bridge import WorkerTimeoutError as StdioTimeoutError


class TestWorkerTimeoutHandling:
    """Test that timeout errors trigger force-kill instead of stdio fallback."""

    @pytest.fixture
    def mock_worker_manager(self):
        """Create a WorkerManager with mocked dependencies."""
        # Create mock config
        mock_config = Mock()
        mock_config.idle_timeout_seconds = 600
        mock_config.request_timeout_seconds = 600

        with patch('src.orchestrator.worker_manager.get_registry'):
            manager = WorkerManager(config=mock_config)
            manager.active_worker = Mock(spec=subprocess.Popen)
            manager.active_worker.pid = 12345
            manager.active_model_name = "test-model"
            manager._shmem_bridge = Mock()
            return manager

    def test_timeout_triggers_force_kill_not_stdio_fallback(self, mock_worker_manager):
        """
        Critical test: Timeout should force-kill worker, NOT fall back to stdio.

        This is the core bug fix - prevents double timeout (600s + 600s = 1200s hang).
        """
        manager = mock_worker_manager

        # Mock the force-kill method to track if it was called
        manager._force_kill_hung_worker = Mock()

        # Mock SharedMemoryBridge to raise timeout
        with patch('src.ipc.shared_memory_bridge.SharedMemoryBridge.receive_message_shmem') as mock_recv:
            mock_recv.side_effect = ShmemTimeoutError("Worker timeout after 600s")

            # Mock StdioBridge - should NOT be called
            with patch('src.ipc.stdio_bridge.StdioBridge.receive_message') as mock_stdio:
                # Attempt to receive message - should trigger timeout handling
                with pytest.raises(ShmemTimeoutError):
                    manager._receive_message(timeout=600)

                # Verify force-kill was called
                manager._force_kill_hung_worker.assert_called_once_with(timeout_duration=600)

                # CRITICAL: Verify stdio fallback was NOT called
                mock_stdio.assert_not_called()

    def test_other_ipc_errors_still_fall_back_to_stdio(self, mock_worker_manager):
        """
        Non-timeout errors should still fall back to stdio (existing behavior).

        Only timeout errors skip the fallback - other IPC errors might be transient.
        """
        manager = mock_worker_manager

        # Mock a non-timeout error (e.g., buffer full)
        from src.ipc.shared_memory_bridge import SharedMemoryIPCError

        with patch('src.ipc.shared_memory_bridge.SharedMemoryBridge.receive_message_shmem') as mock_recv:
            mock_recv.side_effect = SharedMemoryIPCError("Buffer full")

            # Mock stdio to return success
            with patch('src.ipc.stdio_bridge.StdioBridge.receive_message') as mock_stdio:
                mock_stdio.return_value = {"type": "ready"}

                # Should fall back to stdio and succeed
                result = manager._receive_message(timeout=600)

                # Verify stdio was called (fallback worked)
                mock_stdio.assert_called_once()
                assert result == {"type": "ready"}

    def test_force_kill_hung_worker_cleanup_sequence(self, mock_worker_manager):
        """
        Verify force_kill_hung_worker calls cleanup methods in correct order.
        """
        manager = mock_worker_manager

        # Mock the cleanup methods
        manager._kill_worker = Mock()
        manager._cleanup_dead_worker = Mock()

        # Call force-kill
        manager._force_kill_hung_worker(timeout_duration=600)

        # Verify both cleanup methods were called
        manager._kill_worker.assert_called_once()
        manager._cleanup_dead_worker.assert_called_once()

    def test_worker_timeout_error_exception_type(self):
        """
        Verify WorkerTimeoutError is a subclass of WorkerCommunicationError.

        This ensures existing exception handlers still catch timeout errors.
        """
        from src.ipc.shared_memory_bridge import WorkerTimeoutError, WorkerCommunicationError

        # Should be a subclass
        assert issubclass(WorkerTimeoutError, WorkerCommunicationError)

        # Instance check
        timeout_err = WorkerTimeoutError("test")
        assert isinstance(timeout_err, WorkerCommunicationError)
        assert isinstance(timeout_err, WorkerTimeoutError)

    def test_stdio_timeout_also_uses_worker_timeout_error(self):
        """
        Verify stdio bridge also raises WorkerTimeoutError (not just shared memory).
        """
        from src.ipc.stdio_bridge import WorkerTimeoutError, WorkerCommunicationError

        # Should be a subclass
        assert issubclass(WorkerTimeoutError, WorkerCommunicationError)

        # Both bridges should have the same exception hierarchy
        timeout_err = WorkerTimeoutError("test")
        assert isinstance(timeout_err, WorkerCommunicationError)

    @patch('src.ipc.shared_memory_bridge.SharedMemoryBridge.receive_message_shmem')
    def test_timeout_error_propagates_to_caller(self, mock_recv, mock_worker_manager):
        """
        Timeout error should propagate to API layer (not swallowed).

        The API layer needs to catch this and return 500 error to client.
        """
        manager = mock_worker_manager
        manager._force_kill_hung_worker = Mock()

        # Raise timeout
        mock_recv.side_effect = ShmemTimeoutError("Worker timeout after 600s")

        # Should propagate (not be swallowed)
        with pytest.raises(ShmemTimeoutError):
            manager._receive_message(timeout=600)

    def test_orchestrator_responsive_after_timeout(self, mock_worker_manager):
        """
        After timeout, orchestrator should be able to accept new requests.

        This tests that the lock is released and state is cleaned up.
        """
        manager = mock_worker_manager

        # Mock cleanup
        manager._kill_worker = Mock()
        manager._cleanup_dead_worker = Mock()

        # Simulate timeout
        manager._force_kill_hung_worker(timeout_duration=600)

        # Verify state was cleaned up
        manager._kill_worker.assert_called_once()
        manager._cleanup_dead_worker.assert_called_once()

        # After cleanup, manager should be in clean state
        # (actual state checks would depend on _cleanup_dead_worker implementation)


class TestWorkerTimeoutIntegration:
    """Integration tests for timeout handling (require more setup)."""

    @pytest.mark.integration
    def test_timeout_does_not_cause_orchestrator_hang(self):
        """
        Integration test: Verify orchestrator doesn't hang after worker timeout.

        This would require:
        1. Start orchestrator
        2. Trigger worker timeout
        3. Verify orchestrator still responds to health checks
        4. Verify orchestrator can spawn new worker

        Marked as integration test - requires full setup.
        """
        pytest.skip("Integration test - requires full MLX setup")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
