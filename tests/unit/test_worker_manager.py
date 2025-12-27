"""Unit tests for WorkerManager."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from src.orchestrator.worker_manager import (
    WorkerManager,
    NoModelLoadedError,
    WorkerError,
    ModelLoadResult,
    UnloadResult,
)


class TestModelLoadResult:
    """Tests for ModelLoadResult."""

    def test_create_result(self):
        """Test creating a ModelLoadResult."""
        result = ModelLoadResult(
            model_name="test-model",
            memory_gb=4.5,
            load_time=12.3
        )

        assert result.model_name == "test-model"
        assert result.memory_gb == 4.5
        assert result.load_time == 12.3

    def test_to_dict(self):
        """Test converting to dict."""
        result = ModelLoadResult(
            model_name="test-model",
            memory_gb=4.5,
            load_time=12.3
        )

        d = result.to_dict()
        assert d["model_name"] == "test-model"
        assert d["memory_gb"] == 4.5
        assert d["load_time"] == 12.3


class TestUnloadResult:
    """Tests for UnloadResult."""

    def test_create_result(self):
        """Test creating an UnloadResult."""
        result = UnloadResult(
            model_name="test-model",
            memory_freed_gb=4.5
        )

        assert result.model_name == "test-model"
        assert result.memory_freed_gb == 4.5

    def test_to_dict(self):
        """Test converting to dict."""
        result = UnloadResult(
            model_name="test-model",
            memory_freed_gb=4.5
        )

        d = result.to_dict()
        assert d["status"] == "success"
        assert d["model_unloaded"] == "test-model"
        assert d["memory_freed_gb"] == 4.5


class TestWorkerManager:
    """Tests for WorkerManager."""

    def test_init(self):
        """Test WorkerManager initialization."""
        config = Mock()
        manager = WorkerManager(config)

        assert manager.config == config
        assert manager.active_worker is None
        assert manager.active_model_name is None
        assert manager.active_memory_gb == 0.0

    def test_get_status_no_worker(self):
        """Test get_status with no active worker."""
        config = Mock()
        manager = WorkerManager(config)

        status = manager.get_status()

        assert status["model_loaded"] is False
        assert status["model_name"] is None
        assert status["memory_gb"] == 0.0

    def test_get_status_with_worker(self):
        """Test get_status with active worker."""
        config = Mock()
        manager = WorkerManager(config)

        # Simulate active worker
        manager.active_worker = Mock()
        manager.active_worker.poll.return_value = None  # Worker is alive
        manager.active_model_name = "test-model"
        manager.active_memory_gb = 4.5

        status = manager.get_status()

        assert status["model_loaded"] is True
        assert status["model_name"] == "test-model"
        assert status["memory_gb"] == 4.5

    def test_unload_with_no_worker_raises_error(self):
        """Test unload_model with no active worker."""
        config = Mock()
        manager = WorkerManager(config)

        with pytest.raises(NoModelLoadedError):
            manager.unload_model()

    def test_generate_with_no_worker_raises_error(self):
        """Test generate with no active worker."""
        config = Mock()
        manager = WorkerManager(config)

        from src.ipc.messages import CompletionRequest
        request = CompletionRequest(model="test", prompt="test")

        with pytest.raises(NoModelLoadedError):
            manager.generate(request)


class TestWorkerAbstractionLayer:
    """
    NASA-Grade Unit Tests for Worker Abstraction Layer.
    
    Tests the _get_worker_for_request() abstraction that enables
    future multi-worker support without API changes.
    
    Coverage:
    - Happy path: Worker available and healthy
    - Error path: No worker loaded
    - Error path: Worker died unexpectedly
    - Health check integration
    - Thread-safety assumptions
    """

    def test_get_worker_for_request_success(self):
        """Test _get_worker_for_request returns worker when available."""
        config = Mock()
        manager = WorkerManager(config)
        
        # Setup: Active, healthy worker
        mock_worker = Mock()
        mock_worker.poll.return_value = None  # Still alive
        manager.active_worker = mock_worker
        manager.active_model_name = "mlx-community/Qwen2.5-32B-Instruct-4bit"
        
        # Act: Get worker through abstraction
        worker = manager._get_worker_for_request("mlx-community/Qwen2.5-32B-Instruct-4bit")
        
        # Assert: Returns the active worker
        assert worker is mock_worker
        # Assert: Health check was performed
        mock_worker.poll.assert_called_once()

    def test_get_worker_for_request_no_worker_raises_error(self):
        """Test _get_worker_for_request raises NoModelLoadedError when no worker."""
        config = Mock()
        manager = WorkerManager(config)
        
        # Setup: No worker loaded
        manager.active_worker = None
        
        # Act & Assert: Should raise NoModelLoadedError with helpful message
        with pytest.raises(NoModelLoadedError) as exc_info:
            manager._get_worker_for_request("mlx-community/Qwen2.5-32B-Instruct-4bit")
        
        # Assert: Error message is helpful
        assert "No worker available" in str(exc_info.value)
        assert "mlx-community/Qwen2.5-32B-Instruct-4bit" in str(exc_info.value)

    def test_get_worker_for_request_dead_worker_raises_error(self):
        """Test _get_worker_for_request detects dead worker and cleans up."""
        config = Mock()
        manager = WorkerManager(config)

        # Setup: Worker exists but died (poll returns exit code)
        mock_worker = Mock()
        mock_worker.poll.return_value = 1  # Exit code 1 = dead
        mock_worker.returncode = 1
        mock_worker.pid = 12345  # For IO lock cleanup
        manager.active_worker = mock_worker
        manager.active_model_name = "mlx-community/Qwen2.5-32B-Instruct-4bit"

        # Act & Assert: New behavior - cleans up and retries, then raises NoModelLoadedError
        # (because after cleanup, no worker is available)
        with pytest.raises(NoModelLoadedError) as exc_info:
            manager._get_worker_for_request("mlx-community/Qwen2.5-32B-Instruct-4bit")

        # Assert: Worker was cleaned up
        assert manager.active_worker is None
        assert manager.active_model_name is None

    def test_get_worker_for_request_different_models(self):
        """Test abstraction works with different model names (future routing)."""
        config = Mock()
        manager = WorkerManager(config)
        
        # Setup: Worker with one model loaded
        mock_worker = Mock()
        mock_worker.poll.return_value = None
        manager.active_worker = mock_worker
        manager.active_model_name = "mlx-community/Qwen2.5-32B-Instruct-4bit"
        
        # Act: Request different model (currently returns same worker)
        worker1 = manager._get_worker_for_request("mlx-community/Qwen2.5-32B-Instruct-4bit")
        worker2 = manager._get_worker_for_request("mlx-community/Qwen2.5-72B-Instruct-4bit")
        
        # Assert: Currently returns same worker (single-worker impl)
        assert worker1 is mock_worker
        assert worker2 is mock_worker
        # Future: Will route to different workers based on model

    @patch('src.orchestrator.worker_manager.StdioBridge')
    def test_generate_uses_abstraction(self, mock_stdio):
        """Test that generate() method uses worker abstraction."""
        config = Mock()
        manager = WorkerManager(config)
        
        # Setup: Active worker
        mock_worker = Mock()
        mock_worker.poll.return_value = None
        manager.active_worker = mock_worker
        
        # Setup: Mock IPC response
        mock_response = Mock()
        mock_response.type = "completion_response"
        mock_response.text = "Test response"
        mock_response.tokens = 10
        mock_response.finish_reason = "stop"
        mock_stdio.receive_message.return_value = mock_response
        
        # Act: Generate via abstraction
        from src.ipc.messages import CompletionRequest
        request = CompletionRequest(
            model="mlx-community/Qwen2.5-32B-Instruct-4bit",
            prompt="Test prompt"
        )
        result = manager.generate(request)
        
        # Assert: Worker was obtained through abstraction
        # (implicitly tested by not raising NoModelLoadedError)
        assert result["text"] == "Test response"
        assert result["tokens"] == 10

    def test_health_check_uses_abstraction(self):
        """Test that health_check() uses worker abstraction."""
        config = Mock()
        manager = WorkerManager(config)
        
        # Setup: No worker
        manager.active_worker = None
        
        # Act: Health check via abstraction
        result = manager.health_check()
        
        # Assert: Returns proper status (abstraction handled NoModelLoadedError)
        assert result["healthy"] is False
        assert result["status"] == "no_worker"

    def test_abstraction_layer_is_thread_safe_assumption(self):
        """
        Document thread-safety assumption for abstraction layer.
        
        The _get_worker_for_request() method does NOT acquire locks itself,
        relying on callers (generate, health_check) to hold the lock.
        This is by design for performance and to avoid deadlocks.
        """
        config = Mock()
        manager = WorkerManager(config)
        
        # Setup: Worker
        mock_worker = Mock()
        mock_worker.poll.return_value = None
        manager.active_worker = mock_worker
        
        # Act: Get worker (no lock acquired inside)
        worker = manager._get_worker_for_request("test-model")
        
        # Assert: This test documents the threading contract
        # Callers MUST hold manager.lock before calling _get_worker_for_request()
        # This is tested in integration tests (test_concurrent_requests.py)
        assert worker is mock_worker

    def test_future_multi_worker_migration_path(self):
        """
        Document migration path to multi-worker (future).
        
        This test serves as documentation for future developers.
        To add multi-worker support:
        1. Change active_worker (single) → worker_pool (list)
        2. Update ONLY _get_worker_for_request() method
        3. API layer requires ZERO changes
        """
        # This is a documentation test - always passes
        # See: docs/WORKER-ABSTRACTION-LAYER.md for full migration guide
        assert True, "Migration path documented in WORKER-ABSTRACTION-LAYER.md"
