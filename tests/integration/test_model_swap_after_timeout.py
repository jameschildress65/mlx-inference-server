"""
Integration test for model swap after worker timeout.

Tests the fix for SharedMemoryBridge AttributeError that occurred when:
1. Worker times out during unload
2. Worker is force-killed
3. New model is loaded
4. Bridge object was reused with deleted attributes

This test validates the shared memory bridge lifecycle fixes.
"""

import pytest
import time
from unittest.mock import patch, MagicMock

from src.orchestrator.worker_manager import WorkerManager
from src.config.server_config import ServerConfig
from src.ipc.shared_memory_bridge import SharedMemoryIPCError


class TestModelSwapAfterTimeout:
    """Test model swap recovery after worker timeout."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        # Use auto-detection for configuration
        return ServerConfig.auto_detect()

    @pytest.fixture
    def manager(self, config):
        """Create WorkerManager instance."""
        manager = WorkerManager(config)
        yield manager
        # Cleanup
        try:
            manager.unload_model()
        except Exception:
            pass

    def test_model_swap_after_worker_timeout(self, manager):
        """
        Test that model swap works after worker cleanup.

        This replicates the production bug scenario:
        - Load model A
        - Force-kill worker (simulates timeout scenario)
        - Load model B
        - Verify no AttributeError

        The bug was: AttributeError: 'SharedMemoryBridge' object has no attribute 'resp_header'
        because bridge was reused after attributes were deleted.

        With the fix:
        - Bridge set to None BEFORE close()
        - close() is idempotent
        - Public methods check _closed state
        """
        # Load first model
        model_a = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
        result_a = manager.load_model(model_a)
        assert result_a is not None
        assert result_a.model_name == model_a
        assert manager.active_model_name == model_a

        # Force-kill worker (simulates timeout scenario)
        # This is the critical path that had the bug
        manager._kill_worker()

        # Verify bridge was cleaned up (set to None BEFORE close - the key fix)
        assert manager._shmem_bridge is None

        # Now load a different model (this is where the bug occurred)
        # With the old code, this would fail with AttributeError
        model_b = "mlx-community/Qwen2.5-3B-Instruct-4bit"
        result_b = manager.load_model(model_b)

        # Should succeed without AttributeError
        assert result_b is not None
        assert result_b.model_name == model_b
        assert manager.active_model_name == model_b
        assert manager._shmem_bridge is not None

        # Clean up
        manager.unload_model()

    def test_bridge_methods_raise_on_closed(self):
        """
        Test that bridge methods raise clear error when bridge is closed.

        Validates defensive checks in public methods.
        """
        from src.ipc.shared_memory_bridge import SharedMemoryBridge

        # Create and immediately close a bridge
        bridge = SharedMemoryBridge("test_bridge", is_server=True)
        bridge.close()

        # All public methods should raise SharedMemoryIPCError
        with pytest.raises(SharedMemoryIPCError, match="Bridge is closed"):
            bridge.send_request(b"test")

        with pytest.raises(SharedMemoryIPCError, match="Bridge is closed"):
            bridge.recv_request(timeout=1.0)

        with pytest.raises(SharedMemoryIPCError, match="Bridge is closed"):
            bridge.send_response(b"test")

        with pytest.raises(SharedMemoryIPCError, match="Bridge is closed"):
            bridge.recv_response(timeout=1.0)

        with pytest.raises(SharedMemoryIPCError, match="Bridge is closed"):
            bridge.write_image(b"fake_image_data")

    def test_bridge_close_is_idempotent(self):
        """
        Test that bridge close() can be called multiple times safely.

        Validates idempotent cleanup with try/except.
        """
        from src.ipc.shared_memory_bridge import SharedMemoryBridge

        bridge = SharedMemoryBridge("test_bridge_2", is_server=True)

        # Should not raise on multiple close() calls
        bridge.close()
        bridge.close()  # Second call
        bridge.close()  # Third call

        # Bridge should remain closed
        assert bridge._closed is True

    def test_cleanup_dead_worker_sets_bridge_to_none_first(self, manager):
        """
        Test that _cleanup_dead_worker sets bridge to None before closing.

        Validates fix to prevent race conditions.
        """
        # Load a model
        model = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
        manager.load_model(model)

        # Get reference to bridge
        bridge = manager._shmem_bridge
        assert bridge is not None

        # Call cleanup (simulates dead worker scenario)
        manager._cleanup_dead_worker()

        # Bridge should be set to None immediately
        assert manager._shmem_bridge is None

        # Bridge should be closed
        assert bridge._closed is True

    def test_kill_worker_sets_bridge_to_none_first(self, manager):
        """
        Test that _kill_worker sets bridge to None before closing.

        Validates fix in the kill worker path.
        """
        # Load a model
        model = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
        manager.load_model(model)

        # Get reference to bridge
        bridge = manager._shmem_bridge
        assert bridge is not None

        # Call kill (simulates force-kill scenario)
        manager._kill_worker()

        # Bridge should be set to None immediately
        assert manager._shmem_bridge is None

        # Bridge should be closed
        assert bridge._closed is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
