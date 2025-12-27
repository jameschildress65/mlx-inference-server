"""Integration tests for idle timeout functionality."""

import pytest
import time
import os
from src.orchestrator.worker_manager import WorkerManager
from src.config.server_config import ServerConfig


class TestIdleTimeout:
    """Test idle timeout and auto-unload functionality."""

    @pytest.fixture
    def short_timeout_config(self):
        """Config with very short idle timeout for testing."""
        config = ServerConfig.auto_detect()
        config.idle_timeout_seconds = 15  # 15 second timeout for testing
        return config

    @pytest.fixture
    def worker_manager(self, short_timeout_config):
        """WorkerManager with short idle timeout."""
        manager = WorkerManager(short_timeout_config)
        yield manager
        # Cleanup
        try:
            manager.shutdown()
        except Exception:
            pass

    def test_idle_time_tracking(self, worker_manager):
        """Test that idle time is tracked correctly."""
        model_path = os.getenv("MLX_TEST_MODEL", "mlx-community/Qwen2.5-0.5B-Instruct-4bit")

        # Load model
        worker_manager.load_model(model_path)

        # Check initial idle time (should be ~0)
        idle_time = worker_manager.get_idle_time()
        assert idle_time < 1.0, f"Initial idle time should be near 0, got {idle_time}s"

        # Wait 2 seconds
        time.sleep(2)

        # Check idle time increased
        idle_time = worker_manager.get_idle_time()
        assert 1.5 < idle_time < 3.0, f"Idle time should be ~2s, got {idle_time}s"

        print(f"Idle time tracking: ✓ ({idle_time:.1f}s)")

        # Cleanup
        worker_manager.unload_model()

    def test_activity_resets_idle_time(self, worker_manager):
        """Test that activity resets idle timer."""
        from src.ipc.messages import CompletionRequest

        model_path = os.getenv("MLX_TEST_MODEL", "mlx-community/Qwen2.5-0.5B-Instruct-4bit")

        # Load model
        worker_manager.load_model(model_path)

        # Wait 2 seconds
        time.sleep(2)
        idle_before = worker_manager.get_idle_time()
        assert idle_before > 1.5

        # Generate (activity)
        request = CompletionRequest(
            model=model_path,
            prompt="Test",
            max_tokens=5,
            temperature=0.7,
            top_p=1.0,
            stream=False
        )
        worker_manager.generate(request)

        # Idle time should reset to near 0
        idle_after = worker_manager.get_idle_time()
        assert idle_after < 0.5, f"Idle time should reset after activity, got {idle_after}s"

        print(f"Idle reset on activity: ✓ (before: {idle_before:.1f}s, after: {idle_after:.1f}s)")

        # Cleanup
        worker_manager.unload_model()

    def test_active_requests_prevent_unload(self, worker_manager):
        """Test that active requests prevent idle unload."""
        model_path = os.getenv("MLX_TEST_MODEL", "mlx-community/Qwen2.5-0.5B-Instruct-4bit")

        # Load model
        worker_manager.load_model(model_path)

        # Initially no active requests
        assert not worker_manager.has_active_requests()

        # Simulate active request (increment counter)
        worker_manager._increment_active_requests()
        assert worker_manager.has_active_requests()

        # Decrement
        worker_manager._decrement_active_requests()
        assert not worker_manager.has_active_requests()

        print("Active request tracking: ✓")

        # Cleanup
        worker_manager.unload_model()

    @pytest.mark.skipif(
        os.getenv("MLX_SKIP_SLOW_TESTS") == "1",
        reason="Slow test - skipped (set MLX_SKIP_SLOW_TESTS=0 to run)"
    )
    def test_idle_monitor_auto_unload(self, short_timeout_config):
        """Test that IdleMonitor auto-unloads after timeout (SLOW - 20s)."""
        from src.orchestrator.idle_monitor import IdleMonitor

        model_path = os.getenv("MLX_TEST_MODEL", "mlx-community/Qwen2.5-0.5B-Instruct-4bit")

        # Create worker manager and idle monitor
        worker_manager = WorkerManager(short_timeout_config)
        idle_monitor = IdleMonitor(
            worker_manager=worker_manager,
            idle_timeout_seconds=10,  # 10 second timeout
            check_interval_seconds=2   # Check every 2 seconds
        )

        try:
            # Start monitor
            idle_monitor.start()

            # Load model
            result = worker_manager.load_model(model_path)
            print(f"\nLoaded: {result.model_name} ({result.memory_gb:.2f} GB)")

            # Verify loaded
            status = worker_manager.get_status()
            assert status["model_loaded"]

            # Wait for timeout + margin (10s + 5s = 15s)
            print("Waiting 15s for idle timeout...")
            time.sleep(15)

            # Check if auto-unloaded
            status = worker_manager.get_status()
            if not status["model_loaded"]:
                print("Auto-unload successful: ✓")
            else:
                # Might need a bit more time for cleanup
                time.sleep(3)
                status = worker_manager.get_status()
                assert not status["model_loaded"], "Model should be auto-unloaded after idle timeout"
                print("Auto-unload successful (after grace period): ✓")

        finally:
            idle_monitor.stop()
            try:
                worker_manager.shutdown()
            except Exception:
                pass

    def test_idle_monitor_respects_active_requests(self, short_timeout_config):
        """Test that IdleMonitor doesn't unload during active requests."""
        from src.orchestrator.idle_monitor import IdleMonitor
        from src.ipc.messages import CompletionRequest

        model_path = os.getenv("MLX_TEST_MODEL", "mlx-community/Qwen2.5-0.5B-Instruct-4bit")

        # Create worker manager and idle monitor
        worker_manager = WorkerManager(short_timeout_config)
        idle_monitor = IdleMonitor(
            worker_manager=worker_manager,
            idle_timeout_seconds=5,   # Very short timeout
            check_interval_seconds=1   # Check frequently
        )

        try:
            # Start monitor
            idle_monitor.start()

            # Load model
            worker_manager.load_model(model_path)

            # Simulate active request (long-running)
            worker_manager._increment_active_requests()

            # Wait past timeout
            print("\nWaiting 8s (past 5s timeout) with active request...")
            time.sleep(8)

            # Should still be loaded (active request)
            status = worker_manager.get_status()
            assert status["model_loaded"], "Model should NOT unload during active request"
            print("Active request protection: ✓")

            # End request
            worker_manager._decrement_active_requests()

        finally:
            idle_monitor.stop()
            try:
                worker_manager.shutdown()
            except Exception:
                pass
