"""Worker crash recovery tests.

Tests V3's critical feature: orchestrator survives worker crashes.

V1/V2: Worker crash = server death
V3: Worker crash = isolated failure, server continues
"""

import pytest
import time
import os
import signal
from src.orchestrator.worker_manager import WorkerManager, WorkerError
from src.config.server_config import ServerConfig
from src.ipc.messages import CompletionRequest


@pytest.fixture
def config():
    """Get server configuration."""
    return ServerConfig.auto_detect()


@pytest.fixture
def worker_manager(config):
    """Create WorkerManager for testing."""
    manager = WorkerManager(config)
    yield manager
    # Cleanup
    try:
        manager.unload_model()
    except:
        pass


class TestCrashRecovery:
    """Worker crash and recovery tests."""

    @pytest.mark.skipif(
        os.getenv("MLX_SKIP_MODEL_TESTS") == "1",
        reason="Skipping model tests (MLX_SKIP_MODEL_TESTS=1)"
    )
    def test_worker_crash_orchestrator_survives(self, worker_manager):
        """
        Test that orchestrator survives when worker crashes.

        This is V3's killer feature: process isolation.
        """
        model_path = os.getenv("MLX_TEST_MODEL", "mlx-community/Qwen2.5-0.5B-Instruct-4bit")

        # Load model
        worker_manager.load_model(model_path)

        # Get worker PID
        status = worker_manager.get_status()
        assert status["model_loaded"]

        worker_pid = worker_manager.active_worker.pid
        print(f"\nWorker PID: {worker_pid}")

        # Kill worker suddenly (SIGKILL)
        os.kill(worker_pid, signal.SIGKILL)
        time.sleep(1)

        # Worker should be dead
        assert worker_manager.active_worker.poll() is not None
        print("Worker killed successfully")

        # Status should reflect crashed worker
        status = worker_manager.get_status()
        assert not status["model_loaded"]

        # Health check should show unhealthy
        health = worker_manager.health_check()
        assert not health["healthy"]

        print("Orchestrator survived worker crash ✓")

    @pytest.mark.skipif(
        os.getenv("MLX_SKIP_MODEL_TESTS") == "1",
        reason="Skipping model tests (MLX_SKIP_MODEL_TESTS=1)"
    )
    def test_generate_after_crash_fails_gracefully(self, worker_manager):
        """
        Test that generation after worker crash fails gracefully.

        Should raise WorkerError, not crash orchestrator.
        """
        model_path = os.getenv("MLX_TEST_MODEL", "mlx-community/Qwen2.5-0.5B-Instruct-4bit")

        # Load model
        worker_manager.load_model(model_path)

        # Kill worker
        worker_pid = worker_manager.active_worker.pid
        os.kill(worker_pid, signal.SIGKILL)
        time.sleep(1)

        # Try to generate (should fail gracefully)
        request = CompletionRequest(
            model=model_path,
            prompt="This will fail",
            max_tokens=10,
            temperature=0.0,
            top_p=1.0,
            stream=False
        )

        with pytest.raises((WorkerError, Exception)) as exc_info:
            worker_manager.generate(request)

        print(f"Failed gracefully with: {exc_info.value}")

        # Orchestrator should still be functional
        # Can load new model
        result = worker_manager.load_model(model_path)
        assert result.model_name == model_path
        print("Recovered: loaded new model after crash ✓")

    @pytest.mark.skipif(
        os.getenv("MLX_SKIP_MODEL_TESTS") == "1",
        reason="Skipping model tests (MLX_SKIP_MODEL_TESTS=1)"
    )
    def test_graceful_shutdown(self, worker_manager):
        """
        Test graceful worker shutdown with SIGTERM.

        Worker should cleanup and exit(0) on SIGTERM.
        """
        model_path = os.getenv("MLX_TEST_MODEL", "mlx-community/Qwen2.5-0.5B-Instruct-4bit")

        # Load model
        worker_manager.load_model(model_path)
        worker_pid = worker_manager.active_worker.pid

        # Send SIGTERM (graceful shutdown)
        os.kill(worker_pid, signal.SIGTERM)
        time.sleep(2)

        # Worker should have exited gracefully
        assert worker_manager.active_worker.poll() is not None

        # Check exit code (should be 0 for graceful, or -15 for SIGTERM)
        exit_code = worker_manager.active_worker.poll()
        print(f"Worker exit code: {exit_code}")
        # Either 0 (graceful) or -15 (terminated) is acceptable
        assert exit_code in [0, -15, 143], f"Unexpected exit code: {exit_code}"

    @pytest.mark.skipif(
        os.getenv("MLX_SKIP_MODEL_TESTS") == "1",
        reason="Skipping model tests (MLX_SKIP_MODEL_TESTS=1)"
    )
    def test_unload_kills_worker_completely(self, worker_manager):
        """
        Test that unload() kills worker process completely.

        Worker PID should not exist after unload.
        """
        model_path = os.getenv("MLX_TEST_MODEL", "mlx-community/Qwen2.5-0.5B-Instruct-4bit")

        # Load model
        worker_manager.load_model(model_path)
        worker_pid = worker_manager.active_worker.pid

        print(f"\nWorker PID before unload: {worker_pid}")

        # Unload
        worker_manager.unload_model()

        # Wait a bit for process cleanup
        time.sleep(1)

        # Worker process should not exist
        try:
            os.kill(worker_pid, 0)  # Signal 0 = check if process exists
            pytest.fail(f"Worker process {worker_pid} still exists after unload!")
        except OSError:
            # Expected: process doesn't exist
            print(f"Worker process {worker_pid} killed successfully ✓")

    @pytest.mark.skipif(
        os.getenv("MLX_SKIP_MODEL_TESTS") == "1",
        reason="Skipping model tests (MLX_SKIP_MODEL_TESTS=1)"
    )
    def test_load_while_worker_running_kills_old_worker(self, worker_manager):
        """
        Test that loading a new model kills the old worker first.

        No orphan processes should remain.
        """
        model_path = os.getenv("MLX_TEST_MODEL", "mlx-community/Qwen2.5-0.5B-Instruct-4bit")

        # Load first worker
        worker_manager.load_model(model_path)
        first_pid = worker_manager.active_worker.pid
        print(f"\nFirst worker PID: {first_pid}")

        # Load second worker (should kill first)
        worker_manager.load_model(model_path)
        second_pid = worker_manager.active_worker.pid
        print(f"Second worker PID: {second_pid}")

        # PIDs should be different
        assert first_pid != second_pid

        # First worker should be dead
        time.sleep(1)
        try:
            os.kill(first_pid, 0)
            pytest.fail(f"Old worker {first_pid} still exists!")
        except OSError:
            print(f"Old worker {first_pid} killed successfully ✓")

        # Second worker should be alive
        assert worker_manager.active_worker.poll() is None
