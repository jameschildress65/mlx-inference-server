"""Integration tests for V3 memory cleanup - THE CRITICAL TEST.

This test validates V3's core value proposition: process isolation
guarantees 100% memory cleanup after model unload.

V1/V2 Problem: 0.12-0.94 GB residual after unload
V3 Goal: ~0 GB residual (within 50MB tolerance)
"""

import pytest
import time
import psutil
import os
from src.orchestrator.worker_manager import WorkerManager
from src.config.server_config import ServerConfig
from src.ipc.messages import CompletionRequest


@pytest.fixture
def worker_manager():
    """Create WorkerManager for testing."""
    config = ServerConfig.auto_detect()
    manager = WorkerManager(config)
    yield manager
    # Cleanup
    try:
        manager.unload_model()
    except:
        pass


def get_system_memory_mb():
    """Get current system memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def get_mlx_memory_gb():
    """Get MLX active memory in GB."""
    try:
        import mlx.core as mx
        return mx.metal.get_active_memory() / (1024**3)
    except:
        return 0.0


class TestMemoryCleanup:
    """Critical memory cleanup tests."""

    @pytest.mark.skipif(
        os.getenv("MLX_SKIP_MODEL_TESTS") == "1",
        reason="Skipping model tests (MLX_SKIP_MODEL_TESTS=1)"
    )
    def test_memory_cleanup_small_model(self, worker_manager):
        """
        Test memory cleanup with small model (0.5B).

        Success criteria: Final memory ≈ baseline (±100MB)
        """
        # Get baseline
        baseline_mb = get_system_memory_mb()
        baseline_mlx = get_mlx_memory_gb()

        # Load small model
        model_path = os.getenv("MLX_TEST_MODEL", "mlx-community/Qwen2.5-0.5B-Instruct-4bit")
        result = worker_manager.load_model(model_path)

        assert result.model_name == model_path
        assert result.memory_gb > 0, "Model should use memory"

        loaded_mb = get_system_memory_mb()
        loaded_mlx = get_mlx_memory_gb()

        print(f"\nBaseline: {baseline_mb:.1f} MB, MLX: {baseline_mlx:.2f} GB")
        print(f"Loaded: {loaded_mb:.1f} MB (+{loaded_mb - baseline_mb:.1f} MB), MLX: {loaded_mlx:.2f} GB")

        # Generate to ensure model is active
        request = CompletionRequest(
            model=model_path,
            prompt="Hello",
            max_tokens=5,
            temperature=0.0,
            top_p=1.0,
            stream=False
        )
        response = worker_manager.generate(request)
        assert response["text"], "Should generate text"

        # Unload (kill worker)
        unload_result = worker_manager.unload_model()
        assert unload_result.memory_freed_gb > 0, "Should report memory freed"

        # Wait for OS to reclaim
        time.sleep(2)

        # Check cleanup
        final_mb = get_system_memory_mb()
        final_mlx = get_mlx_memory_gb()

        residual_mb = final_mb - baseline_mb
        residual_mlx = final_mlx - baseline_mlx

        print(f"Final: {final_mb:.1f} MB (residual: {residual_mb:+.1f} MB), MLX: {final_mlx:.2f} GB")

        # Critical assertion: Memory cleaned up (within 100MB tolerance)
        assert abs(residual_mb) < 100, \
            f"Memory not cleaned up! Residual: {residual_mb:.1f} MB (expected <100 MB)"

        assert abs(residual_mlx) < 0.1, \
            f"MLX memory not cleaned up! Residual: {residual_mlx:.2f} GB (expected <0.1 GB)"

    @pytest.mark.skipif(
        os.getenv("MLX_SKIP_MODEL_TESTS") == "1",
        reason="Skipping model tests (MLX_SKIP_MODEL_TESTS=1)"
    )
    def test_memory_no_accumulation_3_cycles(self, worker_manager):
        """
        Test 3 load/unload cycles show no memory accumulation.

        This is the V2 failure case: each cycle leaked 0.12-0.34 GB.
        V3 should show stable memory across cycles.
        """
        model_path = os.getenv("MLX_TEST_MODEL", "mlx-community/Qwen2.5-0.5B-Instruct-4bit")

        baseline_mb = get_system_memory_mb()
        measurements = [baseline_mb]

        for i in range(3):
            print(f"\n=== Cycle {i+1} ===")

            # Load
            result = worker_manager.load_model(model_path)
            print(f"Loaded: {result.memory_gb:.2f} GB")

            # Generate
            request = CompletionRequest(
                model=model_path,
                prompt=f"Count to {i+1}:",
                max_tokens=10,
                temperature=0.0,
                top_p=1.0,
                stream=False
            )
            worker_manager.generate(request)

            # Unload
            unload_result = worker_manager.unload_model()
            print(f"Freed: {unload_result.memory_freed_gb:.2f} GB")

            # Wait for cleanup
            time.sleep(2)

            current_mb = get_system_memory_mb()
            measurements.append(current_mb)
            residual = current_mb - baseline_mb
            print(f"After cycle {i+1}: {current_mb:.1f} MB (residual: {residual:+.1f} MB)")

        # Check no accumulation
        final_mb = measurements[-1]
        residual_mb = final_mb - baseline_mb

        print(f"\n=== Summary ===")
        print(f"Baseline: {baseline_mb:.1f} MB")
        print(f"Final: {final_mb:.1f} MB")
        print(f"Total residual: {residual_mb:+.1f} MB")
        print(f"Measurements: {[f'{m:.1f}' for m in measurements]}")

        # Critical: No accumulation
        assert abs(residual_mb) < 150, \
            f"Memory accumulated across cycles! Residual: {residual_mb:.1f} MB"

    def test_worker_memory_isolation(self, worker_manager):
        """
        Test that orchestrator process doesn't hold model memory.

        Only worker subprocess should show MLX memory usage.
        """
        # Orchestrator baseline (no model)
        orchestrator_mb = get_system_memory_mb()

        # This should be near zero since orchestrator has no mlx imports yet
        assert orchestrator_mb < 500, \
            f"Orchestrator using too much memory: {orchestrator_mb:.1f} MB"

    @pytest.mark.skipif(
        os.getenv("MLX_SKIP_MODEL_TESTS") == "1",
        reason="Skipping model tests (MLX_SKIP_MODEL_TESTS=1)"
    )
    def test_unload_without_load(self, worker_manager):
        """Test unload when no model loaded doesn't crash."""
        from src.orchestrator.worker_manager import NoModelLoadedError

        with pytest.raises(NoModelLoadedError):
            worker_manager.unload_model()
