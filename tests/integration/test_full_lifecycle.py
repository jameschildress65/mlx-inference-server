"""Full lifecycle integration tests for V3.

Tests the complete workflow:
- Server startup
- Model loading (on-demand)
- Generation (non-streaming)
- Model unloading
- Worker health checks
"""

import pytest
import time
import os
from src.orchestrator.worker_manager import WorkerManager, NoModelLoadedError
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


class TestFullLifecycle:
    """End-to-end lifecycle tests."""

    @pytest.mark.skipif(
        os.getenv("MLX_SKIP_MODEL_TESTS") == "1",
        reason="Skipping model tests (MLX_SKIP_MODEL_TESTS=1)"
    )
    def test_complete_lifecycle(self, worker_manager):
        """
        Test complete lifecycle: load → generate → unload.

        This is the happy path that should always work.
        """
        model_path = os.getenv("MLX_TEST_MODEL", "mlx-community/Qwen2.5-0.5B-Instruct-4bit")

        # 1. Initial state: no model loaded
        status = worker_manager.get_status()
        assert not status["model_loaded"], "Should start with no model"
        assert status["model_name"] is None

        # 2. Load model
        load_result = worker_manager.load_model(model_path)
        assert load_result.model_name == model_path
        assert load_result.memory_gb > 0
        assert load_result.load_time > 0

        print(f"\nLoaded {model_path}: {load_result.memory_gb:.2f} GB in {load_result.load_time:.1f}s")

        # 3. Check status after load
        status = worker_manager.get_status()
        assert status["model_loaded"], "Model should be loaded"
        assert status["model_name"] == model_path
        assert status["memory_gb"] > 0

        # 4. Generate completion
        request = CompletionRequest(
            model=model_path,
            prompt="Write a haiku about MLX:",
            max_tokens=50,
            temperature=0.7,
            top_p=1.0,
            stream=False
        )

        response = worker_manager.generate(request)
        assert "text" in response
        assert response["text"], "Should generate non-empty text"
        assert response["tokens"] > 0
        assert response["finish_reason"] in ["stop", "length"]

        print(f"Generated: {response['text'][:100]}...")
        print(f"Tokens: {response['tokens']}")

        # 5. Health check while loaded
        health = worker_manager.health_check()
        assert health["healthy"], "Worker should be healthy"
        assert health["status"] == "healthy"

        # 6. Unload model
        unload_result = worker_manager.unload_model()
        assert unload_result.memory_freed_gb > 0
        assert unload_result.model_name == model_path

        print(f"Unloaded: freed {unload_result.memory_freed_gb:.2f} GB")

        # 7. Status after unload
        status = worker_manager.get_status()
        assert not status["model_loaded"], "Model should be unloaded"
        assert status["model_name"] is None
        assert status["memory_gb"] == 0.0

    @pytest.mark.skipif(
        os.getenv("MLX_SKIP_MODEL_TESTS") == "1",
        reason="Skipping model tests (MLX_SKIP_MODEL_TESTS=1)"
    )
    def test_multiple_generations(self, worker_manager):
        """
        Test multiple sequential generations with same model.

        Verifies worker stability across requests.
        """
        model_path = os.getenv("MLX_TEST_MODEL", "mlx-community/Qwen2.5-0.5B-Instruct-4bit")

        # Load once
        worker_manager.load_model(model_path)

        prompts = [
            "Count to 3:",
            "What is 2+2?",
            "Name a color:",
            "Say hello:",
            "Complete: The sky is"
        ]

        responses = []
        for i, prompt in enumerate(prompts):
            request = CompletionRequest(
                model=model_path,
                prompt=prompt,
                max_tokens=10,
                temperature=0.0,
                top_p=1.0,
                stream=False
            )

            response = worker_manager.generate(request)
            responses.append(response)

            print(f"{i+1}. '{prompt}' → '{response['text'][:50]}'")

            # Verify each response
            assert response["text"], f"Request {i+1} failed"
            assert response["tokens"] > 0

        # All 5 requests should succeed
        assert len(responses) == 5

        # Worker should still be healthy
        health = worker_manager.health_check()
        assert health["healthy"]

    @pytest.mark.skipif(
        os.getenv("MLX_SKIP_MODEL_TESTS") == "1",
        reason="Skipping model tests (MLX_SKIP_MODEL_TESTS=1)"
    )
    def test_model_switching(self, worker_manager):
        """
        Test switching between different models.

        Load model A → generate → unload
        Load model B → generate → unload
        """
        # Note: For quick testing, we'll use the same model path twice
        # In real scenarios, this would be different models
        model_a = os.getenv("MLX_TEST_MODEL", "mlx-community/Qwen2.5-0.5B-Instruct-4bit")
        model_b = os.getenv("MLX_TEST_MODEL_B", model_a)  # Same model for testing

        # Load model A
        result_a = worker_manager.load_model(model_a)
        assert result_a.model_name == model_a

        # Generate with A
        request = CompletionRequest(
            model=model_a,
            prompt="Hello from A",
            max_tokens=5,
            temperature=0.0,
            top_p=1.0,
            stream=False
        )
        response_a = worker_manager.generate(request)
        assert response_a["text"]

        # Switch to model B (unload A, load B)
        worker_manager.unload_model()
        result_b = worker_manager.load_model(model_b)
        assert result_b.model_name == model_b

        # Generate with B
        request = CompletionRequest(
            model=model_b,
            prompt="Hello from B",
            max_tokens=5,
            temperature=0.0,
            top_p=1.0,
            stream=False
        )
        response_b = worker_manager.generate(request)
        assert response_b["text"]

        # Cleanup
        worker_manager.unload_model()

        print(f"Model switching successful: A → B")

    def test_generate_without_model_fails(self, worker_manager):
        """
        Test that generation without loaded model raises error.
        """
        request = CompletionRequest(
            model="any-model",
            prompt="This should fail",
            max_tokens=10,
            temperature=0.0,
            top_p=1.0,
            stream=False
        )

        with pytest.raises(NoModelLoadedError):
            worker_manager.generate(request)

    def test_health_check_no_worker(self, worker_manager):
        """
        Test health check when no worker is running.
        """
        health = worker_manager.health_check()
        assert not health["healthy"]
        assert health["status"] == "no_worker"

    @pytest.mark.skipif(
        os.getenv("MLX_SKIP_MODEL_TESTS") == "1",
        reason="Skipping model tests (MLX_SKIP_MODEL_TESTS=1)"
    )
    def test_load_replaces_existing_model(self, worker_manager):
        """
        Test that loading a new model automatically unloads the current one.
        """
        model_path = os.getenv("MLX_TEST_MODEL", "mlx-community/Qwen2.5-0.5B-Instruct-4bit")

        # Load first model
        result1 = worker_manager.load_model(model_path)
        assert result1.model_name == model_path

        # Load same model again (should kill old worker, spawn new)
        result2 = worker_manager.load_model(model_path)
        assert result2.model_name == model_path

        # Should still work
        status = worker_manager.get_status()
        assert status["model_loaded"]
        assert status["model_name"] == model_path
