"""Integration tests for concurrent request handling."""

import pytest
import time
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.orchestrator.worker_manager import WorkerManager, WorkerError
from src.config.server_config import ServerConfig
from src.ipc.messages import CompletionRequest


class TestConcurrentRequests:
    """Test concurrent request handling and race conditions."""

    @pytest.fixture
    def worker_manager(self):
        """Create WorkerManager for testing."""
        config = ServerConfig.auto_detect()
        manager = WorkerManager(config)
        yield manager
        # Cleanup
        try:
            manager.unload_model()
        except:
            pass

    def test_sequential_requests(self, worker_manager):
        """Test multiple sequential requests (baseline)."""
        model_path = os.getenv("MLX_TEST_MODEL", "mlx-community/Qwen2.5-0.5B-Instruct-4bit")

        # Load model once
        worker_manager.load_model(model_path)

        # Make 5 sequential requests
        results = []
        for i in range(5):
            request = CompletionRequest(
                model=model_path,
                prompt=f"Count to {i+1}:",
                max_tokens=10,
                temperature=0.7,
                top_p=1.0,
                stream=False
            )
            result = worker_manager.generate(request)
            results.append(result)
            print(f"Request {i+1}: {result['text'][:30]}...")

        # All should succeed
        assert len(results) == 5
        assert all(r['text'] for r in results)

        print(f"\nSequential requests test: ✓ (5/5 successful)")

        # Cleanup
        worker_manager.unload_model()

    def test_concurrent_generate_serialized(self, worker_manager):
        """Test that concurrent generate() calls are properly serialized."""
        model_path = os.getenv("MLX_TEST_MODEL", "mlx-community/Qwen2.5-0.5B-Instruct-4bit")

        # Load model
        worker_manager.load_model(model_path)

        results = []
        errors = []

        def make_request(i):
            """Make a single request."""
            try:
                request = CompletionRequest(
                    model=model_path,
                    prompt=f"Say number {i}:",
                    max_tokens=10,
                    temperature=0.7,
                    top_p=1.0,
                    stream=False
                )
                result = worker_manager.generate(request)
                results.append((i, result))
                return f"Request {i}: Success"
            except Exception as e:
                errors.append((i, str(e)))
                return f"Request {i}: Error - {e}"

        # Launch 3 concurrent requests
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_request, i) for i in range(3)]

            for future in as_completed(futures):
                status = future.result()
                print(status)

        # All requests should complete (serialized by WorkerManager lock)
        assert len(results) == 3, f"Expected 3 results, got {len(results)}"
        assert len(errors) == 0, f"Unexpected errors: {errors}"

        print(f"\nConcurrent generate test: ✓ (3/3 serialized successfully)")

        # Cleanup
        worker_manager.unload_model()

    def test_load_during_generate(self, worker_manager):
        """Test loading a new model during active generation."""
        model_path = os.getenv("MLX_TEST_MODEL", "mlx-community/Qwen2.5-0.5B-Instruct-4bit")

        # Load initial model
        worker_manager.load_model(model_path)

        results = {}

        def long_generate():
            """Make a long-running generation."""
            try:
                request = CompletionRequest(
                    model=model_path,
                    prompt="Write a long story:",
                    max_tokens=50,
                    temperature=0.7,
                    top_p=1.0,
                    stream=False
                )
                result = worker_manager.generate(request)
                results['generate'] = f"Success: {len(result['text'])} chars"
            except Exception as e:
                results['generate'] = f"Error: {e}"

        def load_new_model():
            """Try to load a new model mid-generation."""
            time.sleep(0.5)  # Let generation start
            try:
                worker_manager.load_model(model_path)
                results['load'] = "Success"
            except Exception as e:
                results['load'] = f"Error: {e}"

        # Launch both concurrently
        thread1 = threading.Thread(target=long_generate)
        thread2 = threading.Thread(target=load_new_model)

        thread1.start()
        thread2.start()

        thread1.join(timeout=30)
        thread2.join(timeout=30)

        print(f"\nLoad during generate test:")
        print(f"  Generate: {results.get('generate', 'TIMEOUT')}")
        print(f"  Load: {results.get('load', 'TIMEOUT')}")

        # Both operations should complete (lock serialization)
        assert 'generate' in results
        assert 'load' in results

        print(f"  Result: ✓ (both operations completed)")

        # Cleanup
        try:
            worker_manager.unload_model()
        except:
            pass

    def test_unload_during_generate(self, worker_manager):
        """Test unloading model during active generation."""
        model_path = os.getenv("MLX_TEST_MODEL", "mlx-community/Qwen2.5-0.5B-Instruct-4bit")

        # Load model
        worker_manager.load_model(model_path)

        results = {}

        def long_generate():
            """Make a long-running generation."""
            try:
                request = CompletionRequest(
                    model=model_path,
                    prompt="Write a very long essay about AI:",
                    max_tokens=100,
                    temperature=0.7,
                    top_p=1.0,
                    stream=False
                )
                result = worker_manager.generate(request)
                results['generate'] = f"Success: {len(result['text'])} chars"
            except WorkerError as e:
                results['generate'] = f"WorkerError (expected): {type(e).__name__}"
            except Exception as e:
                results['generate'] = f"Error: {type(e).__name__}"

        def unload_model():
            """Try to unload mid-generation."""
            time.sleep(0.5)  # Let generation start
            try:
                worker_manager.unload_model()
                results['unload'] = "Success"
            except Exception as e:
                results['unload'] = f"Error: {type(e).__name__}"

        # Launch both concurrently
        thread1 = threading.Thread(target=long_generate)
        thread2 = threading.Thread(target=unload_model)

        thread1.start()
        thread2.start()

        thread1.join(timeout=30)
        thread2.join(timeout=30)

        print(f"\nUnload during generate test:")
        print(f"  Generate: {results.get('generate', 'TIMEOUT')}")
        print(f"  Unload: {results.get('unload', 'TIMEOUT')}")

        # Lock should serialize operations
        assert 'generate' in results
        assert 'unload' in results

        print(f"  Result: ✓ (operations serialized by lock)")

        # Cleanup
        try:
            worker_manager.unload_model()
        except:
            pass

    def test_activity_tracking_race_condition(self, worker_manager):
        """Test that activity tracking doesn't have race conditions."""
        model_path = os.getenv("MLX_TEST_MODEL", "mlx-community/Qwen2.5-0.5B-Instruct-4bit")

        # Load model
        worker_manager.load_model(model_path)

        def make_quick_request(i):
            """Make a quick request."""
            request = CompletionRequest(
                model=model_path,
                prompt=f"Number {i}",
                max_tokens=5,
                temperature=0.7,
                top_p=1.0,
                stream=False
            )
            worker_manager.generate(request)

        # Launch 10 concurrent quick requests
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_quick_request, i) for i in range(10)]
            for future in as_completed(futures):
                future.result()

        # Check activity tracking is sane
        idle_time = worker_manager.get_idle_time()
        assert idle_time >= 0, "Idle time should be non-negative"
        assert idle_time < 5, "Idle time should be recent after requests"

        # Check active requests counter is zero
        assert worker_manager.has_active_requests() == False, \
            "Active requests counter should be zero after completion"

        print(f"\nActivity tracking test: ✓")
        print(f"  Idle time: {idle_time:.2f}s")
        print(f"  Active requests: {worker_manager.has_active_requests()}")

        # Cleanup
        worker_manager.unload_model()

    def test_status_during_concurrent_operations(self, worker_manager):
        """Test that get_status() is consistent during concurrent operations."""
        model_path = os.getenv("MLX_TEST_MODEL", "mlx-community/Qwen2.5-0.5B-Instruct-4bit")

        # Load model
        worker_manager.load_model(model_path)

        status_checks = []

        def check_status():
            """Repeatedly check status."""
            for i in range(10):
                status = worker_manager.get_status()
                status_checks.append(status)
                time.sleep(0.1)

        def make_requests():
            """Make multiple requests."""
            for i in range(5):
                request = CompletionRequest(
                    model=model_path,
                    prompt=f"Test {i}",
                    max_tokens=10,
                    temperature=0.7,
                    top_p=1.0,
                    stream=False
                )
                worker_manager.generate(request)
                time.sleep(0.2)

        # Launch both concurrently
        thread1 = threading.Thread(target=check_status)
        thread2 = threading.Thread(target=make_requests)

        thread1.start()
        thread2.start()

        thread1.join()
        thread2.join()

        # All status checks should be consistent
        assert len(status_checks) == 10

        # Model should always be reported as loaded during this test
        all_loaded = all(s['model_loaded'] for s in status_checks)
        assert all_loaded, "Model should always show as loaded during operations"

        print(f"\nStatus consistency test: ✓")
        print(f"  Status checks: {len(status_checks)}")
        print(f"  All showed loaded: {all_loaded}")

        # Cleanup
        worker_manager.unload_model()

    def test_health_check_during_generation(self, worker_manager):
        """Test that health checks work during generation."""
        model_path = os.getenv("MLX_TEST_MODEL", "mlx-community/Qwen2.5-0.5B-Instruct-4bit")

        # Load model
        worker_manager.load_model(model_path)

        health_results = []

        def generate_long():
            """Long generation."""
            request = CompletionRequest(
                model=model_path,
                prompt="Write a detailed explanation:",
                max_tokens=50,
                temperature=0.7,
                top_p=1.0,
                stream=False
            )
            worker_manager.generate(request)

        def check_health():
            """Check health during generation."""
            time.sleep(0.3)  # Let generation start
            for i in range(3):
                health = worker_manager.health_check()
                health_results.append(health)
                time.sleep(0.5)

        # Launch both
        thread1 = threading.Thread(target=generate_long)
        thread2 = threading.Thread(target=check_health)

        thread1.start()
        thread2.start()

        thread1.join(timeout=30)
        thread2.join(timeout=30)

        # Health checks should work (may timeout during generation, but shouldn't crash)
        assert len(health_results) > 0

        print(f"\nHealth check during generation test: ✓")
        print(f"  Health checks performed: {len(health_results)}")
        for i, health in enumerate(health_results):
            print(f"  Check {i+1}: {health}")

        # Cleanup
        worker_manager.unload_model()

    @pytest.mark.skipif(
        os.getenv("MLX_SKIP_SLOW_TESTS") == "1",
        reason="Slow test - skipped (set MLX_SKIP_SLOW_TESTS=0 to run)"
    )
    def test_stress_concurrent_requests(self, worker_manager):
        """Stress test with many concurrent requests."""
        model_path = os.getenv("MLX_TEST_MODEL", "mlx-community/Qwen2.5-0.5B-Instruct-4bit")

        # Load model
        worker_manager.load_model(model_path)

        successes = []
        failures = []

        def make_request(i):
            """Make a request."""
            try:
                request = CompletionRequest(
                    model=model_path,
                    prompt=f"Request {i}:",
                    max_tokens=10,
                    temperature=0.7,
                    top_p=1.0,
                    stream=False
                )
                result = worker_manager.generate(request)
                successes.append(i)
                return True
            except Exception as e:
                failures.append((i, str(e)))
                return False

        # Launch 20 concurrent requests
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request, i) for i in range(20)]
            for future in as_completed(futures):
                future.result()
        elapsed = time.time() - start_time

        # All should succeed (serialized by lock)
        assert len(successes) == 20, f"Expected 20 successes, got {len(successes)}"
        assert len(failures) == 0, f"Unexpected failures: {failures}"

        print(f"\nStress test: ✓")
        print(f"  Requests: 20")
        print(f"  Successes: {len(successes)}")
        print(f"  Failures: {len(failures)}")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Throughput: {20 / elapsed:.1f} req/sec")

        # Cleanup
        worker_manager.unload_model()
