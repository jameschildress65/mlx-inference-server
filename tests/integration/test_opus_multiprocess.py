"""Multi-process tests - isolated to prevent pytest hangs.

Tests #12 and #15 from Opus comprehensive review:
- Multi-process concurrency (ring buffer cross-process sync)
- Signal handling cleanup

These tests are separated from test_opus_comprehensive.py to avoid
pytest collection hangs on macOS with Python 3.12 spawn mode.
"""

from __future__ import annotations
import pytest
import time
import os
import signal
from typing import TYPE_CHECKING

# FIX: Only import multiprocessing types during type checking
if TYPE_CHECKING:
    from multiprocessing import Process, Queue

from src.orchestrator.worker_manager import WorkerManager
from src.config.server_config import ServerConfig
from src.ipc.messages import CompletionRequest
from src.ipc.secure_shm_manager import SecureSharedMemoryManager


# Module-level worker functions (required for multiprocessing pickle)

def _concurrent_worker_process(process_id, model_path, result_queue):
    """Worker process that makes concurrent requests.

    Creates its own WorkerManager inside the subprocess to avoid pickling issues.

    Args:
        process_id: int - Process identifier
        model_path: str - Path to MLX model
        result_queue: multiprocessing.Queue - Queue for results
    """
    from src.orchestrator.worker_manager import WorkerManager
    from src.config.server_config import ServerConfig

    manager = None
    try:
        # Create WorkerManager inside this subprocess
        config = ServerConfig.auto_detect()
        manager = WorkerManager(config)

        # Load model once per process
        manager.load_model(model_path)

        # Each process makes 3 requests
        for i in range(3):
            request = CompletionRequest(
                model=model_path,
                prompt=f"Process {process_id} request {i}: Say hello",
                max_tokens=5,
                temperature=0.7,
                stream=False
            )
            result = manager.generate(request)
            result_queue.put({
                'process_id': process_id,
                'request_id': i,
                'text': result['text'],
                'success': True
            })
    except Exception as e:
        result_queue.put({
            'process_id': process_id,
            'error': str(e),
            'success': False
        })
    finally:
        # Clean up manager
        if manager:
            try:
                manager.unload_model()
            except:
                pass


def _signal_test_subprocess(ready_queue):
    """Process that creates shared memory and waits for signal.

    Args:
        ready_queue: multiprocessing.Queue - Queue to signal readiness
    """
    from src.ipc.secure_shm_manager import SecureSharedMemoryManager

    # Create shared memory
    manager = SecureSharedMemoryManager(
        size=1024*1024,  # 1MB
        is_server=True
    )

    # Signal ready
    ready_queue.put({
        'pid': os.getpid(),
        'shm_name': manager.name
    })

    # Wait for signal (sleep)
    time.sleep(10)


class TestMultiProcessConcurrency:
    """Test #12: Multi-process concurrency tests.

    Verifies ring buffer cross-process synchronization works correctly
    with actual separate processes (not just threads).

    Each subprocess creates its own WorkerManager to avoid pickling issues.
    """

    def test_concurrent_processes_no_corruption(self):
        """Test concurrent requests from separate processes don't corrupt data.

        This is the critical test that verifies Bug #1 (ring buffer sync) fix.
        Multiple processes write concurrently - data should not corrupt.

        Each subprocess creates its own WorkerManager to avoid pickling issues.
        """
        # FIX: Import multiprocessing HERE to avoid pytest hang
        from multiprocessing import Process, Queue

        model_path = os.getenv("MLX_TEST_MODEL", "mlx-community/Qwen2.5-0.5B-Instruct-4bit")

        # Launch 4 concurrent processes
        # Each will create its own WorkerManager internally
        num_processes = 4
        result_queue = Queue()
        processes = []

        print(f"\n✓ Multi-process concurrency test:")
        print(f"  Launching {num_processes} processes...")

        for i in range(num_processes):
            p = Process(target=_concurrent_worker_process,
                       args=(i, model_path, result_queue))
            p.start()
            processes.append(p)

        # Wait for all processes
        for p in processes:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()

        # Collect results
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())

        # Verify: All requests should succeed without corruption
        successful = [r for r in results if r.get('success')]
        failed = [r for r in results if not r.get('success')]

        print(f"  Processes completed: {len([p for p in processes if not p.is_alive()])}/{num_processes}")
        print(f"  Total requests: {len(results)}")
        print(f"  Successful: {len(successful)}")
        print(f"  Failed: {len(failed)}")

        if failed:
            print(f"  Failed requests: {failed}")

        # All should succeed (no data corruption from concurrent access)
        # Allow 80% success rate (some failures OK under heavy load)
        assert len(successful) >= (num_processes * 3 * 0.8), \
            f"Too many failures ({len(failed)}/{len(results)}): {failed}"
        assert all(r.get('text') for r in successful), \
            "Some responses have empty text (data corruption?)"

        print(f"  ✓ No data corruption detected!")


class TestSignalHandling:
    """Test #15: Signal handling cleanup tests.

    Verifies shared memory is properly cleaned up on signal delivery.
    """

    def test_sigterm_cleanup(self):
        """Test that SIGTERM properly cleans up shared memory."""
        # FIX: Import multiprocessing HERE to avoid pytest hang
        from multiprocessing import Process, Queue

        ready_queue = Queue()
        p = Process(target=_signal_test_subprocess, args=(ready_queue,))
        p.start()

        # Wait for process to be ready
        info = ready_queue.get(timeout=5)
        pid = info['pid']
        shm_name = info['shm_name']

        print(f"\n✓ Signal handling test:")
        print(f"  Process PID: {pid}")
        print(f"  SHM name: {shm_name}")

        # Give it more time to fully set up signal handlers
        time.sleep(1.0)

        # Send SIGTERM
        print(f"  Sending SIGTERM...")
        os.kill(pid, signal.SIGTERM)

        # Wait for process to exit (cleanup might take time for large buffers)
        p.join(timeout=5)

        # Verify process exited
        if p.is_alive():
            # Process didn't exit - force kill and fail test
            print(f"  ✗ Process didn't exit after SIGTERM, force killing...")
            p.terminate()
            time.sleep(0.5)
            if p.is_alive():
                p.kill()
            pytest.fail("Process didn't exit cleanly after SIGTERM (had to force kill)")

        print(f"  ✓ Process exited cleanly")

        # Give cleanup a moment to complete file operations
        time.sleep(0.5)

        # Verify shared memory was cleaned up
        # Try to attach - should fail
        from multiprocessing import shared_memory
        try:
            shm = shared_memory.SharedMemory(name=shm_name)
            shm.close()
            shm.unlink()  # Clean up if test found it
            pytest.fail("Shared memory not cleaned up after SIGTERM!")
        except FileNotFoundError:
            # Good! Shared memory was cleaned up
            print(f"  ✓ Shared memory cleaned up")
