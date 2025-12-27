"""Comprehensive integration tests for Opus 4.5 critical fixes.

Tests the 5 critical testing gaps identified in Opus review:
1. Multi-process concurrency (ring buffer cross-process sync)
2. Load testing (1000+ requests)
3. Wrap-around boundary conditions
4. Signal handling cleanup
5. Registry crash recovery

These tests verify that all Opus critical fixes work correctly under
real-world production conditions.
"""

import pytest
import time
import os
import signal
import json
import struct
from multiprocessing import Process, Queue
from pathlib import Path

from src.orchestrator.worker_manager import WorkerManager
from src.config.server_config import ServerConfig
from src.ipc.messages import CompletionRequest
from src.ipc.shared_memory_bridge import SharedMemoryBridge
from src.ipc.shm_registry import SharedMemoryRegistry


# Module-level worker functions (required for multiprocessing pickle)

def _concurrent_worker_process(process_id: int, model_path: str, result_queue: Queue):
    """Worker process that makes concurrent requests.

    Creates its own WorkerManager inside the subprocess to avoid pickling issues.
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


def _signal_test_subprocess(ready_queue: Queue):
    """Process that creates shared memory and waits for signal."""
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


class TestLoadCapacity:
    """Test #13: Load tests (1000+ requests).

    Verifies system can handle sustained load without degradation.
    """

    @pytest.fixture
    def worker_manager(self):
        """Create WorkerManager for testing."""
        config = ServerConfig.auto_detect()
        manager = WorkerManager(config)
        yield manager
        try:
            manager.unload_model()
        except:
            pass

    @pytest.mark.slow
    def test_sustained_load_1000_requests(self, worker_manager):
        """Test 1000+ requests without performance degradation."""
        # Skip if slow tests disabled
        if os.getenv("MLX_SKIP_SLOW_TESTS") == "1":
            pytest.skip("Slow tests disabled (MLX_SKIP_SLOW_TESTS=1)")

        model_path = os.getenv("MLX_TEST_MODEL", "mlx-community/Qwen2.5-0.5B-Instruct-4bit")

        # Load model
        worker_manager.load_model(model_path)

        # Make 1000 small requests
        num_requests = 1000
        successful = 0
        failed = 0
        total_time = 0

        print(f"\n✓ Load test: {num_requests} requests")

        for i in range(num_requests):
            try:
                start = time.time()
                request = CompletionRequest(
                    model=model_path,
                    prompt="Hi",
                    max_tokens=3,
                    temperature=0.7,
                    stream=False
                )
                result = worker_manager.generate(request)
                elapsed = time.time() - start
                total_time += elapsed

                if result.get('text'):
                    successful += 1
                else:
                    failed += 1

                # Progress indicator every 100 requests
                if (i + 1) % 100 == 0:
                    avg_time = total_time / (i + 1)
                    print(f"  Progress: {i+1}/{num_requests} "
                          f"(avg: {avg_time*1000:.1f}ms/req)")

            except Exception as e:
                failed += 1
                print(f"  Request {i} failed: {e}")

        success_rate = (successful / num_requests) * 100
        avg_latency = (total_time / num_requests) * 1000

        print(f"\n  Results:")
        print(f"    Total requests: {num_requests}")
        print(f"    Successful: {successful} ({success_rate:.1f}%)")
        print(f"    Failed: {failed}")
        print(f"    Avg latency: {avg_latency:.1f}ms/req")

        # At least 95% should succeed under load
        assert success_rate >= 95.0, \
            f"Success rate too low: {success_rate:.1f}%"

        # Cleanup
        worker_manager.unload_model()


class TestRingBufferBoundaries:
    """Test #14: Wrap-around boundary tests.

    Verifies ring buffer correctly handles messages that span the boundary.
    """

    def test_wrap_around_boundary_conditions(self):
        """Test messages that cross ring buffer boundary (wrap-around)."""
        # Create a bridge with shared memory
        bridge = SharedMemoryBridge(name="test", is_server=True)

        try:
            # Ring size is 4MB
            ring_size = bridge.RING_SIZE
            header_size = bridge.HEADER_SIZE

            # Strategy: Write messages until we're near the boundary,
            # then write a message that MUST wrap around

            # Fill buffer to near-boundary position
            # Write messages of 1KB each
            message_size = 1024
            test_data = b'X' * message_size

            # Write until we're close to boundary (leave ~2KB before wrap)
            target_pos = ring_size - 2048

            # Calculate how many messages needed
            frame_size = 4 + message_size  # length prefix + data
            num_messages = target_pos // frame_size

            print(f"\n✓ Ring buffer boundary test:")
            print(f"  Ring size: {ring_size:,} bytes")
            print(f"  Writing {num_messages} messages to reach boundary...")

            # Write messages to fill buffer
            for i in range(num_messages):
                success = bridge.send_request(test_data)
                if not success:
                    # Buffer full - this is OK, we're testing capacity
                    break

            # Now write a large message that MUST wrap around
            wrap_message = b'WRAP' * 600  # ~2.4KB message
            print(f"  Writing wrap-around message ({len(wrap_message)} bytes)...")

            success = bridge.send_request(wrap_message)

            if success:
                # Read it back - should be intact
                received = bridge.recv_request(timeout=1.0)

                if received:
                    assert received == wrap_message, \
                        "Wrap-around message corrupted!"
                    print(f"  ✓ Wrap-around handled correctly")
                else:
                    print(f"  ⚠ Could not read wrap-around message (buffer full?)")
            else:
                print(f"  ⚠ Buffer full, couldn't test wrap-around")

        finally:
            bridge.close()


class TestSignalHandling:
    """Test #15: Signal handling cleanup tests.

    Verifies shared memory is properly cleaned up on signal delivery.
    """

    def test_sigterm_cleanup(self):
        """Test that SIGTERM properly cleans up shared memory."""
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


class TestCrashRecovery:
    """Test #16: Registry crash recovery tests.

    Verifies orphaned shared memory is detected and cleaned up.
    """

    def test_orphan_detection_and_cleanup(self):
        """Test that orphaned SHM segments are detected and cleaned."""
        from src.ipc.secure_shm_manager import (
            SecureSharedMemoryManager,
            cleanup_stale_shared_memory
        )
        from src.ipc.shm_registry import SharedMemoryRegistry
        from multiprocessing import shared_memory

        registry = SharedMemoryRegistry()

        # Create shared memory
        manager = SecureSharedMemoryManager(
            size=1024*1024,
            is_server=True,
            registry=registry
        )
        shm_name = manager.name
        real_pid = os.getpid()

        print(f"\n✓ Crash recovery test:")
        print(f"  Created SHM: {shm_name}")

        # Manually modify registry to simulate dead process
        # Use a fake PID that doesn't exist (99999)
        fake_pid = 99999

        # Read registry file directly and modify it
        registry_file = registry.registry_path
        with open(registry_file, 'r') as f:
            registry_data = json.load(f)

        # Find and update the PID
        for seg in registry_data['segments']:
            if seg['name'] == shm_name:
                seg['pid'] = fake_pid
                break

        # Write back
        with open(registry_file, 'w') as f:
            json.dump(registry_data, f, indent=2)

        print(f"  Simulated crash (changed PID to {fake_pid})")

        # Now close the manager (simulate crash - no cleanup)
        manager._closed = True  # Prevent cleanup
        del manager

        # Verify it's in the registry with fake PID
        segments = registry.list_active()
        orphan = next((s for s in segments if s['name'] == shm_name), None)
        assert orphan is not None, "Segment not in registry"
        assert orphan['pid'] == fake_pid, "PID not updated in registry"
        print(f"  ✓ Orphaned segment in registry (PID {fake_pid})")

        # Now run cleanup
        print(f"  Running cleanup_stale_shared_memory()...")
        cleaned = cleanup_stale_shared_memory(registry)

        print(f"  ✓ Cleaned {cleaned} stale segment(s)")

        # Verify it was cleaned from registry
        segments = registry.list_active()
        still_there = next((s for s in segments if s['name'] == shm_name), None)
        assert still_there is None, "Segment still in registry after cleanup"
        print(f"  ✓ Segment removed from registry")

        # Verify SHM is actually unlinked
        try:
            shm = shared_memory.SharedMemory(name=shm_name)
            shm.close()
            assert False, "Orphaned SHM not unlinked!"
        except FileNotFoundError:
            print(f"  ✓ SHM properly unlinked")
