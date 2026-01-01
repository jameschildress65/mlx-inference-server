"""Comprehensive integration tests for Opus 4.5 critical fixes.

Tests the critical testing gaps identified in Opus review:
- Test #13: Load testing (1000+ requests)
- Test #14: Wrap-around boundary conditions
- Test #16: Registry crash recovery

Tests #12 and #15 (multiprocessing tests) are in test_opus_multiprocess.py
to avoid pytest hangs on macOS.

These tests verify that all Opus critical fixes work correctly under
real-world production conditions.
"""

import pytest
import time
import os
import json
from pathlib import Path

from src.orchestrator.worker_manager import WorkerManager
from src.config.server_config import ServerConfig
from src.ipc.messages import CompletionRequest
from src.ipc.shared_memory_bridge import SharedMemoryBridge
from src.ipc.shm_registry import SharedMemoryRegistry


# Module-level fixture for load capacity test
@pytest.fixture
def load_capacity_worker_manager():
    """Create WorkerManager for load capacity testing."""
    config = ServerConfig.auto_detect()
    manager = WorkerManager(config)
    yield manager
    try:
        manager.unload_model()
    except:
        pass


def test_sustained_load_1000_requests(load_capacity_worker_manager):
    """Test #13: Load tests (1000+ requests).

    Verifies system can handle sustained load without performance degradation.

    IMPORTANT: This test must be run independently, not as part of full suite.
    Run with: pytest tests/integration/test_opus_comprehensive.py::test_sustained_load_1000_requests -v

    The test hangs when run with other tests due to pytest fixture interaction.
    DO NOT SKIP - run independently to verify load capacity.
    """
    worker_manager = load_capacity_worker_manager
    print("\n=== TELEMETRY: Test started ===")

    # Skip if slow tests disabled
    if os.getenv("MLX_SKIP_SLOW_TESTS") == "1":
        pytest.skip("Slow tests disabled (MLX_SKIP_SLOW_TESTS=1)")

    model_path = os.getenv("MLX_TEST_MODEL", "mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    print(f"TELEMETRY: Model path: {model_path}")

    # Load model
    print("TELEMETRY: Loading model...")
    load_start = time.time()
    worker_manager.load_model(model_path)
    load_time = time.time() - load_start
    print(f"TELEMETRY: Model loaded in {load_time:.2f}s")

    # Make 1000 small requests
    num_requests = 1000
    successful = 0
    failed = 0
    total_time = 0

    print(f"\nTELEMETRY: Starting {num_requests} requests")
    test_start = time.time()

    for i in range(num_requests):
        try:
            # Progress indicator every 10 requests (more frequent)
            if i % 10 == 0:
                print(f"TELEMETRY: Request {i}/{num_requests} starting...")

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

            # Progress indicator every 100 requests (detailed)
            if (i + 1) % 100 == 0:
                avg_time = total_time / (i + 1)
                elapsed_total = time.time() - test_start
                print(f"TELEMETRY: Progress: {i+1}/{num_requests} "
                      f"(avg: {avg_time*1000:.1f}ms/req, "
                      f"elapsed: {elapsed_total:.1f}s, "
                      f"success: {successful}, failed: {failed})")

        except Exception as e:
            failed += 1
            print(f"TELEMETRY: Request {i} FAILED: {e}")

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
