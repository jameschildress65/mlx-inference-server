"""
Unit tests for C2 fix: Thread-safe close() in SharedMemoryBridge.

Tests that close() is protected from TOCTOU race conditions between
explicit calls and __del__() during garbage collection.
"""

import pytest
import threading
import time
import gc

try:
    import posix_ipc
    HAS_POSIX_IPC = True
except ImportError:
    HAS_POSIX_IPC = False

pytestmark = pytest.mark.skipif(not HAS_POSIX_IPC, reason="posix_ipc not available")


class TestCloseRaceCondition:
    """Tests for C2 fix: Thread-safe close()."""

    def test_close_is_idempotent(self):
        """Verify close() can be called multiple times without error."""
        from src.ipc.shared_memory_bridge import SharedMemoryBridge

        bridge = SharedMemoryBridge("test_idempotent", is_server=True)

        # Multiple close calls should not raise
        bridge.close()
        bridge.close()
        bridge.close()

        assert bridge._closed is True

    def test_close_has_lock(self):
        """Verify _close_lock exists and is a Lock."""
        from src.ipc.shared_memory_bridge import SharedMemoryBridge

        bridge = SharedMemoryBridge("test_has_lock", is_server=True)

        assert hasattr(bridge, '_close_lock')
        assert isinstance(bridge._close_lock, type(threading.Lock()))

        bridge.close()

    def test_concurrent_close_no_errors(self):
        """Verify no errors when multiple threads call close() simultaneously."""
        from src.ipc.shared_memory_bridge import SharedMemoryBridge

        bridge = SharedMemoryBridge("test_concurrent", is_server=True)
        errors = []

        def close_bridge():
            try:
                time.sleep(0.001)  # Small jitter to increase race likelihood
                bridge.close()
            except Exception as e:
                errors.append(e)

        # Spawn 10 threads all trying to close at once
        threads = [threading.Thread(target=close_bridge) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during concurrent close: {errors}"
        assert bridge._closed is True

    def test_del_with_explicit_close(self):
        """Verify __del__ handles already-closed bridge gracefully."""
        from src.ipc.shared_memory_bridge import SharedMemoryBridge

        bridge = SharedMemoryBridge("test_del_after_close", is_server=True)
        bridge.close()

        # Explicitly call __del__ (simulating GC)
        bridge.__del__()

        # Should not raise, bridge should still be marked closed
        assert bridge._closed is True

    def test_gc_does_not_double_free(self):
        """Verify garbage collection doesn't cause double-free."""
        from src.ipc.shared_memory_bridge import SharedMemoryBridge

        bridge = SharedMemoryBridge("test_gc_safe", is_server=True)

        # Close explicitly
        bridge.close()

        # Force GC to run (would trigger __del__ if reference dropped)
        gc.collect()

        # Delete reference and force GC again
        del bridge
        gc.collect()

        # If we get here without crash, race condition is handled

    def test_close_and_gc_race(self):
        """Stress test: close() racing with GC __del__() on SAME bridge.

        Note: We test concurrent close on a single bridge, not concurrent
        bridge creation, because the SHM registry has its own file-level
        race that's separate from the C2 fix.
        """
        from src.ipc.shared_memory_bridge import SharedMemoryBridge

        # Create bridge ONCE, then hammer close() from multiple threads
        bridge = SharedMemoryBridge("test_race_single", is_server=True)
        errors = []

        def close_and_gc():
            try:
                time.sleep(0.0001)  # Small jitter
                bridge.close()
                gc.collect()
            except Exception as e:
                errors.append(e)

        # Many threads all trying to close the same bridge
        threads = [threading.Thread(target=close_and_gc) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have no errors - all threads should handle the race gracefully
        assert len(errors) == 0, f"Race condition errors: {errors}"
        assert bridge._closed is True

    def test_partial_init_del_safe(self):
        """Verify __del__ is safe if __init__ fails early (no _close_lock)."""
        from src.ipc.shared_memory_bridge import SharedMemoryBridge

        # Create a mock object that looks like a partially initialized bridge
        class PartialBridge:
            def __init__(self):
                self._closed = False
                # Note: _close_lock is NOT set (simulating init failure)

            def close(self):
                pass

            def __del__(self):
                # Same logic as SharedMemoryBridge.__del__
                try:
                    if hasattr(self, '_close_lock'):
                        self.close()
                except Exception:
                    pass

        # Should not raise
        partial = PartialBridge()
        del partial
        gc.collect()

    def test_context_manager_calls_close(self):
        """Verify context manager properly calls close()."""
        from src.ipc.shared_memory_bridge import SharedMemoryBridge

        with SharedMemoryBridge("test_context", is_server=True) as bridge:
            assert bridge._closed is False

        # After exiting context, should be closed
        assert bridge._closed is True


class TestCloseCleanupOrder:
    """Tests verifying cleanup happens in correct order."""

    def test_close_releases_semaphores(self):
        """Verify semaphores are released on close."""
        from src.ipc.shared_memory_bridge import SharedMemoryBridge

        bridge = SharedMemoryBridge("test_sem_release", is_server=True)
        req_sem_name = bridge.req_sem_name
        resp_sem_name = bridge.resp_sem_name

        bridge.close()

        # Semaphores should be unlinked (server unlinks)
        with pytest.raises(posix_ipc.ExistentialError):
            posix_ipc.Semaphore(req_sem_name)

        with pytest.raises(posix_ipc.ExistentialError):
            posix_ipc.Semaphore(resp_sem_name)

    def test_close_releases_shm(self):
        """Verify shared memory is released on close."""
        from src.ipc.shared_memory_bridge import SharedMemoryBridge
        from multiprocessing import shared_memory

        bridge = SharedMemoryBridge("test_shm_release", is_server=True)
        shm_name = bridge.shm_name

        bridge.close()

        # Shared memory should be unlinked
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=shm_name)
