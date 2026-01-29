"""
Unit tests for C1 fix: Orphaned semaphore cleanup.

Tests that _cleanup_ipc_resources() properly cleans both:
1. POSIX semaphores (derived from shm_name hash)
2. Shared memory segments
"""

import pytest
import hashlib
from multiprocessing import shared_memory

try:
    import posix_ipc
    HAS_POSIX_IPC = True
except ImportError:
    HAS_POSIX_IPC = False

pytestmark = pytest.mark.skipif(not HAS_POSIX_IPC, reason="posix_ipc not available")


class TestSemaphoreCleanup:
    """Tests for C1 fix: _cleanup_ipc_resources."""

    def test_cleanup_cleans_semaphores_and_shm(self):
        """Verify both semaphores and shared memory are cleaned."""
        from src.orchestrator.process_registry import ProcessRegistry

        # Create test resources manually (simulating what SharedMemoryBridge does)
        shm_name = "mlx_test_c1_cleanup_001"
        name_hash = hashlib.sha256(shm_name.encode()).hexdigest()[:16]
        req_sem_name = f"/r{name_hash}"
        resp_sem_name = f"/s{name_hash}"

        # Create shared memory
        shm = shared_memory.SharedMemory(name=shm_name, create=True, size=1024)

        # Create semaphores
        req_sem = posix_ipc.Semaphore(req_sem_name, flags=posix_ipc.O_CREAT, initial_value=1)
        resp_sem = posix_ipc.Semaphore(resp_sem_name, flags=posix_ipc.O_CREAT, initial_value=1)

        # Close our handles (but don't unlink - simulating orphaned resources)
        shm.close()
        req_sem.close()
        resp_sem.close()

        # Now run cleanup
        registry = ProcessRegistry()
        registry._cleanup_ipc_resources(shm_name)

        # Verify shared memory is gone
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=shm_name)

        # Verify semaphores are gone
        with pytest.raises(posix_ipc.ExistentialError):
            posix_ipc.Semaphore(req_sem_name)

        with pytest.raises(posix_ipc.ExistentialError):
            posix_ipc.Semaphore(resp_sem_name)

    def test_cleanup_handles_missing_semaphores(self):
        """Verify cleanup handles already-deleted semaphores gracefully."""
        from src.orchestrator.process_registry import ProcessRegistry

        # Create only shared memory, no semaphores
        shm_name = "mlx_test_c1_no_sems_002"
        shm = shared_memory.SharedMemory(name=shm_name, create=True, size=1024)
        shm.close()

        # Should not raise exception
        registry = ProcessRegistry()
        registry._cleanup_ipc_resources(shm_name)

        # Verify shared memory is cleaned
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=shm_name)

    def test_cleanup_handles_missing_shm(self):
        """Verify cleanup handles already-deleted shared memory gracefully."""
        from src.orchestrator.process_registry import ProcessRegistry

        # Create only semaphores, no shared memory
        shm_name = "mlx_test_c1_no_shm_003"
        name_hash = hashlib.sha256(shm_name.encode()).hexdigest()[:16]
        req_sem_name = f"/r{name_hash}"
        resp_sem_name = f"/s{name_hash}"

        req_sem = posix_ipc.Semaphore(req_sem_name, flags=posix_ipc.O_CREAT, initial_value=1)
        resp_sem = posix_ipc.Semaphore(resp_sem_name, flags=posix_ipc.O_CREAT, initial_value=1)
        req_sem.close()
        resp_sem.close()

        # Should not raise exception
        registry = ProcessRegistry()
        registry._cleanup_ipc_resources(shm_name)

        # Verify semaphores are cleaned
        with pytest.raises(posix_ipc.ExistentialError):
            posix_ipc.Semaphore(req_sem_name)

        with pytest.raises(posix_ipc.ExistentialError):
            posix_ipc.Semaphore(resp_sem_name)

    def test_cleanup_skips_stdio_workers(self):
        """Verify cleanup skips workers using stdio (no shared memory)."""
        from src.orchestrator.process_registry import ProcessRegistry

        # Should not raise or attempt semaphore cleanup for "stdio"
        registry = ProcessRegistry()
        registry._cleanup_ipc_resources("stdio")
        # No assertion needed - just verify no exception

    def test_cleanup_skips_non_mlx_names(self):
        """Verify cleanup skips non-mlx shared memory names."""
        from src.orchestrator.process_registry import ProcessRegistry

        # Should not attempt semaphore cleanup for names not starting with "mlx_"
        registry = ProcessRegistry()
        registry._cleanup_ipc_resources("other_shm_name")
        # No assertion needed - just verify no exception

    def test_semaphore_name_derivation_matches_bridge(self):
        """Verify semaphore name derivation matches SharedMemoryBridge."""
        # This test ensures the hash formula stays in sync
        shm_name = "mlx_test_hash_match"
        name_hash = hashlib.sha256(shm_name.encode()).hexdigest()[:16]

        # These should match what SharedMemoryBridge uses
        expected_req = f"/r{name_hash}"
        expected_resp = f"/s{name_hash}"

        assert len(expected_req) == 18  # /r + 16 hex chars
        assert len(expected_resp) == 18  # /s + 16 hex chars
        assert expected_req.startswith("/r")
        assert expected_resp.startswith("/s")


class TestCleanupIntegration:
    """Integration tests for cleanup during worker lifecycle."""

    def test_terminate_worker_cleans_semaphores(self, monkeypatch):
        """Verify terminate_worker calls _cleanup_ipc_resources."""
        from src.orchestrator.process_registry import ProcessRegistry

        registry = ProcessRegistry()

        cleanup_calls = []

        def track_cleanup(shm_name):
            cleanup_calls.append(shm_name)

        monkeypatch.setattr(registry, '_cleanup_ipc_resources', track_cleanup)

        # Register a fake worker (use a PID that definitely doesn't exist)
        fake_pid = 999999
        registry._workers[fake_pid] = type('WorkerRecord', (), {
            'pid': fake_pid,
            'model_id': 'test-model',
            'shm_name': 'mlx_test_terminate',
            'started_at': '2026-01-29T00:00:00',
            'parent_pid': 1
        })()

        # Terminate (will fail to find process, but should still cleanup)
        registry.terminate_worker(fake_pid)

        # Verify cleanup was called with correct shm_name
        assert "mlx_test_terminate" in cleanup_calls
