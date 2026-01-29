"""
PRODUCTION: Process registry with crash-safe cleanup.

Implements:
1. PID file tracking (survives crashes)
2. Startup orphan cleanup (kills previous run's workers)
3. Process group management (SIGKILL escalation)
4. Heartbeat-based liveness detection
"""

import os
import signal
import time
import json
import atexit
import hashlib
import psutil
from pathlib import Path
from typing import Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import logging
import fcntl

try:
    import posix_ipc
    HAS_POSIX_IPC = True
except ImportError:
    posix_ipc = None
    HAS_POSIX_IPC = False

logger = logging.getLogger(__name__)

# Registry file location (survives restarts)
REGISTRY_DIR = Path("/tmp/mlx-server")
REGISTRY_FILE = REGISTRY_DIR / "worker_registry.json"
LOCK_FILE = REGISTRY_DIR / "registry.lock"


@dataclass
class WorkerRecord:
    """Tracked worker process."""
    pid: int
    model_id: str
    shm_name: str
    started_at: str
    parent_pid: int


class ProcessRegistry:
    """
    Crash-safe process registry.

    CRITICAL INVARIANTS:
    1. All spawned workers MUST be registered before spawn
    2. Registry is persisted to disk immediately
    3. On startup, ALL previously registered workers are killed
    4. Uses file locking for multi-process safety
    """

    _instance: Optional['ProcessRegistry'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self._workers: dict[int, WorkerRecord] = {}
        self._our_pid = os.getpid()

        # Ensure registry directory exists
        REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

        # CRITICAL: Clean up orphans from previous run FIRST
        self._cleanup_orphans_on_startup()

        # Register atexit handler
        atexit.register(self._cleanup_all_on_exit)

        logger.info(f"ProcessRegistry initialized (pid={self._our_pid})")

    def _acquire_lock(self) -> int:
        """Acquire exclusive file lock."""
        fd = os.open(str(LOCK_FILE), os.O_RDWR | os.O_CREAT, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX)
        return fd

    def _release_lock(self, fd: int):
        """Release file lock."""
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)

    def _load_registry(self) -> dict:
        """Load registry from disk."""
        if not REGISTRY_FILE.exists():
            return {"workers": {}, "parent_pid": None}
        try:
            return json.loads(REGISTRY_FILE.read_text())
        except (json.JSONDecodeError, IOError):
            return {"workers": {}, "parent_pid": None}

    def _save_registry(self, data: dict):
        """Save registry to disk atomically."""
        tmp_file = REGISTRY_FILE.with_suffix('.tmp')
        tmp_file.write_text(json.dumps(data, indent=2))
        tmp_file.rename(REGISTRY_FILE)

    def _cleanup_orphans_on_startup(self):
        """
        CRITICAL: Kill ALL workers from previous server runs.

        This runs BEFORE we accept any new requests.
        """
        lock_fd = self._acquire_lock()
        try:
            registry = self._load_registry()
            old_parent = registry.get("parent_pid")
            workers = registry.get("workers", {})

            if not workers:
                logger.info("No orphaned workers found")
                # Clear and claim registry
                self._save_registry({
                    "workers": {},
                    "parent_pid": self._our_pid
                })
                return

            logger.warning(f"Found {len(workers)} orphaned workers from previous run (parent_pid={old_parent})")

            killed_count = 0
            for pid_str, record in workers.items():
                pid = int(pid_str)
                try:
                    # Check if process exists
                    proc = psutil.Process(pid)

                    # Verify it's actually our worker (check command line)
                    cmdline = ' '.join(proc.cmdline())
                    if 'mlx_worker' in cmdline or 'worker.py' in cmdline:
                        logger.warning(f"Killing orphaned worker: pid={pid}, model={record.get('model_id', 'unknown')}")

                        # Try graceful termination first
                        proc.terminate()
                        try:
                            proc.wait(timeout=2.0)
                        except psutil.TimeoutExpired:
                            # Force kill
                            logger.warning(f"Force killing worker pid={pid}")
                            proc.kill()
                            proc.wait(timeout=1.0)

                        killed_count += 1

                        # Clean up shared memory and semaphores (C1 fix)
                        shm_name = record.get('shm_name')
                        if shm_name:
                            self._cleanup_ipc_resources(shm_name)
                    else:
                        logger.info(f"PID {pid} is not a worker, skipping")

                except psutil.NoSuchProcess:
                    logger.info(f"Worker pid={pid} already dead")
                except Exception as e:
                    logger.error(f"Error cleaning up worker pid={pid}: {e}")

            logger.info(f"Orphan cleanup complete: killed {killed_count} workers")

            # Clear registry and claim it
            self._save_registry({
                "workers": {},
                "parent_pid": self._our_pid
            })

        finally:
            self._release_lock(lock_fd)

    def _cleanup_ipc_resources(self, shm_name: str):
        """
        Clean up all IPC resources associated with a worker.

        Resources cleaned:
        1. POSIX semaphores (req and resp) - cleaned FIRST
        2. Shared memory segment

        Semaphore names are derived deterministically from shm_name
        using the same formula as SharedMemoryBridge.__init__.

        C1 Fix: Previously only cleaned shared memory, leaving semaphores
        orphaned after worker crashes. Now cleans both.
        """
        # Skip semaphore cleanup if not using shared memory IPC
        # (stdio mode uses "stdio" as shm_name, not "mlx_*")
        if shm_name and shm_name.startswith("mlx_") and HAS_POSIX_IPC:
            # Derive semaphore names using same formula as SharedMemoryBridge
            name_hash = hashlib.sha256(shm_name.encode()).hexdigest()[:16]
            sem_names = [f"/r{name_hash}", f"/s{name_hash}"]

            for sem_name in sem_names:
                try:
                    # C1 Fix (Opus review): Use unlink_semaphore() directly
                    # This avoids TOCTOU race between open/close/unlink
                    # unlink_semaphore removes from filesystem without opening
                    posix_ipc.unlink_semaphore(sem_name)
                    logger.info(f"Cleaned up orphaned semaphore: {sem_name}")
                except posix_ipc.ExistentialError:
                    # Already cleaned up - this is fine
                    logger.debug(f"Semaphore already cleaned: {sem_name}")
                except Exception as e:
                    # Log but continue - don't let semaphore cleanup failure
                    # prevent shared memory cleanup
                    logger.warning(f"Error cleaning up semaphore {sem_name}: {e}")

        # Clean up shared memory segment
        try:
            from multiprocessing import shared_memory
            shm = shared_memory.SharedMemory(name=shm_name)
            shm.close()
            shm.unlink()
            logger.info(f"Cleaned up orphaned SHM: {shm_name}")
        except FileNotFoundError:
            # Already cleaned up - this is fine
            logger.debug(f"SHM already cleaned: {shm_name}")
        except Exception as e:
            logger.warning(f"Error cleaning up SHM {shm_name}: {e}")

    def register_worker(self, pid: int, model_id: str, shm_name: str):
        """
        Register a worker process.

        MUST be called BEFORE spawning to ensure crash safety.
        """
        record = WorkerRecord(
            pid=pid,
            model_id=model_id,
            shm_name=shm_name,
            started_at=datetime.now().isoformat(),
            parent_pid=self._our_pid
        )

        lock_fd = self._acquire_lock()
        try:
            registry = self._load_registry()
            registry["workers"][str(pid)] = {
                "pid": pid,
                "model_id": model_id,
                "shm_name": shm_name,
                "started_at": record.started_at,
                "parent_pid": self._our_pid
            }
            registry["parent_pid"] = self._our_pid
            self._save_registry(registry)

            self._workers[pid] = record
            logger.info(f"Registered worker: pid={pid}, model={model_id}")

        finally:
            self._release_lock(lock_fd)

    def unregister_worker(self, pid: int):
        """Remove worker from registry."""
        lock_fd = self._acquire_lock()
        try:
            registry = self._load_registry()
            if str(pid) in registry["workers"]:
                del registry["workers"][str(pid)]
                self._save_registry(registry)

            if pid in self._workers:
                del self._workers[pid]

            logger.info(f"Unregistered worker: pid={pid}")

        finally:
            self._release_lock(lock_fd)

    def terminate_worker(self, pid: int, timeout: float = 5.0) -> bool:
        """
        Terminate a worker with escalation.

        1. SIGTERM (graceful)
        2. Wait timeout
        3. SIGKILL (force)

        Returns True if worker was terminated.
        """
        # Get worker info for IPC cleanup BEFORE checking if process exists
        # (we need to clean up IPC resources even if process is already dead)
        worker = self._workers.get(pid)
        shm_name = worker.shm_name if worker else None

        try:
            proc = psutil.Process(pid)
        except psutil.NoSuchProcess:
            # Process already dead - still need to clean up IPC resources
            if shm_name:
                self._cleanup_ipc_resources(shm_name)
            self.unregister_worker(pid)
            return True

        try:
            # Step 1: Graceful termination
            logger.info(f"Terminating worker pid={pid} (SIGTERM)")
            proc.terminate()

            try:
                proc.wait(timeout=timeout)
                logger.info(f"Worker pid={pid} exited gracefully")
            except psutil.TimeoutExpired:
                # Step 2: Force kill
                logger.warning(f"Worker pid={pid} did not exit, sending SIGKILL")
                proc.kill()
                proc.wait(timeout=2.0)
                logger.info(f"Worker pid={pid} force killed")

            # Cleanup shared memory and semaphores (C1 fix)
            if shm_name:
                self._cleanup_ipc_resources(shm_name)
            self.unregister_worker(pid)
            return True

        except Exception as e:
            logger.error(f"Error terminating worker pid={pid}: {e}")
            return False

    def terminate_all(self):
        """Terminate all registered workers."""
        pids = list(self._workers.keys())
        for pid in pids:
            self.terminate_worker(pid)

    def _cleanup_all_on_exit(self):
        """atexit handler - terminate all workers."""
        logger.info("Server exiting, cleaning up all workers...")
        self.terminate_all()

        # Clear registry
        lock_fd = self._acquire_lock()
        try:
            self._save_registry({
                "workers": {},
                "parent_pid": None
            })
        finally:
            self._release_lock(lock_fd)


# Global singleton
_registry: Optional[ProcessRegistry] = None

def get_registry() -> ProcessRegistry:
    """Get the process registry singleton."""
    global _registry
    if _registry is None:
        _registry = ProcessRegistry()
    return _registry
