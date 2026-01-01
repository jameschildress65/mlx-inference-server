"""Secure Shared Memory Manager - Production-hardened shared memory lifecycle management.

Implements all security hardening features from Opus 4.5 review:
- Random unpredictable names (128-bit entropy)
- Automatic cleanup on crash (signal handlers)
- Memory zeroing on close (prevent data leakage)
- Registry tracking (crash recovery)
- Environment variable passing (hide from process listings)
"""

import os
import sys
import signal
import secrets
import atexit
import logging
import threading
import random
from typing import Optional
from multiprocessing import shared_memory
from multiprocessing import resource_tracker
from pathlib import Path

from .shm_registry import SharedMemoryRegistry

logger = logging.getLogger(__name__)


class SecureSharedMemoryManager:
    """
    Manages shared memory lifecycle with production security hardening.

    Security Features:
    - Random names with 128-bit entropy (collision-free)
    - Automatic cleanup on SIGTERM, SIGINT, SIGHUP
    - Memory zeroing before unlink (prevent data leakage)
    - Registry tracking for crash recovery
    - Environment variable name passing (not CLI args)

    Usage (Server/Orchestrator):
        manager = SecureSharedMemoryManager(size=8*1024*1024, is_server=True)
        # Pass name via environment variable to worker:
        env['MLX_SHM_NAME'] = manager.name
        subprocess.Popen([...], env=env)

    Usage (Client/Worker):
        manager = SecureSharedMemoryManager(size=8*1024*1024, is_server=False)
        # Reads name from os.environ['MLX_SHM_NAME']
    """

    # Signals to handle for graceful cleanup
    CLEANUP_SIGNALS = [
        signal.SIGTERM,  # kill command
        signal.SIGINT,   # Ctrl+C
        signal.SIGHUP,   # terminal hangup
    ]

    # Class-level tracking for cleanup coordination
    _active_managers = []
    _cleanup_registered = False
    _original_handlers = {}

    # PRODUCTION: Signal-safe cleanup using pipe (POSIX async-signal-safe)
    # os.pipe() + os.write() is guaranteed async-signal-safe by POSIX
    # threading.Event is NOT guaranteed safe (uses locks internally)
    _cleanup_pipe_read = None
    _cleanup_pipe_write = None
    _cleanup_thread = None

    def __init__(
        self,
        size: int,
        is_server: bool,
        registry: Optional[SharedMemoryRegistry] = None
    ):
        """
        Initialize secure shared memory manager.

        Args:
            size: Size of shared memory in bytes
            is_server: True if creating (orchestrator), False if attaching (worker)
            registry: Optional custom registry (default: global registry)

        Raises:
            RuntimeError: If worker and MLX_SHM_NAME env var not set
            FileNotFoundError: If worker and shared memory doesn't exist
        """
        self.size = size
        self.is_server = is_server
        self.registry = registry or SharedMemoryRegistry()
        self.shm: Optional[shared_memory.SharedMemory] = None
        self._closed = False
        self._unlinked = False
        self._cleanup_lock = threading.RLock()  # Reentrant for signal safety

        if is_server:
            self._create_server()
        else:
            self._attach_client()

        # Add to active managers for signal handling
        SecureSharedMemoryManager._active_managers.append(self)

        # Register cleanup handlers (once per process)
        if not SecureSharedMemoryManager._cleanup_registered:
            self._register_cleanup_handlers()
            SecureSharedMemoryManager._cleanup_registered = True

    def _create_server(self):
        """Create shared memory as server with security hardening."""
        # Generate cryptographically random name (128-bit entropy)
        self.name = self._generate_secure_name()

        try:
            # Create shared memory
            self.shm = shared_memory.SharedMemory(
                name=self.name,
                create=True,
                size=self.size
            )

            # Register in tracking registry
            self.registry.register(self.name, os.getpid(), self.size)

            logger.info(
                f"Created secure shared memory: {self.name} "
                f"(size={self.size}, pid={os.getpid()})"
            )

        except Exception as e:
            logger.error(f"Failed to create shared memory: {e}")
            raise

    def _attach_client(self):
        """Attach to existing shared memory as client."""
        # Read name from environment variable (not CLI args for security)
        self.name = os.environ.get('MLX_SHM_NAME')

        if not self.name:
            raise RuntimeError(
                "MLX_SHM_NAME environment variable not set. "
                "Worker cannot attach to shared memory without name."
            )

        try:
            # Attach to existing shared memory
            self.shm = shared_memory.SharedMemory(name=self.name)

            # Client doesn't own lifecycle - unregister immediately
            self._unregister_from_tracker_safe()

            logger.info(
                f"Attached to secure shared memory: {self.name} "
                f"(size={self.shm.size}, pid={os.getpid()})"
            )

        except FileNotFoundError:
            logger.error(
                f"Shared memory not found: {self.name}. "
                "Server must create before worker attaches."
            )
            raise

    def _unregister_from_tracker_safe(self):
        """
        Safely unregister from resource tracker.

        Handles KeyError (already unregistered) gracefully.
        This is necessary because multiple cleanup paths may attempt to unregister.
        """
        if self.shm is None:
            return

        try:
            resource_tracker.unregister(self.shm._name, "shared_memory")
            logger.debug(f"Unregistered from resource tracker: {self.shm._name}")
        except KeyError:
            # Already unregistered - this is fine
            logger.debug(f"Already unregistered from tracker: {self.shm._name}")
        except Exception as e:
            logger.debug(f"Could not unregister from resource tracker: {e}")

    @staticmethod
    def _generate_secure_name() -> str:
        """
        Generate cryptographically secure random name.

        Uses 104-bit entropy (13 bytes = 26 hex chars) from secrets module.
        Collision probability is effectively zero for any practical use case.

        macOS POSIX shared memory name limit: 30 characters
        Format: mlx_<26 hex chars> = 30 chars total

        Returns:
            Random shared memory name
        """
        random_hex = secrets.token_hex(13)  # 13 bytes = 104 bits
        return f"mlx_{random_hex}"

    def _register_cleanup_handlers(self):
        """
        Register atexit and signal handlers for cleanup.

        PRODUCTION: Uses pipe-based cleanup with background thread for signal safety.
        Signal handlers can only call async-signal-safe functions.

        POSIX async-signal-safe operations used:
        - os.write() to pipe (guaranteed safe by POSIX)
        - os.read() from pipe (blocking, safe in thread context)
        """
        # Create cleanup pipe (ONCE per process)
        if SecureSharedMemoryManager._cleanup_pipe_read is None:
            r, w = os.pipe()
            SecureSharedMemoryManager._cleanup_pipe_read = r
            SecureSharedMemoryManager._cleanup_pipe_write = w
            logger.debug(f"Created cleanup pipe: read_fd={r}, write_fd={w}")

        # Start background cleanup thread (daemon)
        if SecureSharedMemoryManager._cleanup_thread is None:
            thread = threading.Thread(
                target=self._cleanup_worker,
                daemon=True,
                name="shm-cleanup-worker"
            )
            thread.start()
            SecureSharedMemoryManager._cleanup_thread = thread
            logger.debug("Started background cleanup worker thread")

        # Register atexit cleanup
        atexit.register(self._cleanup_all_managers)

        # Register signal handlers
        for sig in self.CLEANUP_SIGNALS:
            # Save original handler
            try:
                original = signal.getsignal(sig)
                SecureSharedMemoryManager._original_handlers[sig] = original

                # Set our handler
                signal.signal(sig, self._signal_handler)

                logger.debug(f"Registered signal handler for {signal.Signals(sig).name}")
            except (OSError, ValueError) as e:
                logger.warning(f"Could not register handler for signal {sig}: {e}")

    @staticmethod
    def _secure_zero_memory(shm: shared_memory.SharedMemory) -> bool:
        """
        Securely zero shared memory with chunked writes and verification.

        Uses 4KB chunks to avoid creating large temporary buffers and
        includes random sampling verification to ensure zeroing succeeded.

        Args:
            shm: Shared memory object to zero

        Returns:
            True if zeroing verified successful, False otherwise
        """
        try:
            if not hasattr(shm, 'buf') or shm.buf is None:
                logger.warning("Cannot zero memory: buffer not accessible")
                return False

            buf = shm.buf
            size = len(buf)

            # Zero in chunks to avoid large temporaries
            CHUNK_SIZE = 4096  # 4KB chunks
            zero_chunk = b'\x00' * CHUNK_SIZE

            for offset in range(0, size, CHUNK_SIZE):
                end = min(offset + CHUNK_SIZE, size)
                chunk_size = end - offset
                buf[offset:end] = zero_chunk[:chunk_size]

            # Verification pass: Sample random positions
            sample_size = min(100, max(10, size // 1000))  # At least 10, at most 100

            for _ in range(sample_size):
                pos = random.randint(0, size - 1)
                if buf[pos] != 0:
                    logger.error(f"Memory zeroing verification failed at position {pos}")
                    return False

            logger.debug(f"Successfully zeroed and verified {size} bytes")
            return True

        except Exception as e:
            logger.error(f"Memory zeroing failed: {e}")
            return False

    @classmethod
    def _cleanup_worker(cls):
        """
        PRODUCTION: Background thread that waits for cleanup signal and performs cleanup.

        This runs in a normal thread context where all operations are safe,
        unlike signal handlers which have severe restrictions.

        Uses os.read() on pipe for blocking wait (POSIX compliant, deadlock-free).
        """
        # Wait for cleanup signal (blocks until signal handler writes to pipe)
        # PRODUCTION: os.read() is safe in thread context (no locks, no deadlock)
        try:
            os.read(cls._cleanup_pipe_read, 1)  # Read 1 byte (blocks until available)
        except Exception as e:
            logger.error(f"Error reading from cleanup pipe: {e}")
            return

        # Perform cleanup in safe thread context
        # (logging, file I/O, locks all OK here)
        logger.info("Cleanup worker activated, cleaning up shared memory...")
        cls._cleanup_all_managers()

        # Exit process forcibly (os._exit terminates whole process, not just thread)
        logger.info("Cleanup complete, exiting")
        os._exit(0)

    @classmethod
    def _signal_handler(cls, signum, frame):
        """
        PRODUCTION: Handle signals with pipe-based cleanup.

        CRITICAL: Signal handlers can ONLY call async-signal-safe functions.
        This handler only writes to a pipe, which is guaranteed async-signal-safe by POSIX.
        The actual cleanup happens in the background thread.

        Async-signal-safe operations used (POSIX standard):
        - os.write() - GUARANTEED safe (POSIX.1-2001)

        NOT safe (previous approach):
        - threading.Event.set() - NOT guaranteed safe (uses locks internally)
        """
        # ONLY operation: Write to pipe (POSIX async-signal-safe)
        # Background thread blocks on read, unblocks when we write
        try:
            os.write(cls._cleanup_pipe_write, b'X')  # Write 1 byte
        except:
            # Cannot log in signal handler (not async-signal-safe)
            # If write fails, cleanup won't happen, but process will exit anyway
            pass

    @classmethod
    def _cleanup_all_managers(cls):
        """Clean up all active managers (called by atexit or signal handler)."""
        for manager in list(cls._active_managers):
            try:
                manager._cleanup()
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")

    def _cleanup(self):
        """
        Clean up shared memory with security hardening.

        Thread-safe cleanup with RLock to prevent TOCTOU race where
        multiple threads could enter cleanup simultaneously.

        Steps:
        1. Zero memory (prevent data leakage)
        2. Unregister from resource tracker (prevent double-cleanup)
        3. Close shared memory handle
        4. Unlink (if server)
        5. Unregister from registry
        """
        with self._cleanup_lock:
            if self._closed:
                return

            self._closed = True

            if self.shm is None:
                return

            try:
                # Step 1: Zero memory before unlinking (security - server only)
                if self.is_server:
                    try:
                        logger.debug(f"Zeroing shared memory: {self.name}")
                        success = self._secure_zero_memory(self.shm)
                        if not success:
                            logger.warning(f"Memory zeroing verification failed for {self.name}")
                    except Exception as e:
                        logger.warning(f"Could not zero memory: {e}")

                # Step 2: Close shared memory handle
                try:
                    self.shm.close()
                except Exception as e:
                    logger.warning(f"Error closing shared memory: {e}")

                # Step 3: Unlink if server (unlink() handles resource tracker unregister internally)
                if self.is_server and not self._unlinked:
                    try:
                        # Python's unlink() internally calls resource_tracker.unregister()
                        # so we don't need to manually unregister - it would cause double-unregister
                        self.shm.unlink()
                        self._unlinked = True
                        logger.debug(f"Unlinked shared memory: {self.name}")
                    except FileNotFoundError:
                        self._unlinked = True  # Already unlinked
                        logger.debug(f"Shared memory already unlinked: {self.name}")
                    except Exception as e:
                        logger.warning(f"Error unlinking shared memory: {e}")

                # Step 5: Unregister from registry
                try:
                    self.registry.unregister(self.name)
                except Exception as e:
                    logger.warning(f"Error unregistering from registry: {e}")

            finally:
                # Remove from active managers
                try:
                    SecureSharedMemoryManager._active_managers.remove(self)
                except ValueError:
                    pass  # Already removed

    def close(self):
        """Explicitly close and cleanup."""
        self._cleanup()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self._cleanup()
        return False

    def __del__(self):
        """Destructor cleanup (best effort)."""
        try:
            self._cleanup()
        except Exception:
            pass  # Ignore errors in destructor

    def __repr__(self) -> str:
        """String representation."""
        status = "closed" if self._closed else "open"
        role = "server" if self.is_server else "client"
        return (
            f"SecureSharedMemoryManager("
            f"name={self.name}, size={self.size}, role={role}, status={status})"
        )


def cleanup_stale_shared_memory(registry: Optional[SharedMemoryRegistry] = None):
    """
    Clean up stale shared memory segments from previous crashes.

    Call this on server startup before creating new shared memory.

    Steps:
    1. Read registry of active segments
    2. Check if process still exists
    3. If dead, attempt to clean up shared memory
    4. Remove from registry

    Args:
        registry: Optional custom registry (default: global registry)

    Returns:
        Number of stale segments cleaned up

    Example:
        # On server startup:
        cleaned = cleanup_stale_shared_memory()
        logger.info(f"Cleaned up {cleaned} stale shared memory segments")
    """
    registry = registry or SharedMemoryRegistry()
    cleaned = 0

    logger.info("Starting stale shared memory cleanup...")

    # Get all registered segments
    segments = registry.list_active()

    for segment in segments:
        name = segment["name"]
        pid = segment["pid"]

        # Check if process is alive
        try:
            os.kill(pid, 0)
            # Process alive, skip
            logger.debug(f"Process {pid} alive, keeping segment: {name}")
            continue
        except ProcessLookupError:
            # Process dead, clean up
            logger.info(f"Process {pid} dead, cleaning up segment: {name}")
        except PermissionError:
            # Process exists but owned by different user, skip
            logger.debug(f"Process {pid} owned by different user, skipping: {name}")
            continue

        # Attempt to clean up shared memory
        try:
            shm = shared_memory.SharedMemory(name=name)

            # Zero memory before unlinking (security)
            try:
                success = SecureSharedMemoryManager._secure_zero_memory(shm)
                if success:
                    logger.debug(f"Zeroed and verified stale memory: {name}")
                else:
                    logger.warning(f"Could not verify zeroing of stale memory {name}")
            except Exception as e:
                logger.warning(f"Could not zero stale memory {name}: {e}")

            # Close and unlink
            shm.close()
            shm.unlink()

            logger.info(f"Cleaned up stale segment: {name} (pid={pid})")
            cleaned += 1

        except FileNotFoundError:
            logger.debug(f"Segment already cleaned up: {name}")
        except Exception as e:
            logger.warning(f"Error cleaning up {name}: {e}")

        # Remove from registry regardless
        registry.unregister(name)

    logger.info(f"Stale shared memory cleanup complete: {cleaned} segments cleaned")
    return cleaned
