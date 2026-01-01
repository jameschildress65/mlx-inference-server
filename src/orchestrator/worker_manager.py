"""WorkerManager - Manages model worker subprocess lifecycle."""

import subprocess
import sys
import os
import re
import threading
import time
import logging
from typing import Optional, Dict, Any, Set
from pathlib import Path

from ..ipc.stdio_bridge import StdioBridge, WorkerCommunicationError
from ..ipc.shared_memory_bridge import (
    SharedMemoryBridge,
    SharedMemoryIPCError,
    WorkerCommunicationError as ShmemWorkerError
)
from ..ipc.messages import CompletionRequest, PingMessage, ShutdownMessage

logger = logging.getLogger(__name__)

# Security: Whitelist of trusted model organizations
ALLOWED_MODEL_ORGS: Set[str] = {
    'mlx-community',
    'meta-llama',
    'mistralai',
    'Qwen',
    'microsoft',
    'google',
}

# Security: Model path format validation pattern
ALLOWED_MODEL_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+/[a-zA-Z0-9._-]+$')


class WorkerError(Exception):
    """Base exception for worker-related errors."""
    pass


class WorkerSpawnError(WorkerError):
    """Failed to spawn worker subprocess."""
    pass


class WorkerTimeoutError(WorkerError):
    """Worker did not respond in time."""
    pass


class NoModelLoadedError(WorkerError):
    """Attempted operation with no active worker."""
    pass


class ModelLoadResult:
    """Result of model load operation."""
    def __init__(self, model_name: str, memory_gb: float, load_time: float):
        self.model_name = model_name
        self.memory_gb = memory_gb
        self.load_time = load_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "memory_gb": self.memory_gb,
            "load_time": self.load_time
        }


class UnloadResult:
    """Result of model unload operation."""
    def __init__(self, model_name: str, memory_freed_gb: float):
        self.model_name = model_name
        self.memory_freed_gb = memory_freed_gb

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": "success",
            "model_unloaded": self.model_name,
            "memory_freed_gb": self.memory_freed_gb
        }


class WorkerManager:
    """Manages model worker subprocess lifecycle."""

    def __init__(self, config):
        """
        Initialize WorkerManager.

        Args:
            config: ServerConfig instance
        """
        self.config = config
        self.active_worker: Optional[subprocess.Popen] = None
        self.active_model_name: Optional[str] = None
        self.active_memory_gb: float = 0.0
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)

        # Activity tracking for idle monitoring
        self.last_activity_time: float = time.time()
        self.active_requests: int = 0
        self.activity_lock = threading.Lock()

        # Worker ID counter for process naming
        self._worker_counter: int = 0

        # PHASE 2: Shared memory IPC (Opus 4.5 optimization)
        self._shmem_bridge: Optional[SharedMemoryBridge] = None
        self._use_shmem: bool = config.use_shared_memory  # Can fall back to stdio if needed

    # =========================================================================
    # Worker Abstraction Layer (Production-Ready for Future Multi-Worker)
    # =========================================================================
    # These methods provide a clean abstraction for worker access, enabling
    # future migration to multi-worker pool with minimal code changes.
    # Current implementation: Single worker
    # Future implementation: Worker pool with load balancing
    # =========================================================================

    def _get_worker_for_request(self, model_name: str) -> subprocess.Popen:
        """
        Get worker process for handling a request.

        This abstraction enables future multi-worker support without changing
        calling code. Currently returns single active worker, but future
        implementation can add load balancing across worker pool.

        CRITICAL FIX (Opus 4.5 Review):
        - Always acquires lock internally (prevents race conditions)
        - Bounded iterative retry (prevents stack overflow)
        - Thread-safe for concurrent async requests

        Args:
            model_name: Model being requested (for future routing logic)

        Returns:
            Worker subprocess handle

        Raises:
            NoModelLoadedError: If no worker is available
            WorkerError: If worker is dead/unresponsive after retries

        Thread-safe: Yes (always acquires lock internally)
        """
        # Opus Fix: Always acquire lock internally to prevent race conditions
        # Previous code assumed caller held lock, but async requests can interleave
        max_retries = 3  # Bounded retries to prevent stack overflow

        for attempt in range(max_retries):
            with self.lock:
                if self.active_worker is None:
                    raise NoModelLoadedError(
                        f"No worker available for model: {model_name}. "
                        f"Load model first via /v1/chat/completions or admin API."
                    )

                # Health check: Verify worker is alive
                if self.active_worker.poll() is None:
                    # Worker is healthy - return it
                    return self.active_worker

                # Worker is dead - cleanup and retry
                returncode = self.active_worker.returncode
                self.logger.warning(
                    f"Worker process died (returncode: {returncode}, attempt {attempt+1}/{max_retries}). "
                    f"Model: {self.active_model_name}"
                )
                self._cleanup_dead_worker()

                # If this was the last retry, raise error
                if attempt == max_retries - 1:
                    raise WorkerError(
                        f"Worker process died (exit code: {returncode}). "
                        f"Tried {max_retries} times. Reload model required."
                    )

        # Should never reach here, but satisfy type checker
        raise WorkerError("Failed to get healthy worker after retries")

        # Future multi-worker implementation will replace above with:
        # return self._get_next_healthy_worker_from_pool(model_name)

    # =========================================================================
    # IPC Abstraction Layer (PHASE 2: Shared Memory + Stdio Fallback)
    # =========================================================================

    def _send_message(self, message):
        """
        Send message via shared memory or stdio (with fallback).

        PHASE 2: Uses SharedMemoryBridge if available, falls back to stdio.
        """
        if self._shmem_bridge:
            try:
                SharedMemoryBridge.send_message_shmem(self._shmem_bridge, message)
            except (SharedMemoryIPCError, ShmemWorkerError) as e:
                self.logger.warning(f"Shared memory send failed, falling back to stdio: {e}")
                self._shmem_bridge = None  # Disable shared memory for this worker
                self._use_shmem = False
                StdioBridge.send_message(self.active_worker, message)
        else:
            StdioBridge.send_message(self.active_worker, message)

    def _receive_message(self, timeout=None):
        """
        Receive message via shared memory or stdio (with fallback).

        PHASE 2: Uses SharedMemoryBridge if available, falls back to stdio.
        """
        if self._shmem_bridge:
            try:
                return SharedMemoryBridge.receive_message_shmem(self._shmem_bridge, timeout)
            except (SharedMemoryIPCError, ShmemWorkerError) as e:
                self.logger.warning(f"Shared memory receive failed, falling back to stdio: {e}")
                self._shmem_bridge = None  # Disable shared memory for this worker
                self._use_shmem = False
                return StdioBridge.receive_message(self.active_worker, timeout)
        else:
            return StdioBridge.receive_message(self.active_worker, timeout)

    # =========================================================================
    # End IPC Abstraction Layer
    # =========================================================================

    def _validate_model_path(self, model_path: str) -> None:
        """
        Validate model path against injection attacks.

        Security checks:
        1. Format validation (org/model pattern)
        2. Organization whitelist
        3. Path traversal prevention

        Args:
            model_path: HuggingFace model path (e.g., "mlx-community/Qwen2.5-7B-Instruct-4bit")

        Raises:
            ValueError: If model path fails validation
        """
        # 1. Check format
        if not ALLOWED_MODEL_PATTERN.match(model_path):
            raise ValueError(
                f"Invalid model path format: {model_path}. "
                f"Expected format: organization/model-name"
            )

        # 2. Whitelist organization
        org = model_path.split('/')[0]
        if org not in ALLOWED_MODEL_ORGS:
            raise ValueError(
                f"Untrusted organization: {org}. "
                f"Allowed organizations: {', '.join(sorted(ALLOWED_MODEL_ORGS))}"
            )

        # 3. Prevent path traversal
        if '..' in model_path or model_path.startswith('/'):
            raise ValueError(f"Path traversal attempt detected: {model_path}")

        self.logger.debug(f"Model path validation passed: {model_path}")

    def load_model(self, model_path: str, timeout: int = 120) -> ModelLoadResult:
        """
        Spawn new worker subprocess for model.

        Steps:
        1. Kill existing worker if any (unload_model)
        2. Spawn new subprocess with model_path
        3. Wait for ready signal via IPC
        4. Return load stats

        Args:
            model_path: HuggingFace model path (e.g., "mlx-community/Qwen2.5-7B-Instruct-4bit")
            timeout: Max seconds to wait for ready signal

        Returns:
            ModelLoadResult with load_time, memory_gb, model_name

        Raises:
            WorkerSpawnError: If worker fails to start
            WorkerTimeoutError: If ready signal not received
        """
        with self.lock:
            start_time = time.time()

            # Security: Validate model path before spawning worker
            self._validate_model_path(model_path)

            # Kill existing worker if any
            if self.active_worker is not None:
                self.logger.info(f"Unloading existing model: {self.active_model_name}")
                self._unload_model_internal()

            # PHASE 2: Create shared memory BEFORE spawning worker (proper POSIX pattern)
            # This prevents race condition where worker tries to attach before orchestrator creates it
            self._worker_counter += 1
            worker_id = self._worker_counter
            shm_name = None

            if self._use_shmem:
                try:
                    # Create shared memory with SecureSharedMemoryManager (random name)
                    # Note: name parameter is legacy/unused, kept for compatibility
                    self._shmem_bridge = SharedMemoryBridge(
                        name=f"worker_{worker_id}",  # Legacy, unused
                        is_server=True
                    )
                    # Get the actual random shared memory name for env var passing
                    shm_name = self._shmem_bridge.shm_name
                    self.logger.info(f"Shared memory created: {shm_name}")
                except Exception as e:
                    self.logger.warning(f"Shared memory creation failed, falling back to stdio: {e}")
                    self._shmem_bridge = None
                    self._use_shmem = False
                    shm_name = None

            # Spawn worker subprocess
            self.logger.info(f"Spawning worker #{worker_id} for model: {model_path}")
            try:
                # Phase 3: Detect model capabilities and route to appropriate venv
                from src.worker.model_loader import ModelLoader
                capabilities = ModelLoader.detect_model_capabilities(model_path)

                # Choose Python executable based on model type
                if capabilities["vision"]:
                    # Vision models use venv-vision with mlx-vlm
                    python_exe = os.path.join(os.getcwd(), "venv-vision", "bin", "python")

                    # Validate venv-vision exists (Opus security recommendation)
                    if not os.path.exists(python_exe):
                        raise EnvironmentError(
                            f"Vision environment not found: {python_exe}\n"
                            f"Vision models require venv-vision with mlx-vlm.\n"
                            f"Install with:\n"
                            f"  python3 -m venv venv-vision\n"
                            f"  venv-vision/bin/pip install mlx-vlm pillow transformers>=4.44.0,<5.0\n"
                            f"  venv-vision/bin/pip install setproctitle pyyaml posix-ipc"
                        )

                    self.logger.info(f"Using vision venv for vision model: {model_path}")
                else:
                    # Text models use main venv with mlx-lm
                    python_exe = sys.executable
                    self.logger.info(f"Using text venv for text model: {model_path}")

                worker_module = "src.worker"  # python -m src.worker
                worker_args = [python_exe, "-m", worker_module, model_path, str(worker_id)]

                # Prepare environment with shared memory name (if using shmem)
                worker_env = os.environ.copy()
                if shm_name:
                    # PHASE 2 Security: Pass shared memory name via environment variable
                    # (not CLI args, to prevent exposure in process listings)
                    worker_env['MLX_SHM_NAME'] = shm_name

                self.active_worker = subprocess.Popen(
                    worker_args,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    env=worker_env,  # Pass environment with MLX_SHM_NAME
                    cwd=os.getcwd()  # Ensure correct working directory
                )
                self.logger.debug(f"Worker spawned with PID: {self.active_worker.pid}")
            except Exception as e:
                # Clean up shared memory if worker spawn failed
                if self._shmem_bridge:
                    try:
                        self._shmem_bridge.close()
                    except Exception:
                        pass
                    self._shmem_bridge = None
                raise WorkerSpawnError(f"Failed to spawn worker: {e}")

            # Wait for ready signal
            self.logger.info(f"Waiting for worker ready signal (timeout: {timeout}s)")
            try:
                ready_msg = self._receive_message(timeout=timeout)

                if ready_msg.type == "error":
                    self.logger.error(f"Worker failed to load model: {ready_msg.message}")
                    self._kill_worker()
                    raise WorkerSpawnError(f"Worker error: {ready_msg.message}")

                if ready_msg.type != "ready":
                    self.logger.error(f"Unexpected message type: {ready_msg.type}")
                    self._kill_worker()
                    raise WorkerSpawnError(f"Expected 'ready', got '{ready_msg.type}'")

                # Success - store worker info
                self.active_model_name = ready_msg.model_name
                self.active_memory_gb = ready_msg.memory_gb
                load_time = time.time() - start_time

                self.logger.info(
                    f"Worker ready: {self.active_model_name} "
                    f"({self.active_memory_gb:.2f} GB in {load_time:.1f}s)"
                )

                # Update activity timestamp - loading is an activity
                self._update_activity()

                return ModelLoadResult(
                    model_name=self.active_model_name,
                    memory_gb=self.active_memory_gb,
                    load_time=load_time
                )

            except WorkerCommunicationError as e:
                self.logger.error(f"Worker communication failed: {e}")
                self._kill_worker()
                raise WorkerSpawnError(f"Worker died during startup: {e}")

    def unload_model(self) -> UnloadResult:
        """
        Terminate worker subprocess - guarantees memory cleanup.

        Steps:
        1. Send shutdown signal (graceful)
        2. Wait up to 5 seconds
        3. Send SIGKILL if still alive
        4. Verify process dead
        5. Return memory freed

        Returns:
            UnloadResult with model_name and memory_freed_gb

        Raises:
            NoModelLoadedError: If no worker active
        """
        with self.lock:
            return self._unload_model_internal()

    def unload_model_if_idle(self) -> Optional[UnloadResult]:
        """
        Atomically check if worker is idle and unload if so.

        This method prevents TOCTOU race conditions by performing the idle check
        and unload operation under the same lock that protects generate().

        Race condition scenario (without this method):
        1. IdleMonitor: has_active_requests() returns False
        2. HTTP Thread: generate() starts, increments active_requests
        3. IdleMonitor: unload_model() kills worker
        4. HTTP Thread: send_message() fails - worker dead

        This atomic method prevents this by holding the lock throughout.

        M6: Fixed lock ordering to prevent deadlock - always acquire
        activity_lock first, then self.lock (consistent with other methods).

        Returns:
            UnloadResult if unloaded, None if not idle or no worker

        Raises:
            None (silently returns None if conditions not met)
        """
        # M6: Acquire locks in consistent order: activity_lock first, then main lock
        with self.activity_lock:
            if self.active_requests > 0:
                # Not idle - active generation in progress
                return None

            # Now safe to acquire main lock (no deadlock risk)
            with self.lock:
                if self.active_worker is None:
                    return None

                # Safe to unload - no race possible since we hold both locks
                return self._unload_model_internal()

    def _unload_model_internal(self) -> UnloadResult:
        """Internal unload (assumes lock held)."""
        if self.active_worker is None:
            raise NoModelLoadedError("No model currently loaded")

        model_name = self.active_model_name
        memory_gb = self.active_memory_gb

        self.logger.info(f"Terminating worker for model: {model_name}")

        # Send graceful shutdown signal
        try:
            self._send_message(ShutdownMessage())
            self.logger.debug("Sent shutdown signal to worker")
        except (WorkerCommunicationError, ShmemWorkerError, SharedMemoryIPCError):
            self.logger.warning("Could not send shutdown signal (worker may be dead)")

        # Wait for graceful exit (up to 5 seconds)
        try:
            self.active_worker.wait(timeout=5)
            self.logger.info(f"Worker exited gracefully (returncode: {self.active_worker.returncode})")
        except subprocess.TimeoutExpired:
            self.logger.warning("Worker did not exit gracefully, sending SIGKILL")
            self._kill_worker()

        # Clear state
        self.active_worker = None
        self.active_model_name = None
        self.active_memory_gb = 0.0

        self.logger.info(f"Worker terminated, memory freed: {memory_gb:.2f} GB")

        return UnloadResult(model_name=model_name, memory_freed_gb=memory_gb)

    def generate(self, request: CompletionRequest) -> Dict[str, Any]:
        """
        Forward completion request to worker via IPC.

        Args:
            request: CompletionRequest message

        Returns:
            Response dict with 'text', 'tokens', 'finish_reason'

        Raises:
            NoModelLoadedError: If no worker active
            WorkerCommunicationError: If worker communication fails
        """
        # DEADLOCK FIX: Track activity BEFORE getting worker to enforce consistent lock ordering
        # Lock order: activity_lock → self.lock (matches unload_model_if_idle)
        # Previous order (worker first, then increment) caused ABBA deadlock
        self._increment_active_requests()
        self._update_activity()

        # Get worker through abstraction (enables future multi-worker)
        worker = self._get_worker_for_request(request.model)

        try:
            # Send request
            self._send_message(request)

            # Receive response (longer timeout for generation - can take time with large models)
            # 300s timeout for 72B+ models generating long responses
            response = self._receive_message(timeout=300)

            if response.type == "error":
                raise WorkerError(f"Worker error: {response.message}")

            if response.type != "completion_response":
                raise WorkerError(f"Expected 'completion_response', got '{response.type}'")

            return {
                "text": response.text,
                "tokens": response.tokens,
                "finish_reason": response.finish_reason
            }
        except BrokenPipeError as e:
            # Opus 4.5 High Priority Fix H4: Handle race condition where worker dies
            # between health check and message send (TOCTOU race)
            self.logger.error(f"Worker died during generation: {e}")
            with self.lock:
                self._cleanup_dead_worker()
            raise WorkerError("Worker process died during generation. Reload model required.")
        finally:
            # Always decrement active requests (even on error)
            self._decrement_active_requests()

    def generate_stream(self, request: CompletionRequest):
        """
        Forward streaming completion request to worker via IPC.

        Yields chunks as they arrive from worker.

        Args:
            request: CompletionRequest message with stream=True

        Yields:
            Message objects (stream_chunk type) from worker

        Raises:
            NoModelLoadedError: If no worker active
            WorkerCommunicationError: If worker communication fails
        """
        # DEADLOCK FIX: Track activity BEFORE getting worker to enforce consistent lock ordering
        # Lock order: activity_lock → self.lock (matches unload_model_if_idle)
        # Previous order (worker first, then increment) caused ABBA deadlock
        self._increment_active_requests()
        self._update_activity()

        # Get worker through abstraction (enables future multi-worker)
        worker = self._get_worker_for_request(request.model)

        try:
            # Send request
            self._send_message(request)

            # Receive chunks as worker generates them
            # Use adaptive timeout: longer for first chunk (model init), shorter for subsequent
            first_chunk = True
            while True:
                # First chunk: 300s timeout (model may need to initialize)
                # Subsequent chunks: 30s timeout (should be fast once streaming)
                timeout = 300 if first_chunk else 30
                chunk = self._receive_message(timeout=timeout)

                if chunk.type == "error":
                    raise WorkerError(f"Worker error: {chunk.message}")

                if chunk.type == "stream_chunk":
                    yield chunk
                    first_chunk = False  # After first chunk, use shorter timeout
                    if chunk.done:
                        break
                else:
                    raise WorkerError(f"Expected 'stream_chunk', got '{chunk.type}'")
        except BrokenPipeError as e:
            # Opus 4.5 High Priority Fix H4: Handle race condition where worker dies
            # between health check and message send (TOCTOU race)
            self.logger.error(f"Worker died during streaming generation: {e}")
            with self.lock:
                self._cleanup_dead_worker()
            raise WorkerError("Worker process died during streaming. Reload model required.")
        finally:
            # Always decrement active requests (even on error)
            self._decrement_active_requests()

    def health_check(self) -> Dict[str, Any]:
        """
        Check worker health via ping.

        Returns:
            Dict with 'healthy' (bool) and 'status' (str)
        """
        try:
            # Get worker through abstraction (includes alive check)
            # Pass empty model name since health check doesn't care about model
            worker = self._get_worker_for_request("")
        except NoModelLoadedError:
            return {"healthy": False, "status": "no_worker"}
        except WorkerError:
            return {"healthy": False, "status": "dead"}

        # Send ping
        try:
            self._send_message(PingMessage())
            pong = self._receive_message(timeout=2.0)

            if pong.type == "pong":
                return {"healthy": True, "status": "healthy"}
            else:
                return {"healthy": False, "status": "unexpected_response"}

        except (WorkerCommunicationError, ShmemWorkerError, SharedMemoryIPCError) as e:
            self.logger.error(f"Health check failed: {e}")
            self._cleanup_dead_worker()
            return {"healthy": False, "status": "communication_error"}

    def _kill_worker(self) -> None:
        """Force-kill worker process (SIGKILL)."""
        if self.active_worker is not None:
            self.logger.info("Force-killing worker process")
            self.active_worker.kill()
            self.active_worker.wait()

            # PHASE 2: Clean up shared memory
            if self._shmem_bridge:
                try:
                    self._shmem_bridge.close()
                    self.logger.debug("Shared memory cleaned up")
                except Exception as e:
                    self.logger.warning(f"Failed to clean up shared memory: {e}")
                finally:
                    self._shmem_bridge = None

    def _cleanup_dead_worker(self) -> None:
        """Clean up state after worker died unexpectedly."""
        self.logger.warning("Cleaning up dead worker")

        # PHASE 2: Clean up shared memory
        if self._shmem_bridge:
            try:
                self._shmem_bridge.close()
                self.logger.debug("Shared memory cleaned up")
            except Exception as e:
                self.logger.warning(f"Failed to clean up shared memory: {e}")
            finally:
                self._shmem_bridge = None

        self.active_worker = None
        self.active_model_name = None
        self.active_memory_gb = 0.0

    def get_status(self) -> Dict[str, Any]:
        """Get current worker status."""
        if self.active_worker is None:
            return {
                "model_loaded": False,
                "model_name": None,
                "memory_gb": 0.0
            }

        # Check if worker is actually alive
        if self.active_worker.poll() is not None:
            # Worker died - cleanup state
            self.logger.warning(f"Worker process died unexpectedly in get_status() (returncode: {self.active_worker.returncode})")
            self._cleanup_dead_worker()
            return {
                "model_loaded": False,
                "model_name": None,
                "memory_gb": 0.0
            }

        return {
            "model_loaded": True,
            "model_name": self.active_model_name,
            "memory_gb": self.active_memory_gb
        }

    def get_idle_time(self) -> float:
        """
        Get time since last activity in seconds.

        Returns:
            Seconds since last request activity
        """
        with self.activity_lock:
            return time.time() - self.last_activity_time

    def has_active_requests(self) -> bool:
        """
        Check if there are active requests being processed.

        Returns:
            True if requests are being processed
        """
        with self.activity_lock:
            return self.active_requests > 0

    def _update_activity(self) -> None:
        """Update last activity timestamp."""
        with self.activity_lock:
            self.last_activity_time = time.time()

    def _increment_active_requests(self) -> None:
        """Increment active request counter."""
        with self.activity_lock:
            self.active_requests += 1

    def _decrement_active_requests(self) -> None:
        """Decrement active request counter."""
        with self.activity_lock:
            self.active_requests = max(0, self.active_requests - 1)

    def shutdown(self) -> None:
        """Shutdown worker manager (cleanup on server exit)."""
        with self.lock:
            if self.active_worker is not None:
                self.logger.info("Shutting down worker manager")
                self._unload_model_internal()
