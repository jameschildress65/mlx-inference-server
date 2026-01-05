"""ModelWorker - Worker subprocess that loads and serves one MLX model."""

import sys
import os
import signal
import logging
from pathlib import Path
import setproctitle

# CRITICAL: Configure MLX environment BEFORE any mlx imports
# Based on Opus 4.5 Apple Silicon performance review
def configure_mlx_environment():
    """
    Configure MLX for maximum performance on Apple Silicon.
    Must be called before any mlx import.

    Optimizations (Opus 4.5 recommendations):
    1. Enable Metal shader cache (faster startup, pre-compiled kernels)
    2. Use system malloc for better large allocation performance
    3. Prepare for future memory limits (multi-worker scenarios)

    Expected gain: 1-2% performance improvement + faster startup
    """
    # Enable Metal shader cache (compiles Metal kernels once, reuses on subsequent runs)
    os.environ['MTL_SHADER_CACHE_ENABLE'] = '1'

    # Use system malloc instead of Python's pymalloc
    # Better performance for large allocations (model weights in unified memory)
    os.environ['PYTHONMALLOC'] = 'malloc'

    # Optional: Set Metal memory limit for multi-worker scenarios
    # Uncomment when running multiple workers to prevent one from consuming all memory
    # Example for M4 Max (128GB): limit each worker to 80GB
    # os.environ['MLX_METAL_MEMORY_LIMIT'] = str(80 * 1024**3)

    logging.getLogger(__name__).debug("MLX environment configured for Apple Silicon")

# Configure before any imports that might use mlx
configure_mlx_environment()

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ipc.stdio_bridge import WorkerStdioHandler
from src.ipc.shared_memory_bridge import SharedMemoryBridge, WorkerSharedMemoryHandler
from src.ipc.messages import CompletionRequest, PingMessage, ShutdownMessage
from src.worker.model_loader import ModelLoader
from src.worker.inference import InferenceEngine
from src.worker.generation_monitor import get_monitor, GenerationTimeout
from src.utils.memory_utils import get_memory_usage_gb

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # PHASE 2: DEBUG to see shared memory details
    format='[WORKER] %(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]  # Log to stderr, stdout reserved for IPC
)
logger = logging.getLogger(__name__)


def detect_ipc_method(shm_name: str = None):
    """
    Detect which IPC method orchestrator is using.

    PHASE 2: If shm_name provided, attach to that shared memory. Otherwise use stdio.

    Args:
        shm_name: Shared memory name passed from orchestrator (or None for stdio)

    Returns:
        Tuple of (handler, bridge):
        - handler: WorkerSharedMemoryHandler or WorkerStdioHandler (class, not instance)
        - bridge: SharedMemoryBridge instance or None
    """
    pid = os.getpid()

    if shm_name:
        # Orchestrator passed shared memory name - attach to it
        try:
            logger.debug(f"Worker {pid} attempting to attach to shared memory: {shm_name}")
            bridge = SharedMemoryBridge(name=shm_name, is_server=False)
            logger.debug(f"Worker {pid} shared memory bridge attached successfully")
            handler = WorkerSharedMemoryHandler(bridge)
            logger.info(f"Worker {pid} using shared memory IPC: {shm_name}")
            return handler, bridge
        except FileNotFoundError as e:
            # Shared memory doesn't exist (shouldn't happen - orchestrator creates first)
            logger.error(f"Worker {pid} shared memory not found: {shm_name} (BUG: orchestrator should create first)")
            logger.info(f"Falling back to stdio IPC")
            return WorkerStdioHandler, None
        except Exception as e:
            # Shared memory failed for other reason → fall back to stdio
            logger.error(f"Shared memory attach failed: {type(e).__name__}: {e}", exc_info=True)
            logger.info(f"Falling back to stdio IPC")
            return WorkerStdioHandler, None
    else:
        # No shared memory name provided - use stdio
        logger.info(f"Worker {pid} using stdio IPC (no shared memory name provided)")
        return WorkerStdioHandler, None


class WorkerProcess:
    """Main worker process class."""

    def __init__(self, model_path: str, worker_id: int = 1, shm_name: str = None):
        """
        Initialize worker with model path.

        Args:
            model_path: HuggingFace model path (e.g., "mlx-community/Qwen2.5-7B-Instruct-4bit")
            worker_id: Worker number for process naming (default: 1)
            shm_name: Shared memory name for IPC (None = use stdio)
        """
        # Set process name for easy identification in ps/top
        setproctitle.setproctitle(f"mlx-server-v3-{worker_id}")

        self.model_path = model_path
        self.worker_id = worker_id
        self.model_loader = ModelLoader()
        self.inference_engine = None
        self.running = True

        # PHASE 2: Detect IPC method (shared memory or stdio)
        self.handler, self.shmem_bridge = detect_ipc_method(shm_name)

        # Initialize generation monitor (600s = 10 min timeout, 10s heartbeat)
        self.gen_monitor = get_monitor(max_seconds=600, heartbeat_interval=10)
        self.gen_monitor.start_heartbeat()

        # Set up signal handlers
        signal.signal(signal.SIGTERM, self._handle_sigterm)
        signal.signal(signal.SIGINT, self._handle_sigint)

    def _cleanup(self):
        """Clean up resources (PHASE 2: includes shared memory)."""
        # Stop heartbeat monitoring
        if self.gen_monitor:
            try:
                self.gen_monitor.stop_heartbeat()
            except Exception as e:
                logger.warning(f"Failed to stop heartbeat: {e}")

        # Clean up shared memory
        if self.shmem_bridge:
            try:
                self.shmem_bridge.close()
                logger.debug("Shared memory cleaned up")
            except Exception as e:
                logger.warning(f"Failed to clean up shared memory: {e}")

    def _handle_sigterm(self, signum, frame):
        """Handle SIGTERM - graceful shutdown."""
        logger.info("SIGTERM received, shutting down worker")
        self.running = False
        self._cleanup()
        sys.exit(0)

    def _handle_sigint(self, signum, frame):
        """Handle SIGINT - graceful shutdown."""
        logger.info("SIGINT received, shutting down worker")
        self.running = False
        self._cleanup()
        sys.exit(0)

    def run(self):
        """Main worker loop."""
        try:
            # Load model
            logger.info(f"Loading model: {self.model_path}")
            model, tokenizer_or_processor = self.model_loader.load(self.model_path)

            # Phase 3: Detect model capabilities
            capabilities = ModelLoader.detect_model_capabilities(self.model_path)
            model_type = "vision" if capabilities["vision"] else "text"
            logger.info(f"Model type: {model_type} (capabilities: {capabilities})")

            # Create inference engine with appropriate backend
            self.inference_engine = InferenceEngine(model, tokenizer_or_processor, model_type=model_type)

            # Get memory usage
            memory_gb = get_memory_usage_gb()
            logger.info(f"Model loaded successfully, memory: {memory_gb:.2f} GB")

            # Send ready signal
            self.handler.send_ready(
                model_name=self.model_path,
                memory_gb=memory_gb
            )

            # Main request loop
            logger.info("Worker ready, entering request loop")
            while self.running:
                request = self.handler.receive_request()

                if request is None:
                    # stdin closed, orchestrator died
                    logger.info("stdin closed, exiting")
                    break

                if isinstance(request, PingMessage):
                    # Health check
                    self.handler.send_pong()

                elif isinstance(request, ShutdownMessage):
                    # Graceful shutdown
                    logger.info("Shutdown message received")
                    break

                elif isinstance(request, CompletionRequest):
                    # Generate completion with monitoring
                    try:
                        with self.gen_monitor.monitor_generation(
                            prompt_length=len(request.prompt),
                            max_tokens=request.max_tokens,
                            temperature=request.temperature
                        ):
                            if request.stream:
                                # Streaming completion (Phase 2/3)
                                for chunk in self.inference_engine.generate_stream(
                                    prompt=request.prompt,
                                    max_tokens=request.max_tokens,
                                    temperature=request.temperature,
                                    top_p=request.top_p,
                                    repetition_penalty=request.repetition_penalty,
                                    images=request.images  # Phase 3: Pass images for vision models
                                ):
                                    # Send each chunk
                                    self.handler.send_stream_chunk(
                                        text=chunk["text"],
                                        token=chunk["token"],
                                        done=chunk["done"],
                                        finish_reason=chunk["finish_reason"]
                                    )
                            else:
                                # Non-streaming completion
                                result = self.inference_engine.generate(
                                    prompt=request.prompt,
                                    max_tokens=request.max_tokens,
                                    temperature=request.temperature,
                                    top_p=request.top_p,
                                    repetition_penalty=request.repetition_penalty,
                                    images=request.images  # Phase 3: Pass images for vision models
                                )

                                self.handler.send_completion(
                                    text=result["text"],
                                    tokens=result["tokens"],
                                    finish_reason=result["finish_reason"]
                                )

                    except GenerationTimeout as e:
                        # Worker hung - timeout exceeded
                        logger.error(f"Generation timeout - worker terminating: {e}")
                        self.handler.send_error(
                            error="GenerationTimeout",
                            message=f"Worker hung for {self.gen_monitor.max_seconds}s - terminating"
                        )
                        # Exit immediately - this worker is likely deadlocked
                        self._cleanup()
                        sys.exit(1)

                    except Exception as e:
                        logger.error(f"Generation failed: {e}", exc_info=True)
                        self.handler.send_error(
                            error=e.__class__.__name__,
                            message=str(e)
                        )

            logger.info("Worker exiting normally")
            self._cleanup()  # PHASE 2: Clean up shared memory
            sys.exit(0)

        except Exception as e:
            logger.error(f"Worker failed: {e}", exc_info=True)
            self.handler.send_error(
                error=e.__class__.__name__,
                message=str(e)
            )
            self._cleanup()  # PHASE 2: Clean up shared memory
            sys.exit(1)


def validate_model_path(model_path: str) -> None:
    """
    Validate model path on worker side (defense in depth).

    Security: Opus 4.5 Critical Fix C2
    - Ensures model path follows org/model format
    - Blocks path traversal attempts
    - Validates safe characters only

    Args:
        model_path: Model path to validate

    Raises:
        ValueError: If model path is invalid or unsafe
    """
    import re

    # Must match HuggingFace org/model format
    SAFE_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+/[a-zA-Z0-9._-]+$')

    if not SAFE_PATTERN.match(model_path):
        raise ValueError(
            f"Invalid model path format: {model_path}. "
            f"Expected: org/model (e.g., mlx-community/Qwen2.5-7B-Instruct-4bit)"
        )

    # Explicit path traversal check
    if '..' in model_path or model_path.startswith('/'):
        raise ValueError(f"Path traversal detected: {model_path}")

    # Must be exactly one '/' (org/model format)
    if model_path.count('/') != 1:
        raise ValueError(
            f"Model path must be org/model format: {model_path} "
            f"(has {model_path.count('/')} slashes, expected 1)"
        )

    logger.debug(f"Model path validated: {model_path}")


def main():
    """Worker entry point."""
    if len(sys.argv) < 2:
        print("Usage: python -m src.worker <model_path> [worker_id]", file=sys.stderr)
        sys.exit(1)

    model_path = sys.argv[1]
    worker_id = int(sys.argv[2]) if len(sys.argv) >= 3 else 1

    # Security: Validate model path (Opus 4.5 Critical Fix C2 - defense in depth)
    try:
        validate_model_path(model_path)
    except ValueError as e:
        logger.error(f"Invalid model path: {e}")
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # PHASE 2 Security: Read shared memory name from environment variable
    # (not CLI args, for security - prevents exposure in process listings)
    shm_name = os.environ.get('MLX_SHM_NAME')

    logger.info(f"Worker #{worker_id} starting with model: {model_path}")
    if shm_name:
        logger.info(f"Worker #{worker_id} using shared memory: {shm_name}")
    else:
        logger.info(f"Worker #{worker_id} using stdio IPC (no MLX_SHM_NAME env var)")

    worker = WorkerProcess(model_path, worker_id, shm_name)
    worker.run()


if __name__ == "__main__":
    main()
