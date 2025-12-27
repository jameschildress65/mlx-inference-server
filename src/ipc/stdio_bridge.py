"""stdin/stdout JSON IPC bridge for orchestrator ↔ worker communication."""

import json
import sys
import select
import logging
import threading
from typing import Optional, Union, Dict
from pydantic import ValidationError

from .messages import (
    ReadyMessage,
    CompletionResponse,
    StreamChunk,
    ErrorMessage,
    PongMessage,
    CompletionRequest,
    PingMessage,
    ShutdownMessage,
)

logger = logging.getLogger(__name__)


class StdioIPCError(Exception):
    """Base exception for stdio IPC errors."""
    pass


class WorkerCommunicationError(StdioIPCError):
    """Worker failed to respond or sent invalid message."""
    pass


class StdioBridge:
    """Manages stdin/stdout JSON communication with worker subprocess."""

    # Per-worker IO locks to prevent concurrent write/read corruption
    # Key: process PID, Value: threading.Lock
    # CRITICAL FIX (Opus 4.5 Review): stdin/stdout not thread-safe
    _process_locks: Dict[int, threading.Lock] = {}
    _locks_lock = threading.Lock()  # Protects _process_locks dict

    @classmethod
    def _get_process_lock(cls, process) -> threading.Lock:
        """Get or create lock for a specific process."""
        pid = process.pid
        with cls._locks_lock:
            if pid not in cls._process_locks:
                cls._process_locks[pid] = threading.Lock()
            return cls._process_locks[pid]

    @classmethod
    def _cleanup_process_lock(cls, process) -> None:
        """Remove lock for dead process (cleanup)."""
        pid = process.pid
        with cls._locks_lock:
            cls._process_locks.pop(pid, None)

    @staticmethod
    def send_message(process, message: Union[CompletionRequest, PingMessage, ShutdownMessage]) -> None:
        """
        Send message to worker via stdin.

        Thread-safe: Uses per-process lock to prevent concurrent writes.

        Args:
            process: subprocess.Popen instance
            message: Pydantic message model

        Raises:
            WorkerCommunicationError: If write fails
        """
        # Opus Fix: Acquire per-process lock to prevent write interleaving
        lock = StdioBridge._get_process_lock(process)
        with lock:
            try:
                line = message.model_dump_json() + "\n"
                process.stdin.write(line)
                process.stdin.flush()
            except (BrokenPipeError, OSError) as e:
                raise WorkerCommunicationError(f"Failed to send message to worker: {e}")

    @staticmethod
    def receive_message(
        process,
        timeout: Optional[float] = None
    ) -> Union[ReadyMessage, CompletionResponse, StreamChunk, ErrorMessage, PongMessage]:
        """
        Receive message from worker via stdout.

        Thread-safe: Uses per-process lock to prevent concurrent reads.

        Args:
            process: subprocess.Popen instance
            timeout: Optional timeout in seconds (default: 30s, never blocks forever)

        Returns:
            Parsed message (Pydantic model)

        Raises:
            WorkerCommunicationError: If read fails, timeout occurs, or message invalid
        """
        # Opus Fix: Acquire per-process lock to prevent read interleaving
        lock = StdioBridge._get_process_lock(process)
        with lock:
            try:
                # Default timeout: never block forever
                if timeout is None:
                    timeout = 30.0

                # Use select() to implement timeout
                ready, _, _ = select.select([process.stdout], [], [], timeout)
                if not ready:
                    raise WorkerCommunicationError(
                        f"Worker timeout after {timeout}s - no response received"
                    )

                # Now safe to read (data available)
                line = process.stdout.readline()
                if not line:
                    raise WorkerCommunicationError("Worker closed stdout (died?)")

                data = json.loads(line)
                msg_type = data.get("type")

                # Parse based on type
                if msg_type == "ready":
                    return ReadyMessage(**data)
                elif msg_type == "completion_response":
                    return CompletionResponse(**data)
                elif msg_type == "stream_chunk":
                    return StreamChunk(**data)
                elif msg_type == "error":
                    return ErrorMessage(**data)
                elif msg_type == "pong":
                    return PongMessage(**data)
                else:
                    raise WorkerCommunicationError(f"Unknown message type: {msg_type}")

            except json.JSONDecodeError as e:
                raise WorkerCommunicationError(f"Invalid JSON from worker: {e}")
            except ValidationError as e:
                raise WorkerCommunicationError(f"Invalid message format: {e}")
            except Exception as e:
                raise WorkerCommunicationError(f"Failed to receive message: {e}")


class WorkerStdioHandler:
    """Worker-side stdio message handler (for use in worker process)."""

    @staticmethod
    def send_ready(model_name: str, memory_gb: float) -> None:
        """Send ready signal to orchestrator."""
        msg = ReadyMessage(model_name=model_name, memory_gb=memory_gb)
        print(msg.model_dump_json(), flush=True)

    @staticmethod
    def send_completion(text: str, tokens: int, finish_reason: str = "stop") -> None:
        """Send completion response to orchestrator."""
        msg = CompletionResponse(text=text, tokens=tokens, finish_reason=finish_reason)
        print(msg.model_dump_json(), flush=True)

    @staticmethod
    def send_stream_chunk(text: str, token: int, done: bool = False, finish_reason: str = None) -> None:
        """Send streaming chunk to orchestrator."""
        msg = StreamChunk(text=text, token=token, done=done, finish_reason=finish_reason)
        print(msg.model_dump_json(), flush=True)

    @staticmethod
    def send_error(error: str, message: str) -> None:
        """Send error to orchestrator."""
        msg = ErrorMessage(error=error, message=message)
        print(msg.model_dump_json(), flush=True)

    @staticmethod
    def send_pong() -> None:
        """Respond to health ping."""
        msg = PongMessage()
        print(msg.model_dump_json(), flush=True)

    @staticmethod
    def receive_request() -> Optional[Union[CompletionRequest, PingMessage, ShutdownMessage]]:
        """
        Receive request from orchestrator via stdin.

        Returns:
            Parsed request or None if stdin closed
        """
        try:
            line = sys.stdin.readline()
            if not line:
                return None

            data = json.loads(line)
            msg_type = data.get("type")

            if msg_type == "completion":
                return CompletionRequest(**data)
            elif msg_type == "ping":
                return PingMessage(**data)
            elif msg_type == "shutdown":
                return ShutdownMessage(**data)
            else:
                logger.warning(f"Unknown message type: {msg_type}")
                return None

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON from orchestrator: {e}")
            return None
        except ValidationError as e:
            logger.error(f"Invalid message format: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to receive request: {e}")
            return None
