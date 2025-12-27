"""Unit tests for IPC messages."""

import pytest
from src.ipc.messages import (
    ReadyMessage,
    CompletionRequest,
    CompletionResponse,
    StreamChunk,
    ErrorMessage,
    PingMessage,
    PongMessage,
    ShutdownMessage,
)


class TestReadyMessage:
    """Tests for ReadyMessage."""

    def test_create_ready_message(self):
        """Test creating a ready message."""
        msg = ReadyMessage(
            model_name="mlx-community/Qwen2.5-7B-Instruct-4bit",
            memory_gb=4.12
        )

        assert msg.type == "ready"
        assert msg.model_name == "mlx-community/Qwen2.5-7B-Instruct-4bit"
        assert msg.memory_gb == 4.12

    def test_ready_message_json(self):
        """Test JSON serialization."""
        msg = ReadyMessage(
            model_name="test-model",
            memory_gb=2.5
        )

        json_str = msg.model_dump_json()
        assert "ready" in json_str
        assert "test-model" in json_str
        assert "2.5" in json_str


class TestCompletionRequest:
    """Tests for CompletionRequest."""

    def test_create_completion_request(self):
        """Test creating a completion request."""
        req = CompletionRequest(
            model="mlx-community/Qwen2.5-7B-Instruct-4bit",
            prompt="Hello, world!",
            max_tokens=50,
            temperature=0.8
        )

        assert req.type == "completion"
        assert req.model == "mlx-community/Qwen2.5-7B-Instruct-4bit"
        assert req.prompt == "Hello, world!"
        assert req.max_tokens == 50
        assert req.temperature == 0.8
        assert req.stream is False

    def test_completion_request_defaults(self):
        """Test default values."""
        req = CompletionRequest(
            model="test-model",
            prompt="test prompt"
        )

        assert req.max_tokens == 100
        assert req.temperature == 0.7
        assert req.top_p == 1.0
        assert req.stream is False


class TestCompletionResponse:
    """Tests for CompletionResponse."""

    def test_create_completion_response(self):
        """Test creating a completion response."""
        resp = CompletionResponse(
            text="This is a test response.",
            tokens=6,
            finish_reason="stop"
        )

        assert resp.type == "completion_response"
        assert resp.text == "This is a test response."
        assert resp.tokens == 6
        assert resp.finish_reason == "stop"


class TestStreamChunk:
    """Tests for StreamChunk."""

    def test_create_stream_chunk(self):
        """Test creating a stream chunk."""
        chunk = StreamChunk(token="Hello", done=False)

        assert chunk.type == "stream_chunk"
        assert chunk.token == "Hello"
        assert chunk.done is False

    def test_stream_chunk_done(self):
        """Test final stream chunk."""
        chunk = StreamChunk(token="", done=True)

        assert chunk.done is True


class TestErrorMessage:
    """Tests for ErrorMessage."""

    def test_create_error_message(self):
        """Test creating an error message."""
        err = ErrorMessage(
            error="OutOfMemoryError",
            message="Failed to allocate 8GB"
        )

        assert err.type == "error"
        assert err.error == "OutOfMemoryError"
        assert err.message == "Failed to allocate 8GB"


class TestPingPong:
    """Tests for ping/pong messages."""

    def test_ping_message(self):
        """Test ping message."""
        ping = PingMessage()
        assert ping.type == "ping"

    def test_pong_message(self):
        """Test pong message."""
        pong = PongMessage()
        assert pong.type == "pong"
        assert pong.status == "healthy"


class TestShutdownMessage:
    """Tests for ShutdownMessage."""

    def test_shutdown_message(self):
        """Test shutdown message."""
        shutdown = ShutdownMessage()
        assert shutdown.type == "shutdown"
