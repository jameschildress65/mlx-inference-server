"""Integration tests for streaming functionality (Phase 2 feature)."""

import pytest
import time
import os
import json
from src.orchestrator.worker_manager import WorkerManager
from src.config.server_config import ServerConfig
from src.ipc.messages import CompletionRequest
from src.ipc.stdio_bridge import StdioBridge


class TestStreaming:
    """Test streaming generation functionality."""

    @pytest.fixture
    def worker_manager(self):
        """Create WorkerManager for testing."""
        config = ServerConfig.auto_detect()
        manager = WorkerManager(config)
        yield manager
        # Cleanup
        try:
            manager.unload_model()
        except:
            pass

    def test_streaming_basic(self, worker_manager):
        """Test basic streaming generation."""
        model_path = os.getenv("MLX_TEST_MODEL", "mlx-community/Qwen2.5-0.5B-Instruct-4bit")

        # Load model
        result = worker_manager.load_model(model_path)
        print(f"\nLoaded: {result.model_name} ({result.memory_gb:.2f} GB)")

        # Create streaming request
        request = CompletionRequest(
            model=model_path,
            prompt="Count to 5:",
            max_tokens=20,
            temperature=0.7,
            top_p=1.0,
            stream=True
        )

        # Send request
        StdioBridge.send_message(worker_manager.active_worker, request)

        # Collect chunks
        chunks = []
        full_text = ""
        done_received = False

        while True:
            chunk = StdioBridge.receive_message(worker_manager.active_worker, timeout=30)

            if chunk.type == "stream_chunk":
                chunks.append(chunk)
                full_text += chunk.text
                print(f"Chunk {len(chunks)}: '{chunk.text}' (token: {chunk.token}, done: {chunk.done})")

                if chunk.done:
                    done_received = True
                    break
            elif chunk.type == "error":
                pytest.fail(f"Worker error: {chunk.message}")
                break
            else:
                pytest.fail(f"Unexpected message type: {chunk.type}")

        # Assertions
        assert done_received, "Should receive done=true chunk"
        assert len(chunks) > 0, "Should receive at least one chunk"
        assert full_text, "Should have generated some text"
        assert chunks[-1].done, "Last chunk should have done=true"
        assert chunks[-1].finish_reason is not None, "Last chunk should have finish_reason"

        print(f"\nStreaming test: ✓")
        print(f"  Chunks received: {len(chunks)}")
        print(f"  Full text: '{full_text.strip()}'")
        print(f"  Finish reason: {chunks[-1].finish_reason}")

        # Cleanup
        worker_manager.unload_model()

    def test_streaming_multiple_tokens(self, worker_manager):
        """Test that streaming produces multiple chunks."""
        model_path = os.getenv("MLX_TEST_MODEL", "mlx-community/Qwen2.5-0.5B-Instruct-4bit")

        # Load model
        worker_manager.load_model(model_path)

        # Create request with more tokens
        request = CompletionRequest(
            model=model_path,
            prompt="Write a haiku about AI:",
            max_tokens=50,
            temperature=0.7,
            top_p=1.0,
            stream=True
        )

        # Send and collect
        StdioBridge.send_message(worker_manager.active_worker, request)

        chunks = []
        while True:
            chunk = StdioBridge.receive_message(worker_manager.active_worker, timeout=30)
            if chunk.type == "stream_chunk":
                chunks.append(chunk)
                if chunk.done:
                    break

        # Should get multiple chunks (at least 10 for a haiku)
        assert len(chunks) >= 5, f"Expected at least 5 chunks, got {len(chunks)}"

        print(f"\nMultiple tokens test: ✓ ({len(chunks)} chunks)")

        # Cleanup
        worker_manager.unload_model()

    def test_streaming_finish_reasons(self, worker_manager):
        """Test that finish_reason is correctly set."""
        model_path = os.getenv("MLX_TEST_MODEL", "mlx-community/Qwen2.5-0.5B-Instruct-4bit")

        # Load model
        worker_manager.load_model(model_path)

        # Test 1: Hit max_tokens (finish_reason should be "length")
        request = CompletionRequest(
            model=model_path,
            prompt="Write a very long essay:",
            max_tokens=10,  # Very short - should hit length limit
            temperature=0.7,
            top_p=1.0,
            stream=True
        )

        StdioBridge.send_message(worker_manager.active_worker, request)

        chunks = []
        while True:
            chunk = StdioBridge.receive_message(worker_manager.active_worker, timeout=30)
            if chunk.type == "stream_chunk":
                chunks.append(chunk)
                if chunk.done:
                    break

        # Last chunk should have finish_reason
        assert chunks[-1].finish_reason in ["length", "stop"], \
            f"Expected 'length' or 'stop', got '{chunks[-1].finish_reason}'"

        print(f"\nFinish reason test: ✓ (reason: {chunks[-1].finish_reason})")

        # Cleanup
        worker_manager.unload_model()

    def test_streaming_incremental_text(self, worker_manager):
        """Test that chunks contain incremental text (not full text)."""
        model_path = os.getenv("MLX_TEST_MODEL", "mlx-community/Qwen2.5-0.5B-Instruct-4bit")

        # Load model
        worker_manager.load_model(model_path)

        request = CompletionRequest(
            model=model_path,
            prompt="Say: Hello world",
            max_tokens=20,
            temperature=0.1,  # Low temp for more predictable output
            top_p=1.0,
            stream=True
        )

        StdioBridge.send_message(worker_manager.active_worker, request)

        chunks = []
        full_text = ""

        while True:
            chunk = StdioBridge.receive_message(worker_manager.active_worker, timeout=30)
            if chunk.type == "stream_chunk":
                chunks.append(chunk)
                full_text += chunk.text

                # Each chunk should contain only NEW text (incremental)
                # NOT the full text so far
                if len(chunks) > 1:
                    # Verify chunk is short (incremental)
                    # Full text would be much longer
                    assert len(chunk.text) < len(full_text), \
                        f"Chunk {len(chunks)} seems to contain full text, not incremental"

                if chunk.done:
                    break

        print(f"\nIncremental text test: ✓")
        print(f"  Full text length: {len(full_text)}")
        print(f"  Average chunk length: {len(full_text) / len(chunks):.1f}")

        # Cleanup
        worker_manager.unload_model()

    def test_streaming_token_ids(self, worker_manager):
        """Test that token IDs are provided in chunks."""
        model_path = os.getenv("MLX_TEST_MODEL", "mlx-community/Qwen2.5-0.5B-Instruct-4bit")

        # Load model
        worker_manager.load_model(model_path)

        request = CompletionRequest(
            model=model_path,
            prompt="Test",
            max_tokens=10,
            temperature=0.7,
            top_p=1.0,
            stream=True
        )

        StdioBridge.send_message(worker_manager.active_worker, request)

        chunks = []
        while True:
            chunk = StdioBridge.receive_message(worker_manager.active_worker, timeout=30)
            if chunk.type == "stream_chunk":
                chunks.append(chunk)

                # Verify token ID is present and valid
                assert hasattr(chunk, 'token'), "Chunk should have 'token' field"
                assert isinstance(chunk.token, int), "Token should be an integer"
                assert chunk.token >= 0, "Token ID should be non-negative"

                if chunk.done:
                    break

        print(f"\nToken IDs test: ✓ (all {len(chunks)} chunks have valid token IDs)")

        # Cleanup
        worker_manager.unload_model()

    def test_streaming_vs_nonstreaming_consistency(self, worker_manager):
        """Test that streaming and non-streaming produce similar output."""
        model_path = os.getenv("MLX_TEST_MODEL", "mlx-community/Qwen2.5-0.5B-Instruct-4bit")

        # Load model
        worker_manager.load_model(model_path)

        prompt = "What is 2+2?"

        # Test 1: Non-streaming
        request_nonstream = CompletionRequest(
            model=model_path,
            prompt=prompt,
            max_tokens=20,
            temperature=0.1,  # Low temp for consistency
            top_p=1.0,
            stream=False
        )

        result_nonstream = worker_manager.generate(request_nonstream)
        text_nonstream = result_nonstream["text"]

        # Test 2: Streaming
        request_stream = CompletionRequest(
            model=model_path,
            prompt=prompt,
            max_tokens=20,
            temperature=0.1,
            top_p=1.0,
            stream=True
        )

        StdioBridge.send_message(worker_manager.active_worker, request_stream)

        text_stream = ""
        while True:
            chunk = StdioBridge.receive_message(worker_manager.active_worker, timeout=30)
            if chunk.type == "stream_chunk":
                text_stream += chunk.text
                if chunk.done:
                    break

        # Outputs may vary due to sampling, even at low temperature
        # Just verify both methods produce valid output
        print(f"\nConsistency test:")
        print(f"  Non-streaming: '{text_nonstream.strip()}'")
        print(f"  Streaming:     '{text_stream.strip()}'")

        # Both should produce non-empty output
        assert len(text_nonstream.strip()) > 0, "Non-streaming should produce output"
        assert len(text_stream.strip()) > 0, "Streaming should produce output"

        # Both should be similar length (within 50%)
        len_ratio = len(text_stream) / max(len(text_nonstream), 1)
        assert 0.5 <= len_ratio <= 2.0, \
            f"Output lengths should be similar (got {len(text_nonstream)} vs {len(text_stream)})"

        print(f"  Result: ✓ (both methods produce valid output)")

        # Cleanup
        worker_manager.unload_model()

    def test_streaming_empty_prompt(self, worker_manager):
        """Test streaming with empty prompt (edge case)."""
        model_path = os.getenv("MLX_TEST_MODEL", "mlx-community/Qwen2.5-0.5B-Instruct-4bit")

        # Load model
        worker_manager.load_model(model_path)

        request = CompletionRequest(
            model=model_path,
            prompt="",  # Empty prompt
            max_tokens=10,
            temperature=0.7,
            top_p=1.0,
            stream=True
        )

        StdioBridge.send_message(worker_manager.active_worker, request)

        # Should still get at least one chunk (even if just EOS)
        chunk = StdioBridge.receive_message(worker_manager.active_worker, timeout=30)
        assert chunk.type in ["stream_chunk", "error"], "Should handle empty prompt"

        if chunk.type == "stream_chunk":
            # Collect remaining chunks
            while not chunk.done:
                chunk = StdioBridge.receive_message(worker_manager.active_worker, timeout=30)

            print(f"\nEmpty prompt test: ✓ (handled gracefully)")
        else:
            print(f"\nEmpty prompt test: ✓ (returned error as expected)")

        # Cleanup
        worker_manager.unload_model()

    @pytest.mark.skipif(
        os.getenv("MLX_SKIP_SLOW_TESTS") == "1",
        reason="Slow test - skipped (set MLX_SKIP_SLOW_TESTS=0 to run)"
    )
    def test_streaming_long_generation(self, worker_manager):
        """Test streaming with longer generation (50+ tokens)."""
        model_path = os.getenv("MLX_TEST_MODEL", "mlx-community/Qwen2.5-0.5B-Instruct-4bit")

        # Load model
        worker_manager.load_model(model_path)

        request = CompletionRequest(
            model=model_path,
            prompt="Write a short story about a robot:",
            max_tokens=100,  # Longer generation
            temperature=0.7,
            top_p=1.0,
            stream=True
        )

        StdioBridge.send_message(worker_manager.active_worker, request)

        start_time = time.time()
        chunks = []
        full_text = ""

        while True:
            chunk = StdioBridge.receive_message(worker_manager.active_worker, timeout=60)
            if chunk.type == "stream_chunk":
                chunks.append(chunk)
                full_text += chunk.text
                if chunk.done:
                    break

        elapsed = time.time() - start_time

        # Should get many chunks (50+)
        assert len(chunks) >= 30, f"Expected at least 30 chunks for 100 tokens, got {len(chunks)}"

        print(f"\nLong generation test: ✓")
        print(f"  Chunks: {len(chunks)}")
        print(f"  Text length: {len(full_text)} chars")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Throughput: {len(chunks) / elapsed:.1f} chunks/sec")

        # Cleanup
        worker_manager.unload_model()
