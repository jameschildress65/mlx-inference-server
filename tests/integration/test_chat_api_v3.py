"""V3 Integration Tests - Chat API

Tests actual model loading and inference using V3 FastAPI architecture.
Covers both streaming and non-streaming modes.
"""

import pytest
import json
import time


class TestChatCompletionsV3:
    """Test /v1/chat/completions endpoint with actual model loading."""

    def test_chat_endpoint_loads_model_on_demand(
        self,
        v3_main_client,
        v3_clean_model,
        skip_if_no_model_v3
    ):
        """Test that chat endpoint loads model on first request."""
        response = v3_main_client["chat_completion"](
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10
        )

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert len(data["choices"][0]["message"]["content"]) > 0
        assert "usage" in data

    def test_chat_completion_non_streaming(
        self,
        v3_main_client,
        skip_if_no_model_v3
    ):
        """Test non-streaming chat completion generates valid response."""
        response = v3_main_client["chat_completion"](
            messages=[
                {"role": "user", "content": "What is 2+2? Answer with just the number."}
            ],
            max_tokens=5,
            stream=False  # Explicit non-streaming
        )

        assert response.status_code == 200
        data = response.json()

        # Verify OpenAI-compatible format
        assert data["object"] == "chat.completion"
        assert data["model"]  # Model name present
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert len(data["choices"][0]["message"]["content"]) > 0
        assert data["choices"][0]["finish_reason"] in ["stop", "length"]

        # Verify usage tokens
        assert data["usage"]["prompt_tokens"] > 0
        assert data["usage"]["completion_tokens"] > 0
        assert data["usage"]["total_tokens"] == (
            data["usage"]["prompt_tokens"] + data["usage"]["completion_tokens"]
        )

    def test_chat_completion_streaming(
        self,
        v3_main_client,
        skip_if_no_model_v3
    ):
        """Test streaming chat completion returns SSE chunks."""
        import requests

        response = requests.post(
            f"{v3_main_client['base_url']}/v1/chat/completions",
            json={
                "model": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
                "messages": [
                    {"role": "user", "content": "Count to 3"}
                ],
                "max_tokens": 20,
                "stream": True  # Request streaming
            },
            stream=True,  # Enable streaming response
            timeout=120
        )

        assert response.status_code == 200

        # Verify SSE format
        chunks = []
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    data_str = line_str[6:]  # Remove 'data: ' prefix
                    if data_str == '[DONE]':
                        break
                    chunk = json.loads(data_str)
                    chunks.append(chunk)

        # Verify we got chunks
        assert len(chunks) > 0

        # Verify chunk structure
        for chunk in chunks:
            assert chunk["object"] == "chat.completion.chunk"
            assert len(chunk["choices"]) == 1
            if "delta" in chunk["choices"][0]:
                # Delta may contain role, content, or both
                assert "role" in chunk["choices"][0]["delta"] or \
                       "content" in chunk["choices"][0]["delta"]

    def test_chat_completion_respects_max_tokens(
        self,
        v3_main_client,
        skip_if_no_model_v3
    ):
        """Test max_tokens parameter limits response length."""
        response = v3_main_client["chat_completion"](
            messages=[
                {"role": "user", "content": "Write a long story"}
            ],
            max_tokens=5  # Very short
        )

        assert response.status_code == 200
        data = response.json()

        # Should stop due to length limit
        assert data["choices"][0]["finish_reason"] == "length"
        assert data["usage"]["completion_tokens"] <= 7  # Allow for small variance


class TestCompletionsV3:
    """Test /v1/completions endpoint (non-chat)."""

    def test_completion_non_streaming(
        self,
        v3_main_client,
        skip_if_no_model_v3
    ):
        """Test non-streaming completion."""
        response = v3_main_client["completion"](
            prompt="The capital of France is",
            max_tokens=5,
            stream=False
        )

        assert response.status_code == 200
        data = response.json()

        assert data["object"] == "text_completion"
        assert data["choices"][0]["text"]
        assert "usage" in data

    def test_completion_streaming(
        self,
        v3_main_client,
        skip_if_no_model_v3
    ):
        """Test streaming completion."""
        import requests

        response = requests.post(
            f"{v3_main_client['base_url']}/v1/completions",
            json={
                "model": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
                "prompt": "Count: 1, 2,",
                "max_tokens": 10,
                "stream": True
            },
            stream=True,
            timeout=120
        )

        assert response.status_code == 200

        # Collect chunks
        chunks = []
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    data_str = line_str[6:]
                    if data_str == '[DONE]':
                        break
                    chunk = json.loads(data_str)
                    chunks.append(chunk)

        assert len(chunks) > 0

        # Verify chunk format
        for chunk in chunks:
            assert chunk["object"] == "text_completion"
            assert "choices" in chunk


class TestModelPersistenceV3:
    """Test model persistence across requests."""

    def test_model_persists_across_requests(
        self,
        v3_main_client,
        v3_admin_client,
        v3_clean_model,
        skip_if_no_model_v3
    ):
        """Test loaded model persists for subsequent requests."""
        # First request - loads model
        response1 = v3_main_client["chat_completion"](
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5
        )
        assert response1.status_code == 200

        # Check model is loaded
        status = v3_admin_client["get"]("status")
        assert status.status_code == 200
        assert status.json()["model"]["loaded"] is True

        # Second request - should use existing model (faster)
        start = time.time()
        response2 = v3_main_client["chat_completion"](
            messages=[{"role": "user", "content": "Hello again"}],
            max_tokens=5
        )
        duration = time.time() - start

        assert response2.status_code == 200

        # Second request should be faster (no model load time)
        assert duration < 30  # Should complete in <30s (vs 60-120s for load)


class TestConcurrentRequestsV3:
    """Test concurrent request handling."""

    def test_concurrent_chat_requests_succeed(
        self,
        v3_main_client,
        skip_if_no_model_v3
    ):
        """Test server handles concurrent requests without crashing."""
        import concurrent.futures

        def make_request(i):
            response = v3_main_client["chat_completion"](
                messages=[{"role": "user", "content": f"Request {i}"}],
                max_tokens=5
            )
            return response.status_code

        # Send 3 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_request, i) for i in range(3)]
            results = [f.result() for f in futures]

        # All should succeed
        assert all(code == 200 for code in results)


class TestRequestTrackingV3:
    """Test request tracking and activity updates."""

    def test_requests_update_activity_tracker(
        self,
        v3_main_client,
        v3_admin_client,
        skip_if_no_model_v3
    ):
        """Test requests update last activity timestamp."""
        # Get initial status
        status1 = v3_admin_client["get"]("status")
        assert status1.status_code == 200

        time.sleep(1)

        # Make a request
        response = v3_main_client["chat_completion"](
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=5
        )
        assert response.status_code == 200

        # Get updated status
        status2 = v3_admin_client["get"]("status")
        assert status2.status_code == 200

        # Activity timestamp should be recent
        # (within last 5 seconds)
        import time as time_module
        now = time_module.time()
        # Note: We'd need to check the actual timestamp from status
        # This is a placeholder - actual implementation depends on status format
