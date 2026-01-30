"""OpenAI-compatible response formatters for completions and chat.

Provides formatting utilities for both /v1/completions and /v1/chat/completions
endpoints, handling streaming (SSE) and non-streaming responses.

6.2: Supports unique request IDs for tracing/correlation.
"""

import json
import time
import uuid
from typing import Dict, Any, Optional

from ..ipc.messages import StreamChunk


def generate_completion_id() -> str:
    """Generate unique completion ID for tracing (6.2)."""
    return f"cmpl-{uuid.uuid4().hex[:12]}"


def generate_chat_completion_id() -> str:
    """Generate unique chat completion ID for tracing (6.2)."""
    return f"chatcmpl-{uuid.uuid4().hex[:12]}"


class CompletionFormatter:
    """Formats responses for /v1/completions endpoint (text completions)."""

    @staticmethod
    def format_non_streaming(
        model: str,
        result: Dict[str, Any],
        prompt_tokens: int,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Format non-streaming completion response.

        Args:
            model: Model name
            result: Worker generation result with 'text', 'tokens', 'finish_reason'
            prompt_tokens: Number of prompt tokens
            request_id: Optional request ID for tracing (6.2)

        Returns:
            OpenAI-compatible text completion response
        """
        completion_tokens = result["tokens"]
        return {
            "id": request_id or generate_completion_id(),
            "object": "text_completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "text": result["text"],
                "index": 0,
                "logprobs": None,
                "finish_reason": result["finish_reason"]
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }

    @staticmethod
    def format_stream_chunk(
        model: str,
        chunk: StreamChunk,
        request_id: Optional[str] = None
    ) -> str:
        """Format streaming chunk for text completions (SSE).

        Args:
            model: Model name
            chunk: Stream chunk from worker
            request_id: Optional request ID for tracing (6.2)

        Returns:
            SSE-formatted data line
        """
        data = {
            "id": request_id or generate_completion_id(),
            "object": "text_completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "text": chunk.text if chunk.text else "",
                "finish_reason": chunk.finish_reason
            }]
        }
        return f"data: {json.dumps(data)}\n\n"


class ChatCompletionFormatter:
    """Formats responses for /v1/chat/completions endpoint."""

    @staticmethod
    def format_non_streaming(
        model: str,
        result: Dict[str, Any],
        prompt_tokens: int,
        generation_time: float = 0.0,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Format non-streaming chat completion response.

        Args:
            model: Model name
            result: Worker generation result with 'text', 'tokens', 'finish_reason'
            prompt_tokens: Number of prompt tokens
            generation_time: Time taken for generation (seconds), for tokens/sec calculation
            request_id: Optional request ID for tracing (6.2)

        Returns:
            OpenAI-compatible chat completion response
        """
        completion_tokens = result["tokens"]
        tokens_per_sec = completion_tokens / generation_time if generation_time > 0 else 0.0

        return {
            "id": request_id or generate_chat_completion_id(),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result["text"]
                },
                "finish_reason": result["finish_reason"]
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "tokens_per_sec": round(tokens_per_sec, 2)
            }
        }

    @staticmethod
    def format_stream_chunk(
        model: str,
        chunk: StreamChunk,
        request_id: Optional[str] = None
    ) -> str:
        """Format streaming chunk for chat completions (SSE).

        Uses delta.content format for chat completion streaming
        (OpenAI chat streaming standard).

        Args:
            model: Model name
            chunk: Stream chunk from worker
            request_id: Optional request ID for tracing (6.2)

        Returns:
            SSE-formatted data line
        """
        data = {
            "id": request_id or generate_chat_completion_id(),
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": chunk.text} if chunk.text else {},
                "finish_reason": chunk.finish_reason
            }]
        }
        return f"data: {json.dumps(data)}\n\n"
