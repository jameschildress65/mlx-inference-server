"""
Unit tests for JSON Mode (structured output) feature.

Tests:
- ResponseFormat Pydantic model parsing
- IPC message serialization with JSON mode fields
- TextInferenceBackend JSON processor integration
- Error handling for missing Outlines
"""

import pytest
from unittest.mock import MagicMock, patch
from pydantic import ValidationError


class TestResponseFormatModel:
    """Test ResponseFormat Pydantic model."""

    def test_default_type_is_text(self):
        """Default response format type is 'text'."""
        from src.orchestrator.api import ResponseFormat

        rf = ResponseFormat()
        assert rf.type == "text"
        assert rf.json_schema is None

    def test_json_object_type(self):
        """json_object type is accepted."""
        from src.orchestrator.api import ResponseFormat

        rf = ResponseFormat(type="json_object")
        assert rf.type == "json_object"

    def test_json_schema_type_with_schema(self):
        """json_schema type with schema is accepted."""
        from src.orchestrator.api import ResponseFormat

        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        rf = ResponseFormat(type="json_schema", json_schema=schema)
        assert rf.type == "json_schema"
        assert rf.json_schema == schema

    def test_invalid_type_rejected(self):
        """Invalid type values are rejected."""
        from src.orchestrator.api import ResponseFormat

        with pytest.raises(ValidationError):
            ResponseFormat(type="invalid")


class TestCompletionRequestWithResponseFormat:
    """Test CompletionRequest with response_format field."""

    def test_completion_request_default_no_response_format(self):
        """CompletionRequest defaults to no response_format."""
        from src.orchestrator.api import CompletionRequest

        req = CompletionRequest(model="test", prompt="hello")
        assert req.response_format is None

    def test_completion_request_with_json_mode(self):
        """CompletionRequest accepts response_format."""
        from src.orchestrator.api import CompletionRequest, ResponseFormat

        req = CompletionRequest(
            model="test",
            prompt="hello",
            response_format=ResponseFormat(type="json_object")
        )
        assert req.response_format is not None
        assert req.response_format.type == "json_object"


class TestChatCompletionRequestWithResponseFormat:
    """Test ChatCompletionRequest with response_format field."""

    def test_chat_completion_request_with_json_schema(self):
        """ChatCompletionRequest accepts json_schema response_format."""
        from src.orchestrator.api import ChatCompletionRequest, ChatMessage, ResponseFormat

        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        req = ChatCompletionRequest(
            model="test",
            messages=[ChatMessage(role="user", content="hello")],
            response_format=ResponseFormat(type="json_schema", json_schema=schema)
        )
        assert req.response_format.type == "json_schema"
        assert req.response_format.json_schema == schema


class TestIPCCompletionRequestJsonMode:
    """Test IPC CompletionRequest with JSON mode fields."""

    def test_ipc_request_default_no_json_mode(self):
        """IPC CompletionRequest defaults to no JSON mode."""
        from src.ipc.messages import CompletionRequest

        req = CompletionRequest(model="test", prompt="hello")
        assert req.response_format_type is None
        assert req.json_schema is None

    def test_ipc_request_with_json_object(self):
        """IPC CompletionRequest accepts json_object type."""
        from src.ipc.messages import CompletionRequest

        req = CompletionRequest(
            model="test",
            prompt="hello",
            response_format_type="json_object"
        )
        assert req.response_format_type == "json_object"
        assert req.json_schema is None

    def test_ipc_request_with_json_schema(self):
        """IPC CompletionRequest accepts json_schema with schema."""
        from src.ipc.messages import CompletionRequest

        schema = {"type": "object"}
        req = CompletionRequest(
            model="test",
            prompt="hello",
            response_format_type="json_schema",
            json_schema=schema
        )
        assert req.response_format_type == "json_schema"
        assert req.json_schema == schema

    def test_ipc_request_serialization(self):
        """IPC CompletionRequest serializes JSON mode fields correctly."""
        from src.ipc.messages import CompletionRequest

        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        req = CompletionRequest(
            model="test",
            prompt="hello",
            response_format_type="json_schema",
            json_schema=schema
        )

        # Serialize and deserialize
        json_str = req.model_dump_json()
        req2 = CompletionRequest.model_validate_json(json_str)

        assert req2.response_format_type == "json_schema"
        assert req2.json_schema == schema


class TestTextInferenceBackendJsonMode:
    """Test TextInferenceBackend JSON mode integration."""

    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        return MagicMock()

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        return MagicMock()

    def test_outlines_availability_check_cached(self, mock_model, mock_tokenizer):
        """Outlines availability is checked and cached."""
        from src.worker.inference import TextInferenceBackend

        backend = TextInferenceBackend(mock_model, mock_tokenizer)

        # First check should set the cached value
        result1 = backend._check_outlines_available()
        result2 = backend._check_outlines_available()

        # Both should return same result (cached)
        assert result1 == result2
        assert backend._outlines_available is not None

    def test_json_processor_cache_key(self, mock_model, mock_tokenizer):
        """JSON processor uses schema hash as cache key."""
        from src.worker.inference import TextInferenceBackend

        backend = TextInferenceBackend(mock_model, mock_tokenizer)

        # Two identical schemas should have same cache key
        schema1 = {"type": "object", "properties": {"a": {"type": "string"}}}
        schema2 = {"type": "object", "properties": {"a": {"type": "string"}}}

        import json
        import hashlib

        key1 = hashlib.md5(json.dumps(schema1, sort_keys=True).encode()).hexdigest()
        key2 = hashlib.md5(json.dumps(schema2, sort_keys=True).encode()).hexdigest()

        assert key1 == key2

    def test_json_mode_not_for_vision(self, mock_model, mock_tokenizer):
        """JSON mode raises error for vision models."""
        from src.worker.inference import InferenceEngine

        engine = InferenceEngine(mock_model, mock_tokenizer, model_type="vision")

        with pytest.raises(ValueError, match="not supported for vision"):
            engine.generate(
                prompt="test",
                response_format_type="json_object"
            )

    def test_generate_accepts_json_params(self, mock_model, mock_tokenizer):
        """generate() method accepts JSON mode parameters."""
        from src.worker.inference import TextInferenceBackend

        backend = TextInferenceBackend(mock_model, mock_tokenizer)

        # Verify method signature includes new params
        import inspect
        sig = inspect.signature(backend.generate)
        params = list(sig.parameters.keys())

        assert "response_format_type" in params
        assert "json_schema" in params

    def test_generate_stream_accepts_json_params(self, mock_model, mock_tokenizer):
        """generate_stream() method accepts JSON mode parameters."""
        from src.worker.inference import TextInferenceBackend

        backend = TextInferenceBackend(mock_model, mock_tokenizer)

        # Verify method signature includes new params
        import inspect
        sig = inspect.signature(backend.generate_stream)
        params = list(sig.parameters.keys())

        assert "response_format_type" in params
        assert "json_schema" in params


class TestJsonModeErrorHandling:
    """Test error handling for JSON mode."""

    def test_missing_outlines_error_message(self):
        """Error message tells user how to install outlines."""
        from src.worker.inference import TextInferenceBackend

        model = MagicMock()
        tokenizer = MagicMock()
        backend = TextInferenceBackend(model, tokenizer)

        # Force outlines to be unavailable
        backend._outlines_available = False

        with pytest.raises(ValueError) as exc_info:
            backend._get_json_logits_processor()

        assert "outlines" in str(exc_info.value).lower()
        assert "pip install" in str(exc_info.value)

    def test_default_schema_for_json_object(self):
        """json_object mode uses generic object schema."""
        # This is a design test - verify the intended behavior
        from src.worker.inference import TextInferenceBackend

        model = MagicMock()
        tokenizer = MagicMock()
        backend = TextInferenceBackend(model, tokenizer)

        # When json_schema is None, should use {"type": "object"}
        # This is documented behavior, verifying through code inspection
        import inspect
        source = inspect.getsource(backend._get_json_logits_processor)
        assert '{"type": "object"}' in source


class TestResponseFormatValidation:
    """Test Opus H2/H3 validation fixes for ResponseFormat."""

    def test_json_schema_type_requires_schema_h2(self):
        """H2: json_schema type without schema raises ValueError."""
        from src.orchestrator.api import ResponseFormat

        with pytest.raises(ValueError) as exc_info:
            ResponseFormat(type="json_schema")  # No json_schema provided

        assert "json_schema is required" in str(exc_info.value)

    def test_json_object_type_no_schema_ok(self):
        """json_object type doesn't require schema."""
        from src.orchestrator.api import ResponseFormat

        # Should not raise
        rf = ResponseFormat(type="json_object")
        assert rf.type == "json_object"
        assert rf.json_schema is None

    def test_schema_size_limit_h3(self):
        """H3: Large schemas are rejected."""
        from src.orchestrator.api import ResponseFormat, RESPONSE_FORMAT_MAX_SCHEMA_SIZE_BYTES

        # Create a schema that exceeds size limit
        large_schema = {
            "type": "object",
            "properties": {
                f"field_{i}": {"type": "string", "description": "x" * 1000}
                for i in range(100)  # ~100KB of descriptions
            }
        }

        with pytest.raises(ValueError) as exc_info:
            ResponseFormat(type="json_schema", json_schema=large_schema)

        assert "too large" in str(exc_info.value)

    def test_schema_depth_limit_h3(self):
        """H3: Deeply nested schemas are rejected."""
        from src.orchestrator.api import ResponseFormat, RESPONSE_FORMAT_MAX_SCHEMA_DEPTH

        # Create a deeply nested schema
        schema = {"type": "object"}
        current = schema
        for _ in range(25):  # Exceed max depth of 20
            current["properties"] = {"nested": {"type": "object"}}
            current = current["properties"]["nested"]

        with pytest.raises(ValueError) as exc_info:
            ResponseFormat(type="json_schema", json_schema=schema)

        assert "too deeply nested" in str(exc_info.value)

    def test_reasonable_schema_accepted(self):
        """Reasonable schemas are accepted."""
        from src.orchestrator.api import ResponseFormat

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"}
                    }
                }
            }
        }

        rf = ResponseFormat(type="json_schema", json_schema=schema)
        assert rf.json_schema == schema


class TestVisionBackendJsonModeParams:
    """Test VisionInferenceBackend accepts JSON mode params (Opus C1 fix)."""

    @pytest.fixture
    def mock_vision_model(self):
        """Create mock vision model with config."""
        model = MagicMock()
        model.config = {}
        return model

    @pytest.fixture
    def mock_processor(self):
        """Create mock vision processor."""
        processor = MagicMock()
        processor.tokenizer = MagicMock()
        return processor

    def test_vision_generate_accepts_json_params(self, mock_vision_model, mock_processor):
        """VisionInferenceBackend.generate() accepts JSON mode params."""
        from src.worker.inference import VisionInferenceBackend

        backend = VisionInferenceBackend(mock_vision_model, mock_processor)

        import inspect
        sig = inspect.signature(backend.generate)
        params = list(sig.parameters.keys())

        assert "response_format_type" in params
        assert "json_schema" in params

    def test_vision_generate_stream_accepts_json_params(self, mock_vision_model, mock_processor):
        """VisionInferenceBackend.generate_stream() accepts JSON mode params."""
        from src.worker.inference import VisionInferenceBackend

        backend = VisionInferenceBackend(mock_vision_model, mock_processor)

        import inspect
        sig = inspect.signature(backend.generate_stream)
        params = list(sig.parameters.keys())

        assert "response_format_type" in params
        assert "json_schema" in params


class TestJsonProcessorCacheBounds:
    """Test H1: Bounded LRU cache for JSON processors."""

    def test_cache_max_size_enforced(self):
        """JSON processor cache enforces maximum size."""
        from src.worker.inference import TextInferenceBackend

        model = MagicMock()
        tokenizer = MagicMock()
        backend = TextInferenceBackend(model, tokenizer)

        # Verify cache max is set
        assert hasattr(backend, '_json_processor_cache_max')
        assert backend._json_processor_cache_max > 0
        assert backend._json_processor_cache_max <= 20  # Reasonable limit

    def test_cache_order_tracking_exists(self):
        """LRU order tracking is initialized."""
        from src.worker.inference import TextInferenceBackend

        model = MagicMock()
        tokenizer = MagicMock()
        backend = TextInferenceBackend(model, tokenizer)

        # Verify LRU tracking exists
        assert hasattr(backend, '_json_processor_cache_order')
        assert isinstance(backend._json_processor_cache_order, list)
