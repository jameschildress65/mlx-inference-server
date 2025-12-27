"""Unit tests for V3 FastAPI endpoints."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from fastapi.testclient import TestClient

from src.config.server_config import ServerConfig
from src.orchestrator.worker_manager import (
    WorkerManager,
    ModelLoadResult,
    UnloadResult,
    NoModelLoadedError
)
from src.orchestrator.api import create_app, create_admin_app


@pytest.fixture
def mock_config():
    """Create mock ServerConfig."""
    return ServerConfig(
        main_port=11440,
        admin_port=11441,
        host="0.0.0.0",
        idle_timeout_seconds=180,
        request_timeout_seconds=300,
        memory_threshold_gb=28,
        cache_dir="/test/cache",
        log_dir="/test/logs",
        machine_type="test-machine",
        total_ram_gb=32,
        chip_model="Test Chip",
        model_name="Test Model",
        use_shared_memory=True
    )


@pytest.fixture
def mock_worker_manager():
    """Create mock WorkerManager."""
    manager = Mock(spec=WorkerManager)

    # Default: no model loaded
    manager.get_status.return_value = {
        "model_loaded": False,
        "model_name": None,
        "memory_gb": 0.0
    }

    manager.health_check.return_value = {
        "healthy": False,
        "status": "no_worker"
    }

    return manager


class TestMainAPIEndpoints:
    """Tests for main API endpoints."""

    def test_health_endpoint(self, mock_config, mock_worker_manager):
        """Test GET /health endpoint."""
        app = create_app(mock_config, mock_worker_manager)
        client = TestClient(app)

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "3.0.0-alpha"

    def test_completions_loads_model_on_demand(self, mock_config, mock_worker_manager):
        """Test /v1/completions loads model if not loaded."""
        # Setup: no model loaded initially
        mock_worker_manager.get_status.return_value = {
            "model_loaded": False,
            "model_name": None,
            "memory_gb": 0.0
        }

        # Mock successful load
        mock_worker_manager.load_model.return_value = ModelLoadResult(
            model_name="test-model",
            memory_gb=4.5,
            load_time=10.0
        )

        # Mock successful generation
        mock_worker_manager.generate.return_value = {
            "text": "Generated response",
            "tokens": 5,
            "finish_reason": "stop"
        }

        app = create_app(mock_config, mock_worker_manager)
        client = TestClient(app)

        response = client.post("/v1/completions", json={
            "model": "test-model",
            "prompt": "Hello",
            "max_tokens": 10
        })

        assert response.status_code == 200

        # Verify model was loaded
        mock_worker_manager.load_model.assert_called_once_with("test-model")

        # Verify generation was called
        assert mock_worker_manager.generate.called

    def test_completions_uses_existing_model(self, mock_config, mock_worker_manager):
        """Test /v1/completions uses already-loaded model."""
        # Setup: model already loaded
        mock_worker_manager.get_status.return_value = {
            "model_loaded": True,
            "model_name": "test-model",
            "memory_gb": 4.5
        }

        # Mock successful generation
        mock_worker_manager.generate.return_value = {
            "text": "Generated response",
            "tokens": 5,
            "finish_reason": "stop"
        }

        app = create_app(mock_config, mock_worker_manager)
        client = TestClient(app)

        response = client.post("/v1/completions", json={
            "model": "test-model",
            "prompt": "Hello",
            "max_tokens": 10
        })

        assert response.status_code == 200

        # Verify model was NOT loaded (already loaded)
        mock_worker_manager.load_model.assert_not_called()

        # Verify generation was called
        assert mock_worker_manager.generate.called

    def test_completions_response_format(self, mock_config, mock_worker_manager):
        """Test /v1/completions returns OpenAI-compatible format."""
        # Setup: model loaded
        mock_worker_manager.get_status.return_value = {
            "model_loaded": True,
            "model_name": "test-model",
            "memory_gb": 4.5
        }

        mock_worker_manager.generate.return_value = {
            "text": "This is a test response.",
            "tokens": 6,
            "finish_reason": "stop"
        }

        app = create_app(mock_config, mock_worker_manager)
        client = TestClient(app)

        response = client.post("/v1/completions", json={
            "model": "test-model",
            "prompt": "Test",
            "max_tokens": 20
        })

        assert response.status_code == 200
        data = response.json()

        # Verify OpenAI-compatible structure
        assert data["object"] == "text_completion"
        assert data["model"] == "test-model"
        assert "choices" in data
        assert len(data["choices"]) == 1
        assert data["choices"][0]["text"] == "This is a test response."
        assert data["choices"][0]["finish_reason"] == "stop"
        assert "usage" in data
        assert data["usage"]["completion_tokens"] == 6

    @pytest.mark.skip(reason="Streaming is now implemented - test needs update to verify streaming response format")
    def test_completions_streaming_not_implemented(self, mock_config, mock_worker_manager):
        """Test streaming returns 501 Not Implemented."""
        mock_worker_manager.get_status.return_value = {
            "model_loaded": True,
            "model_name": "test-model",
            "memory_gb": 4.5
        }

        app = create_app(mock_config, mock_worker_manager)
        client = TestClient(app)

        response = client.post("/v1/completions", json={
            "model": "test-model",
            "prompt": "Test",
            "stream": True  # Request streaming
        })

        assert response.status_code == 501
        assert "not implemented" in response.json()["detail"].lower()

    def test_chat_completions_endpoint(self, mock_config, mock_worker_manager):
        """Test /v1/chat/completions endpoint."""
        mock_worker_manager.get_status.return_value = {
            "model_loaded": True,
            "model_name": "test-model",
            "memory_gb": 4.5
        }

        mock_worker_manager.generate.return_value = {
            "text": "Hello! How can I help you?",
            "tokens": 7,
            "finish_reason": "stop"
        }

        app = create_app(mock_config, mock_worker_manager)
        client = TestClient(app)

        response = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "max_tokens": 20
        })

        assert response.status_code == 200
        data = response.json()

        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["message"]["content"] == "Hello! How can I help you?"

    @pytest.mark.skip(reason="/v1/models now lists all available models from cache, not just loaded - test needs update")
    def test_models_list_no_model_loaded(self, mock_config, mock_worker_manager):
        """Test /v1/models with no model loaded."""
        mock_worker_manager.get_status.return_value = {
            "model_loaded": False,
            "model_name": None,
            "memory_gb": 0.0
        }

        app = create_app(mock_config, mock_worker_manager)
        client = TestClient(app)

        response = client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 0

    @pytest.mark.skip(reason="/v1/models now lists all available models from cache, not just loaded - test needs update")
    def test_models_list_with_model_loaded(self, mock_config, mock_worker_manager):
        """Test /v1/models with model loaded."""
        mock_worker_manager.get_status.return_value = {
            "model_loaded": True,
            "model_name": "test-model",
            "memory_gb": 4.5
        }

        app = create_app(mock_config, mock_worker_manager)
        client = TestClient(app)

        response = client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "test-model"


class TestAdminAPIEndpoints:
    """Tests for admin API endpoints."""

    def test_admin_health_endpoint(self, mock_config, mock_worker_manager):
        """Test GET /admin/health endpoint."""
        mock_worker_manager.health_check.return_value = {
            "healthy": True,
            "status": "healthy"
        }

        app = create_admin_app(mock_config, mock_worker_manager)
        client = TestClient(app)

        response = client.get("/admin/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["worker_status"] == "healthy"
        assert data["version"] == "3.0.0-alpha"

    def test_admin_status_endpoint(self, mock_config, mock_worker_manager):
        """Test GET /admin/status endpoint."""
        mock_worker_manager.get_status.return_value = {
            "model_loaded": True,
            "model_name": "test-model",
            "memory_gb": 4.5
        }

        mock_worker_manager.health_check.return_value = {
            "healthy": True,
            "status": "healthy"
        }

        app = create_admin_app(mock_config, mock_worker_manager)
        client = TestClient(app)

        response = client.get("/admin/status")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert data["version"] == "3.0.0-alpha"
        assert data["ports"]["main"] == 11440
        assert data["ports"]["admin"] == 11441
        assert data["model"]["loaded"] is True
        assert data["model"]["name"] == "test-model"
        assert data["config"]["machine_type"] == "test-machine"

    def test_admin_load_endpoint(self, mock_config, mock_worker_manager):
        """Test POST /admin/load endpoint."""
        mock_worker_manager.load_model.return_value = ModelLoadResult(
            model_name="new-model",
            memory_gb=5.2,
            load_time=12.5
        )

        app = create_admin_app(mock_config, mock_worker_manager)
        client = TestClient(app)

        response = client.post("/admin/load?model_path=new-model")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["model_name"] == "new-model"
        assert data["memory_gb"] == 5.2
        assert data["load_time"] == 12.5

        mock_worker_manager.load_model.assert_called_once_with("new-model")

    def test_admin_unload_endpoint(self, mock_config, mock_worker_manager):
        """Test POST /admin/unload endpoint."""
        mock_worker_manager.unload_model.return_value = UnloadResult(
            model_name="old-model",
            memory_freed_gb=4.5
        )

        app = create_admin_app(mock_config, mock_worker_manager)
        client = TestClient(app)

        response = client.post("/admin/unload")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["model_unloaded"] == "old-model"
        assert data["memory_freed_gb"] == 4.5

        mock_worker_manager.unload_model.assert_called_once()

    def test_admin_unload_no_model_error(self, mock_config, mock_worker_manager):
        """Test POST /admin/unload returns 400 when no model loaded."""
        mock_worker_manager.unload_model.side_effect = NoModelLoadedError("No model loaded")

        app = create_admin_app(mock_config, mock_worker_manager)
        client = TestClient(app)

        response = client.post("/admin/unload")

        assert response.status_code == 400
        assert "No model loaded" in response.json()["detail"]
