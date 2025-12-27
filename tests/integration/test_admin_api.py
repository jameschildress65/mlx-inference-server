"""
Integration Tests for Admin API

Tests actual HTTP endpoints for server management.
"""

import pytest
import time


class TestAdminHealth:
    """Test suite for /admin/health endpoint"""

    def test_health_check_returns_200(self, admin_client):
        """Test health endpoint returns 200 OK"""
        response = admin_client["get"]("health", timeout=5)

        assert response.status_code == 200

    def test_health_check_json_format(self, admin_client):
        """Test health check returns proper JSON format"""
        response = admin_client["get"]("health", timeout=5)
        data = response.json()

        assert "status" in data
        assert "memory_gb" in data
        assert "model_loaded" in data

    def test_health_check_no_model_loaded(self, admin_client, clean_model):
        """Test health check when no model is loaded"""
        response = admin_client["get"]("health", timeout=5)
        data = response.json()

        assert data["status"] in ["healthy", "degraded"]
        assert data["model_loaded"] is False
        assert data["memory_gb"] >= 0


class TestAdminStatus:
    """Test suite for /admin/status endpoint"""

    def test_status_returns_200(self, admin_client):
        """Test status endpoint returns 200 OK"""
        response = admin_client["get"]("status", timeout=5)

        assert response.status_code == 200

    def test_status_json_structure(self, admin_client):
        """Test status endpoint returns complete structure"""
        response = admin_client["get"]("status", timeout=5)
        data = response.json()

        # Top level keys
        assert "status" in data
        assert "memory" in data
        assert "requests" in data
        assert "idle_timeout_seconds" in data

        # Memory stats
        assert "active_memory_gb" in data["memory"]
        assert "model_loaded" in data["memory"]

        # Request stats
        assert "total_requests" in data["requests"]
        assert "uptime_seconds" in data["requests"]
        assert "last_activity_seconds_ago" in data["requests"]

    def test_status_idle_timeout_value(self, admin_client):
        """Test status returns configured idle timeout"""
        response = admin_client["get"]("status", timeout=5)
        data = response.json()

        # Should be 300 seconds (5 min) from test config
        assert data["idle_timeout_seconds"] == 300

    def test_status_uptime_increases(self, admin_client):
        """Test uptime increases over time"""
        response1 = admin_client["get"]("status", timeout=5)
        data1 = response1.json()
        uptime1 = data1["requests"]["uptime_seconds"]

        time.sleep(1)

        response2 = admin_client["get"]("status", timeout=5)
        data2 = response2.json()
        uptime2 = data2["requests"]["uptime_seconds"]

        assert uptime2 > uptime1


class TestAdminUnload:
    """Test suite for /admin/unload endpoint"""

    def test_unload_no_model_returns_info(self, admin_client, clean_model):
        """Test unload when no model is loaded"""
        response = admin_client["post"]("unload", timeout=5)
        data = response.json()

        assert response.status_code == 200
        assert data["status"] == "info"
        assert "No model loaded" in data["message"]

    def test_unload_returns_json(self, admin_client):
        """Test unload returns proper JSON format"""
        response = admin_client["post"]("unload", timeout=5)

        assert response.status_code == 200
        assert response.headers["Content-Type"] == "application/json"

    def test_unload_clears_model_state(self, admin_client, clean_model, skip_if_no_model, main_client):
        """Test unload actually clears model from memory"""
        # Load a model first by making a request
        chat_response = main_client["chat_completion"]([
            {"role": "user", "content": "Hello"}
        ])

        # Verify model is loaded
        status_response = admin_client["get"]("status", timeout=5)
        status_data = status_response.json()
        assert status_data["memory"]["model_loaded"] is True

        # Unload
        unload_response = admin_client["post"]("unload", timeout=5)
        unload_data = unload_response.json()

        assert unload_response.status_code == 200
        assert unload_data["status"] == "success"
        assert "model_unloaded" in unload_data
        assert "memory_freed_gb" in unload_data

        # Verify model is no longer loaded
        status_response2 = admin_client["get"]("status", timeout=5)
        status_data2 = status_response2.json()
        assert status_data2["memory"]["model_loaded"] is False


class TestAdminTimeout:
    """Test suite for /admin/timeout endpoint (PUT)"""

    def test_timeout_update_requires_body(self, admin_client):
        """Test timeout update without body returns 400"""
        response = admin_client["put"]("timeout", timeout=5)

        assert response.status_code == 400
        data = response.json()
        assert "error" in data

    def test_timeout_update_requires_positive_value(self, admin_client):
        """Test timeout update rejects non-positive values"""
        response = admin_client["put"](
            "timeout",
            json={"timeout_seconds": 0},
            timeout=5
        )

        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "positive" in data["error"].lower()

    def test_timeout_update_success(self, admin_client):
        """Test successful timeout update"""
        new_timeout = 600  # 10 minutes

        response = admin_client["put"](
            "timeout",
            json={"timeout_seconds": new_timeout},
            timeout=5
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["idle_timeout_seconds"] == new_timeout

        # Verify in status endpoint
        status_response = admin_client["get"]("status", timeout=5)
        status_data = status_response.json()
        assert status_data["idle_timeout_seconds"] == new_timeout

        # Reset to original for other tests
        admin_client["put"](
            "timeout",
            json={"timeout_seconds": 300},
            timeout=5
        )

    def test_timeout_update_invalid_json(self, admin_client):
        """Test timeout update with invalid JSON"""
        response = admin_client["put"](
            "timeout",
            data="not json",
            headers={"Content-Length": "8"},
            timeout=5
        )

        assert response.status_code == 400


class TestAdminRestart:
    """Test suite for /admin/restart endpoint"""

    def test_restart_not_implemented(self, admin_client):
        """Test restart returns 501 Not Implemented"""
        response = admin_client["post"]("restart", timeout=5)

        assert response.status_code == 501
        data = response.json()
        assert data["status"] == "not_implemented"


class TestAdminCORS:
    """Test suite for CORS support"""

    def test_cors_preflight(self, admin_client):
        """Test OPTIONS request for CORS preflight"""
        import requests

        response = requests.options(
            f"{admin_client['base_url']}/status",
            headers={"Access-Control-Request-Method": "GET"},
            timeout=5
        )

        assert response.status_code == 204
        assert "Access-Control-Allow-Origin" in response.headers
        assert response.headers["Access-Control-Allow-Origin"] == "*"

    def test_cors_headers_in_response(self, admin_client):
        """Test CORS headers are present in responses"""
        response = admin_client["get"]("health", timeout=5)

        assert "Access-Control-Allow-Origin" in response.headers
        assert response.headers["Access-Control-Allow-Origin"] == "*"


class TestAdminNotFound:
    """Test suite for unknown endpoints"""

    def test_unknown_get_endpoint_404(self, admin_client):
        """Test GET to unknown endpoint returns 404"""
        response = admin_client["get"]("unknown", timeout=5)

        assert response.status_code == 404
        data = response.json()
        assert "error" in data

    def test_unknown_post_endpoint_404(self, admin_client):
        """Test POST to unknown endpoint returns 404"""
        response = admin_client["post"]("unknown", timeout=5)

        assert response.status_code == 404
        data = response.json()
        assert "error" in data
