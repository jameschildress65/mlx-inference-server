"""
Integration Tests for Model Lifecycle

Tests model loading, unloading, and idle timeout behavior.
"""

import pytest
import time


class TestModelLoading:
    """Test suite for on-demand model loading"""

    def test_server_starts_without_model(self, admin_client, clean_model):
        """Test server starts with no model loaded"""
        status = admin_client["get"]("status", timeout=5).json()

        assert status["memory"]["model_loaded"] is False
        assert status["memory"]["model_name"] is None

    def test_model_loads_on_first_request(
        self,
        main_client,
        admin_client,
        clean_model,
        skip_if_no_model
    ):
        """Test model loads automatically on first chat request"""
        # Verify no model loaded
        status_before = admin_client["get"]("status", timeout=5).json()
        assert status_before["memory"]["model_loaded"] is False

        # Make request
        response = main_client["chat_completion"](
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )

        # Should succeed
        assert response.status_code == 200

        # Verify model is loaded
        status_after = admin_client["get"]("status", timeout=5).json()
        assert status_after["memory"]["model_loaded"] is True
        assert status_after["memory"]["model_name"] is not None

    def test_model_load_increases_memory(
        self,
        main_client,
        admin_client,
        clean_model,
        skip_if_no_model
    ):
        """Test model loading increases active memory"""
        # Get baseline memory
        status_before = admin_client["get"]("status", timeout=5).json()
        memory_before = status_before["memory"]["active_memory_gb"]

        # Load model
        main_client["chat_completion"](
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=10
        )

        # Check memory increased
        status_after = admin_client["get"]("status", timeout=5).json()
        memory_after = status_after["memory"]["active_memory_gb"]

        # Should have increased significantly (at least 1GB for model)
        assert memory_after > memory_before + 0.5


class TestModelUnloading:
    """Test suite for manual model unloading"""

    def test_manual_unload_frees_memory(
        self,
        main_client,
        admin_client,
        clean_model,
        skip_if_no_model
    ):
        """Test manual unload via admin API frees memory"""
        # Load model
        main_client["chat_completion"](
            messages=[{"role": "user", "content": "Load model"}],
            max_tokens=10
        )

        # Get memory with model loaded
        status_loaded = admin_client["get"]("status", timeout=5).json()
        memory_loaded = status_loaded["memory"]["active_memory_gb"]
        assert status_loaded["memory"]["model_loaded"] is True

        # Unload
        unload_response = admin_client["post"]("unload", timeout=5)
        unload_data = unload_response.json()

        assert unload_data["status"] == "success"
        assert unload_data["memory_freed_gb"] > 0

        # Verify memory decreased
        status_unloaded = admin_client["get"]("status", timeout=5).json()
        memory_unloaded = status_unloaded["memory"]["active_memory_gb"]

        assert memory_unloaded < memory_loaded
        assert status_unloaded["memory"]["model_loaded"] is False

    def test_reload_after_unload(
        self,
        main_client,
        admin_client,
        clean_model,
        skip_if_no_model
    ):
        """Test model can be reloaded after manual unload"""
        # Load
        response1 = main_client["chat_completion"](
            messages=[{"role": "user", "content": "First load"}],
            max_tokens=10
        )
        assert response1.status_code == 200

        # Unload
        admin_client["post"]("unload", timeout=5)

        # Verify unloaded
        status = admin_client["get"]("status", timeout=5).json()
        assert status["memory"]["model_loaded"] is False

        # Reload (make another request)
        response2 = main_client["chat_completion"](
            messages=[{"role": "user", "content": "Reload"}],
            max_tokens=10
        )
        assert response2.status_code == 200

        # Verify reloaded
        status2 = admin_client["get"]("status", timeout=5).json()
        assert status2["memory"]["model_loaded"] is True


class TestIdleTimeout:
    """Test suite for automatic idle timeout unloading"""

    def test_time_until_unload_calculation(
        self,
        main_client,
        admin_client,
        clean_model,
        skip_if_no_model
    ):
        """Test time_until_unload is calculated correctly"""
        # Set short timeout
        admin_client["put"](
            "timeout",
            json={"timeout_seconds": 60},
            timeout=5
        )

        # Load model
        main_client["chat_completion"](
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=10
        )

        # Check time_until_unload immediately after request
        status = admin_client["get"]("status", timeout=5).json()

        assert "time_until_unload_seconds" in status
        time_until = status["time_until_unload_seconds"]

        # Should be close to timeout value (60s)
        assert time_until is not None
        assert 55 <= time_until <= 61  # Allow some tolerance

        # Reset timeout for other tests
        admin_client["put"](
            "timeout",
            json={"timeout_seconds": 300},
            timeout=5
        )

    def test_idle_timeout_unloads_model(
        self,
        main_client,
        admin_client,
        server_instance,
        clean_model,
        skip_if_no_model
    ):
        """Test model auto-unloads after idle timeout"""
        # Set very short timeout for testing (10 seconds)
        admin_client["put"](
            "timeout",
            json={"timeout_seconds": 10},
            timeout=5
        )

        # Load model
        main_client["chat_completion"](
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=10
        )

        # Verify loaded
        status1 = admin_client["get"]("status", timeout=5).json()
        assert status1["memory"]["model_loaded"] is True

        # Wait for timeout + check interval (10s + 5s = 15s)
        time.sleep(16)

        # Verify unloaded
        status2 = admin_client["get"]("status", timeout=5).json()
        assert status2["memory"]["model_loaded"] is False

        # Reset timeout for other tests
        admin_client["put"](
            "timeout",
            json={"timeout_seconds": 300},
            timeout=5
        )

    def test_activity_prevents_unload(
        self,
        main_client,
        admin_client,
        skip_if_no_model
    ):
        """Test activity prevents idle timeout unload"""
        # Set short timeout
        admin_client["put"](
            "timeout",
            json={"timeout_seconds": 15},
            timeout=5
        )

        # Load model
        main_client["chat_completion"](
            messages=[{"role": "user", "content": "Initial"}],
            max_tokens=10
        )

        # Wait partway through timeout
        time.sleep(8)

        # Make another request (resets idle timer)
        main_client["chat_completion"](
            messages=[{"role": "user", "content": "Prevent timeout"}],
            max_tokens=10
        )

        # Wait a bit more (total would exceed original 15s)
        time.sleep(8)

        # Model should still be loaded (timer was reset)
        status = admin_client["get"]("status", timeout=5).json()
        assert status["memory"]["model_loaded"] is True

        # Reset timeout
        admin_client["put"](
            "timeout",
            json={"timeout_seconds": 300},
            timeout=5
        )


class TestModelUptime:
    """Test suite for model uptime tracking"""

    def test_uptime_starts_on_load(
        self,
        main_client,
        admin_client,
        clean_model,
        skip_if_no_model
    ):
        """Test uptime tracking starts when model loads"""
        # No model loaded - uptime should be None
        status_before = admin_client["get"]("status", timeout=5).json()
        assert status_before["memory"]["uptime_seconds"] is None

        # Load model
        main_client["chat_completion"](
            messages=[{"role": "user", "content": "Load"}],
            max_tokens=10
        )

        # Uptime should now be tracked
        status_after = admin_client["get"]("status", timeout=5).json()
        assert status_after["memory"]["uptime_seconds"] is not None
        assert status_after["memory"]["uptime_seconds"] >= 0

    def test_uptime_increases_over_time(
        self,
        main_client,
        admin_client,
        clean_model,
        skip_if_no_model
    ):
        """Test model uptime increases while loaded"""
        # Load model
        main_client["chat_completion"](
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=10
        )

        # Get initial uptime
        status1 = admin_client["get"]("status", timeout=5).json()
        uptime1 = status1["memory"]["uptime_seconds"]

        # Wait
        time.sleep(2)

        # Get updated uptime
        status2 = admin_client["get"]("status", timeout=5).json()
        uptime2 = status2["memory"]["uptime_seconds"]

        # Should have increased
        assert uptime2 > uptime1
        assert uptime2 >= uptime1 + 1.5  # At least 1.5s increase

    def test_uptime_resets_on_reload(
        self,
        main_client,
        admin_client,
        clean_model,
        skip_if_no_model
    ):
        """Test uptime resets when model is reloaded"""
        # Load model
        main_client["chat_completion"](
            messages=[{"role": "user", "content": "First"}],
            max_tokens=10
        )

        # Wait to build uptime
        time.sleep(2)

        # Get uptime
        status1 = admin_client["get"]("status", timeout=5).json()
        uptime1 = status1["memory"]["uptime_seconds"]
        assert uptime1 >= 1

        # Unload
        admin_client["post"]("unload", timeout=5)

        # Reload
        main_client["chat_completion"](
            messages=[{"role": "user", "content": "Reload"}],
            max_tokens=10
        )

        # Get new uptime
        status2 = admin_client["get"]("status", timeout=5).json()
        uptime2 = status2["memory"]["uptime_seconds"]

        # Should be reset (less than original)
        assert uptime2 < uptime1
        assert uptime2 < 2  # Should be fresh
