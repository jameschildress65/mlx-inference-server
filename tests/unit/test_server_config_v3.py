"""Unit tests for V3 ServerConfig."""

import pytest
import os
from unittest.mock import patch, MagicMock
from src.config.server_config import ServerConfig


class TestServerConfigV3Ports:
    """Tests for V3 port configuration."""

    @patch.dict(os.environ, {}, clear=True)
    @patch('src.config.server_config.ServerConfig._get_ram_gb')
    @patch('src.config.server_config.ServerConfig._get_chip_model')
    @patch('src.config.server_config.ServerConfig._get_model_name')
    def test_default_main_port_v3(self, mock_model, mock_chip, mock_ram):
        """Test V3 uses port 11440 by default."""
        mock_ram.return_value = 32
        mock_chip.return_value = "Apple M4"
        mock_model.return_value = "Apple Silicon Mac"

        config = ServerConfig.auto_detect()

        assert config.main_port == 11440, "V3 should use port 11440 for main API"

    @patch.dict(os.environ, {}, clear=True)
    @patch('src.config.server_config.ServerConfig._get_ram_gb')
    @patch('src.config.server_config.ServerConfig._get_chip_model')
    @patch('src.config.server_config.ServerConfig._get_model_name')
    def test_default_admin_port_v3(self, mock_model, mock_chip, mock_ram):
        """Test V3 uses port 11441 by default."""
        mock_ram.return_value = 32
        mock_chip.return_value = "Apple M4"
        mock_model.return_value = "Apple Silicon Mac"

        config = ServerConfig.auto_detect()

        assert config.admin_port == 11441, "V3 should use port 11441 for admin API"

    @patch.dict(os.environ, {"MLX_ADMIN_PORT": "12000"}, clear=False)
    @patch('src.config.server_config.ServerConfig._get_ram_gb')
    @patch('src.config.server_config.ServerConfig._get_chip_model')
    @patch('src.config.server_config.ServerConfig._get_model_name')
    def test_admin_port_override(self, mock_model, mock_chip, mock_ram):
        """Test MLX_ADMIN_PORT environment variable override."""
        mock_ram.return_value = 32
        mock_chip.return_value = "Apple M4"
        mock_model.return_value = "Apple Silicon Mac"

        config = ServerConfig.auto_detect()

        assert config.admin_port == 12000, "Admin port should be overridable"

    @patch.dict(os.environ, {}, clear=True)
    @patch('src.config.server_config.ServerConfig._get_ram_gb')
    @patch('src.config.server_config.ServerConfig._get_chip_model')
    @patch('src.config.server_config.ServerConfig._get_model_name')
    def test_main_port_not_overridable(self, mock_model, mock_chip, mock_ram):
        """Test main port is fixed at 11440 (not overridable)."""
        mock_ram.return_value = 32
        mock_chip.return_value = "Apple M4"
        mock_model.return_value = "Apple Silicon Mac"

        config = ServerConfig.auto_detect()

        # Main port should always be 11440 for V3
        assert config.main_port == 11440


class TestServerConfigMachineDetection:
    """Tests for machine profile detection (ported from V2)."""

    @patch.dict(os.environ, {}, clear=True)
    @patch('src.config.server_config.ServerConfig._get_ram_gb')
    @patch('src.config.server_config.ServerConfig._get_chip_model')
    @patch('src.config.server_config.ServerConfig._get_model_name')
    @patch('src.config.server_config.os.path.exists')
    def test_studio_128gb_detection(self, mock_exists, mock_model, mock_chip, mock_ram):
        """Test Apple Silicon Mac 128GB detection."""
        mock_ram.return_value = 128
        mock_chip.return_value = "Apple M-series"
        mock_model.return_value = "Apple Silicon Mac"
        mock_exists.return_value = True  # ~/.cache/huggingface exists

        config = ServerConfig.auto_detect()

        assert config.machine_type == "high-memory"
        assert config.total_ram_gb == 128
        assert config.idle_timeout_seconds == 600  # 10 minutes for high-memory
        assert config.model_load_timeout_seconds == 300  # 5 minutes for high-memory
        assert config.cache_dir  # Just verify cache_dir is set

    @patch.dict(os.environ, {}, clear=True)
    @patch('src.config.server_config.ServerConfig._get_ram_gb')
    @patch('src.config.server_config.ServerConfig._get_chip_model')
    @patch('src.config.server_config.ServerConfig._get_model_name')
    @patch('src.config.server_config.os.path.exists')
    def test_air_32gb_detection(self, mock_exists, mock_model, mock_chip, mock_ram):
        """Test Apple Silicon Mac 32GB detection."""
        mock_ram.return_value = 32
        mock_chip.return_value = "Apple M4"
        mock_model.return_value = "Apple Silicon Mac"
        mock_exists.return_value = False  # No ~/.cache/huggingface

        config = ServerConfig.auto_detect()

        assert config.machine_type == "medium-memory"
        assert config.total_ram_gb == 32
        assert config.idle_timeout_seconds == 300  # 5 minutes for medium-memory
        assert config.model_load_timeout_seconds == 180  # 3 minutes for medium-memory

    @patch.dict(os.environ, {}, clear=True)
    @patch('src.config.server_config.ServerConfig._get_ram_gb')
    @patch('src.config.server_config.ServerConfig._get_chip_model')
    @patch('src.config.server_config.ServerConfig._get_model_name')
    def test_mini_16gb_detection(self, mock_model, mock_chip, mock_ram):
        """Test Apple Silicon Mac M4 16GB detection."""
        mock_ram.return_value = 16
        mock_chip.return_value = "Apple M4"
        mock_model.return_value = "Mac mini"

        config = ServerConfig.auto_detect()

        assert config.machine_type == "low-memory"
        assert config.total_ram_gb == 16
        assert config.idle_timeout_seconds == 180  # 3 minutes for low-memory
        assert config.model_load_timeout_seconds == 120  # 2 minutes for low-memory

    @patch.dict(os.environ, {}, clear=True)
    @patch('src.config.server_config.ServerConfig._get_ram_gb')
    @patch('src.config.server_config.ServerConfig._get_chip_model')
    @patch('src.config.server_config.ServerConfig._get_model_name')
    def test_mini_m1_detection(self, mock_model, mock_chip, mock_ram):
        """Test Apple Silicon Mac M1 detection (priority over generic low-memory)."""
        mock_ram.return_value = 16
        mock_chip.return_value = "Apple M1"
        mock_model.return_value = "Mac mini"

        config = ServerConfig.auto_detect()

        assert config.machine_type == "low-memory"
        assert config.total_ram_gb == 16


class TestServerConfigEnvironmentOverrides:
    """Tests for environment variable overrides."""

    @patch.dict(os.environ, {"MLX_IDLE_TIMEOUT": "999"}, clear=False)
    @patch('src.config.server_config.ServerConfig._get_ram_gb')
    @patch('src.config.server_config.ServerConfig._get_chip_model')
    @patch('src.config.server_config.ServerConfig._get_model_name')
    def test_idle_timeout_override(self, mock_model, mock_chip, mock_ram):
        """Test MLX_IDLE_TIMEOUT override."""
        mock_ram.return_value = 32
        mock_chip.return_value = "Apple M4"
        mock_model.return_value = "Apple Silicon Mac"

        config = ServerConfig.auto_detect()

        assert config.idle_timeout_seconds == 999

    @patch.dict(os.environ, {"MLX_MACHINE_TYPE": "custom-machine"}, clear=False)
    @patch('src.config.server_config.ServerConfig._get_ram_gb')
    @patch('src.config.server_config.ServerConfig._get_chip_model')
    @patch('src.config.server_config.ServerConfig._get_model_name')
    def test_machine_type_override(self, mock_model, mock_chip, mock_ram):
        """Test MLX_MACHINE_TYPE override."""
        mock_ram.return_value = 32
        mock_chip.return_value = "Apple M4"
        mock_model.return_value = "Apple Silicon Mac"

        config = ServerConfig.auto_detect()

        # Machine type should be overridable via env var
        assert config.machine_type == "custom-machine"

    @patch.dict(os.environ, {"HF_HOME": "/tmp/custom_cache"}, clear=False)
    @patch('src.config.server_config.ServerConfig._get_ram_gb')
    @patch('src.config.server_config.ServerConfig._get_chip_model')
    @patch('src.config.server_config.ServerConfig._get_model_name')
    def test_cache_dir_override(self, mock_model, mock_chip, mock_ram):
        """Test HF_HOME override for cache directory."""
        mock_ram.return_value = 32
        mock_chip.return_value = "Apple M4"
        mock_model.return_value = "Apple Silicon Mac"

        config = ServerConfig.auto_detect()

        assert config.cache_dir == "/tmp/custom_cache"


class TestServerConfigDataclass:
    """Tests for ServerConfig dataclass structure."""

    def test_config_has_required_fields(self):
        """Test ServerConfig has all required fields."""
        config = ServerConfig(
            main_port=11440,
            admin_port=11441,
            host="0.0.0.0",
            idle_timeout_seconds=180,
            request_timeout_seconds=300,
            model_load_timeout_seconds=120,
            max_concurrent_requests=10,
            memory_threshold_gb=28,
            cache_dir="/test/cache",
            log_dir="/test/logs",
            use_shared_memory=True,
            machine_type="test",
            total_ram_gb=32,
            chip_model="Test Chip",
            model_name="Test Model",
            rate_limit_enabled=False,
            rate_limit_rpm=60,
            rate_limit_burst=10,
            graceful_shutdown_timeout=60
        )

        assert config.main_port == 11440
        assert config.admin_port == 11441
        assert config.host == "0.0.0.0"
        assert config.idle_timeout_seconds == 180
        assert config.model_load_timeout_seconds == 120
        assert config.machine_type == "test"
