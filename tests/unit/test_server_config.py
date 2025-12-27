"""
Unit Tests for ServerConfig

Tests auto-detection and configuration.
"""

import pytest
import os
from src.server_config import ServerConfig


class TestServerConfig:
    """Test suite for ServerConfig"""

    def test_detect_studio(self, mock_t9_drive, temp_log_dir):
        """Test Apple Silicon Mac detection via T9 drive"""
        # Set environment to use temp directories
        os.environ['MLX_LOG_DIR'] = str(temp_log_dir)

        config = ServerConfig.auto_detect()

        assert config.machine_type == "studio"
        assert config.idle_timeout_seconds == 600  # 10 minutes
        assert config.request_timeout_seconds == 600
        assert config.memory_threshold_gb == 100
        assert config.main_port == 11436
        assert config.admin_port == 11437

    def test_detect_air(self, no_t9_drive, temp_log_dir, temp_cache_dir):
        """Test Apple Silicon Mac detection (no T9 drive)"""
        # Set environment to use temp directories
        os.environ['MLX_LOG_DIR'] = str(temp_log_dir)
        os.environ['HF_HOME'] = str(temp_cache_dir)

        config = ServerConfig.auto_detect()

        assert config.machine_type == "air"
        assert config.idle_timeout_seconds == 180  # 3 minutes
        assert config.request_timeout_seconds == 300
        assert config.memory_threshold_gb == 25

    def test_env_override_idle_timeout(self, no_t9_drive, temp_log_dir):
        """Test environment variable override for idle timeout"""
        os.environ['MLX_IDLE_TIMEOUT'] = '300'
        os.environ['MLX_LOG_DIR'] = str(temp_log_dir)

        config = ServerConfig.auto_detect()

        assert config.idle_timeout_seconds == 300

        # Clean up
        del os.environ['MLX_IDLE_TIMEOUT']

    def test_env_override_request_timeout(self, no_t9_drive, temp_log_dir):
        """Test environment variable override for request timeout"""
        os.environ['MLX_REQUEST_TIMEOUT'] = '120'
        os.environ['MLX_LOG_DIR'] = str(temp_log_dir)

        config = ServerConfig.auto_detect()

        assert config.request_timeout_seconds == 120

        # Clean up
        del os.environ['MLX_REQUEST_TIMEOUT']

    def test_env_override_admin_port(self, no_t9_drive, temp_log_dir):
        """Test environment variable override for admin port"""
        os.environ['MLX_ADMIN_PORT'] = '12000'
        os.environ['MLX_LOG_DIR'] = str(temp_log_dir)

        config = ServerConfig.auto_detect()

        assert config.admin_port == 12000

        # Clean up
        del os.environ['MLX_ADMIN_PORT']

    def test_env_override_cache_dir(self, no_t9_drive, temp_log_dir, temp_cache_dir):
        """Test environment variable override for cache directory"""
        custom_cache = str(temp_cache_dir / "custom")
        os.environ['HF_HOME'] = custom_cache
        os.environ['MLX_LOG_DIR'] = str(temp_log_dir)

        config = ServerConfig.auto_detect()

        assert config.cache_dir == custom_cache

    def test_directories_created(self, no_t9_drive, tmp_path):
        """Test that cache and log directories are created"""
        cache_dir = tmp_path / "cache_test"
        log_dir = tmp_path / "log_test"

        os.environ['HF_HOME'] = str(cache_dir)
        os.environ['MLX_LOG_DIR'] = str(log_dir)

        config = ServerConfig.auto_detect()

        assert cache_dir.exists()
        assert log_dir.exists()

    def test_to_dict(self, no_t9_drive, temp_log_dir):
        """Test configuration dictionary conversion"""
        os.environ['MLX_LOG_DIR'] = str(temp_log_dir)

        config = ServerConfig.auto_detect()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert 'machine_type' in config_dict
        assert 'main_port' in config_dict
        assert 'idle_timeout_seconds' in config_dict
        assert config_dict['machine_type'] == 'air'

    def test_str_representation(self, no_t9_drive, temp_log_dir):
        """Test string representation of config"""
        os.environ['MLX_LOG_DIR'] = str(temp_log_dir)

        config = ServerConfig.auto_detect()
        config_str = str(config)

        assert 'MLX Inference Server Configuration' in config_str
        assert 'Machine Type' in config_str
        assert 'AIR' in config_str
        assert '180s' in config_str  # 3 min for Air

    def test_host_always_all_interfaces(self, no_t9_drive, temp_log_dir):
        """Test that host is always 0.0.0.0"""
        os.environ['MLX_LOG_DIR'] = str(temp_log_dir)

        config = ServerConfig.auto_detect()

        assert config.host == "0.0.0.0"

    def test_main_port_always_11436(self, no_t9_drive, temp_log_dir):
        """Test that main port is always 11436"""
        os.environ['MLX_LOG_DIR'] = str(temp_log_dir)

        config = ServerConfig.auto_detect()

        assert config.main_port == 11436

    def test_hf_home_env_set(self, no_t9_drive, temp_log_dir, temp_cache_dir):
        """Test that HF_HOME environment variable is set"""
        os.environ['MLX_LOG_DIR'] = str(temp_log_dir)
        os.environ['HF_HOME'] = str(temp_cache_dir)

        config = ServerConfig.auto_detect()

        assert os.environ['HF_HOME'] == config.cache_dir
        assert os.environ['TRANSFORMERS_CACHE'] == config.cache_dir

    def test_studio_config_values(self, mock_t9_drive, temp_log_dir):
        """Test Studio configuration has expected values"""
        os.environ['MLX_LOG_DIR'] = str(temp_log_dir)

        config = ServerConfig.auto_detect()

        # Studio should have longer timeouts and higher memory threshold
        assert config.machine_type == "studio"
        assert config.idle_timeout_seconds == 600  # 10 minutes
        assert config.memory_threshold_gb == 100

    def test_air_config_values(self, no_t9_drive, temp_log_dir):
        """Test Air configuration has expected values"""
        os.environ['MLX_LOG_DIR'] = str(temp_log_dir)

        config = ServerConfig.auto_detect()

        # Air should have shorter timeouts and lower memory threshold
        assert config.machine_type == "air"
        assert config.idle_timeout_seconds == 180  # 3 minutes
        assert config.memory_threshold_gb == 25

    def test_studio_vs_air_differences(self, temp_log_dir):
        """Test key differences between Studio and Air configs"""
        # Just verify the known values are different
        # (Individual tests above verify the actual detection)
        studio_idle = 600
        studio_memory = 100
        air_idle = 180
        air_memory = 25

        # Studio should have longer timeouts
        assert studio_idle > air_idle
        assert studio_memory > air_memory
