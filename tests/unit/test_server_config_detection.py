"""
Unit tests for Production-grade hardware detection in ServerConfig.

Tests all detection paths, profiles, fallbacks, and edge cases.
"""

import pytest
from unittest.mock import patch, Mock
import subprocess
import os
from src.server_config import ServerConfig


class TestHardwareDetection:
    """Test hardware detection methods."""

    def test_get_ram_gb_success(self):
        """Test successful RAM detection via sysctl."""
        with patch('subprocess.run') as mock_run:
            # Mock 128GB RAM
            mock_run.return_value = Mock(
                stdout="137438953472\n",  # 128 * 1024^3
                returncode=0
            )
            ram_gb = ServerConfig._get_ram_gb()
            assert ram_gb == 128

    def test_get_ram_gb_32gb(self):
        """Test RAM detection for 32GB machine."""
        with patch('subprocess.run') as mock_run:
            # Mock 32GB RAM
            mock_run.return_value = Mock(
                stdout="34359738368\n",  # 32 * 1024^3
                returncode=0
            )
            ram_gb = ServerConfig._get_ram_gb()
            assert ram_gb == 32

    def test_get_ram_gb_16gb(self):
        """Test RAM detection for 16GB machine."""
        with patch('subprocess.run') as mock_run:
            # Mock 16GB RAM
            mock_run.return_value = Mock(
                stdout="17179869184\n",  # 16 * 1024^3
                returncode=0
            )
            ram_gb = ServerConfig._get_ram_gb()
            assert ram_gb == 16

    def test_get_ram_gb_failure_fallback(self):
        """Test RAM detection falls back to 32GB on failure."""
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, 'sysctl')
            ram_gb = ServerConfig._get_ram_gb()
            assert ram_gb == 32  # Default fallback

    def test_get_chip_model_m4_max(self):
        """Test M-series chip detection."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(
                stdout="Apple M-series\n",
                returncode=0
            )
            chip = ServerConfig._get_chip_model()
            assert chip == "Apple M-series"

    def test_get_chip_model_m1(self):
        """Test M1 chip detection."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(
                stdout="Apple M1\n",
                returncode=0
            )
            chip = ServerConfig._get_chip_model()
            assert chip == "Apple M1"

    def test_get_chip_model_failure(self):
        """Test chip detection failure fallback."""
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, 'sysctl')
            chip = ServerConfig._get_chip_model()
            assert chip == "Unknown"

    def test_get_model_name_studio(self):
        """Test Apple Silicon Mac model name detection."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(
                stdout="Model Name: Apple Silicon Mac\nOther: stuff",
                returncode=0
            )
            model = ServerConfig._get_model_name()
            assert model == "Apple Silicon Mac"

    def test_get_model_name_air(self):
        """Test Apple Silicon Mac model name detection."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(
                stdout="Model Name: Apple Silicon Mac\nOther: stuff",
                returncode=0
            )
            model = ServerConfig._get_model_name()
            assert model == "Apple Silicon Mac"

    def test_get_model_name_mini(self):
        """Test Mac mini model name detection."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(
                stdout="Model Name: Mac mini\nOther: stuff",
                returncode=0
            )
            model = ServerConfig._get_model_name()
            assert model == "Mac mini"

    def test_get_model_name_timeout(self):
        """Test model name detection with timeout."""
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired('system_profiler', 5)
            model = ServerConfig._get_model_name()
            assert model == "Unknown Mac"

    def test_get_model_name_failure(self):
        """Test model name detection failure fallback."""
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, 'system_profiler')
            model = ServerConfig._get_model_name()
            assert model == "Unknown Mac"


class TestMachineProfileDetection:
    """Test machine profile selection logic."""

    def test_studio_128gb_profile(self):
        """Test Apple Silicon Mac with 128GB RAM and T9 drive."""
        with patch('os.path.exists', return_value=True):  # T9 exists
            machine_type, config = ServerConfig._detect_machine_profile(
                ram_gb=128,
                model_name="Apple Silicon Mac",
                chip_model="Apple M-series"
            )

        assert machine_type == "high-memory"
        assert config["idle_timeout_seconds"] == 600
        assert config["request_timeout_seconds"] == 600
        assert config["memory_threshold_gb"] == 100
        assert config["cache_dir"] == "~/.cache/huggingface"
        assert "Production" in config["description"]

    def test_air_32gb_profile(self):
        """Test Apple Silicon Mac with 32GB RAM."""
        with patch('os.path.exists', return_value=False):  # No T9
            machine_type, config = ServerConfig._detect_machine_profile(
                ram_gb=32,
                model_name="Apple Silicon Mac",
                chip_model="Apple M4"
            )

        assert machine_type == "medium-memory"
        assert config["idle_timeout_seconds"] == 180
        assert config["request_timeout_seconds"] == 300
        assert config["memory_threshold_gb"] == 25
        assert ".local/mlx-models" in config["cache_dir"]
        assert "Interactive" in config["description"]

    def test_mini_16gb_profile(self):
        """Test Mac mini with 16GB RAM."""
        with patch('os.path.exists', return_value=False):  # No T9
            machine_type, config = ServerConfig._detect_machine_profile(
                ram_gb=16,
                model_name="Mac mini",
                chip_model="Apple M4"
            )

        assert machine_type == "low-memory"
        assert config["idle_timeout_seconds"] == 120
        assert config["request_timeout_seconds"] == 180
        assert config["memory_threshold_gb"] == 12
        assert ".local/mlx-models" in config["cache_dir"]
        assert "Edge Case" in config["description"]

    def test_mini_m1_profile(self):
        """Test M1 Mac mini."""
        with patch('os.path.exists', return_value=False):  # No T9
            machine_type, config = ServerConfig._detect_machine_profile(
                ram_gb=16,
                model_name="Mac mini",
                chip_model="Apple M1"
            )

        assert machine_type == "low-memory"
        assert config["idle_timeout_seconds"] == 120
        assert config["request_timeout_seconds"] == 180
        assert config["memory_threshold_gb"] == 12  # max(12, 16-4)
        assert ".local/mlx-models" in config["cache_dir"]
        assert "Legacy" in config["description"]

    def test_mini_m1_32gb_profile(self):
        """Test M1 Mac mini with 32GB RAM."""
        with patch('os.path.exists', return_value=False):  # No T9
            machine_type, config = ServerConfig._detect_machine_profile(
                ram_gb=32,
                model_name="Mac mini",
                chip_model="Apple M1"
            )

        assert machine_type == "low-memory"
        assert config["memory_threshold_gb"] == 28  # max(12, 32-4)

    def test_unknown_high_ram_fallback(self):
        """Test unknown machine with high RAM (≥100GB)."""
        with patch('os.path.exists', return_value=False):
            machine_type, config = ServerConfig._detect_machine_profile(
                ram_gb=128,
                model_name="Unknown Mac",
                chip_model="Unknown"
            )

        assert machine_type == "high-memory"
        assert config["idle_timeout_seconds"] == 600
        assert config["memory_threshold_gb"] == 108  # 128 - 20

    def test_unknown_mid_ram_fallback(self):
        """Test unknown machine with mid RAM (28-99GB)."""
        with patch('os.path.exists', return_value=False):
            machine_type, config = ServerConfig._detect_machine_profile(
                ram_gb=64,
                model_name="Unknown Mac",
                chip_model="Unknown"
            )

        assert machine_type == "unknown-64gb"
        assert config["idle_timeout_seconds"] == 180
        assert config["memory_threshold_gb"] == 56  # max(20, 64-8)

    def test_unknown_low_ram_fallback(self):
        """Test unknown machine with low RAM (<28GB)."""
        with patch('os.path.exists', return_value=False):
            machine_type, config = ServerConfig._detect_machine_profile(
                ram_gb=8,
                model_name="Unknown Mac",
                chip_model="Unknown"
            )

        assert machine_type == "unknown-8gb"
        assert config["idle_timeout_seconds"] == 120
        assert config["memory_threshold_gb"] == 10  # max(10, 8-4)

    def test_studio_without_t9_fallback(self):
        """Test Apple Silicon Mac without T9 drive falls back to unknown profile."""
        with patch('os.path.exists', return_value=False):  # No T9
            machine_type, config = ServerConfig._detect_machine_profile(
                ram_gb=128,
                model_name="Apple Silicon Mac",
                chip_model="Apple M-series"
            )

        # Should match high-memory since Studio needs T9 for studio profile
        assert machine_type == "high-memory"
        assert ".local/mlx-models" in config["cache_dir"]


class TestAutoDetectIntegration:
    """Integration tests for auto_detect() method."""

    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('src.server_config.subprocess.run')
    def test_auto_detect_studio(self, mock_run, mock_exists, mock_makedirs):
        """Test full auto-detection for Apple Silicon Mac."""
        # Mock hardware detection
        def side_effect(cmd, **kwargs):
            if 'memsize' in str(cmd):
                return Mock(stdout="137438953472\n", returncode=0)
            elif 'brand_string' in str(cmd):
                return Mock(stdout="Apple M-series\n", returncode=0)
            elif 'system_profiler' in str(cmd):
                return Mock(stdout="Model Name: Apple Silicon Mac\n", returncode=0)
            return Mock(stdout="", returncode=0)

        mock_run.side_effect = side_effect
        mock_exists.return_value = True  # T9 exists

        config = ServerConfig.auto_detect()

        assert config.machine_type == "high-memory"
        assert config.total_ram_gb == 128
        assert config.chip_model == "Apple M-series"
        assert config.model_name == "Apple Silicon Mac"
        assert config.idle_timeout_seconds == 600
        assert config.cache_dir == "~/.cache/huggingface"

    @patch.dict(os.environ, {}, clear=True)  # Clear all env vars including HF_HOME
    @patch('src.server_config.os.makedirs')
    @patch('src.server_config.os.path.exists')
    @patch('src.server_config.subprocess.run')
    def test_auto_detect_air(self, mock_run, mock_exists, mock_makedirs):
        """Test full auto-detection for Apple Silicon Mac."""
        # Mock hardware detection
        def side_effect(cmd, **kwargs):
            if 'memsize' in str(cmd):
                return Mock(stdout="34359738368\n", returncode=0)  # 32GB
            elif 'brand_string' in str(cmd):
                return Mock(stdout="Apple M4\n", returncode=0)
            elif 'system_profiler' in str(cmd):
                return Mock(stdout="Model Name: Apple Silicon Mac\n", returncode=0)
            return Mock(stdout="", returncode=0)

        mock_run.side_effect = side_effect
        mock_exists.return_value = False  # No T9

        config = ServerConfig.auto_detect()

        assert config.machine_type == "medium-memory"
        assert config.total_ram_gb == 32
        assert config.idle_timeout_seconds == 180
        assert ".local/mlx-models" in config.cache_dir

    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('subprocess.run')
    @patch.dict(os.environ, {'MLX_MACHINE_TYPE': 'low-memory', 'MLX_IDLE_TIMEOUT': '90'})
    def test_auto_detect_with_overrides(self, mock_run, mock_exists, mock_makedirs):
        """Test auto-detection respects environment variable overrides."""
        # Mock hardware detection (Air)
        def side_effect(cmd, **kwargs):
            if cmd[0] == 'sysctl' and 'memsize' in cmd:
                return Mock(stdout="34359738368\n", returncode=0)
            elif cmd[0] == 'sysctl' and 'brand_string' in cmd:
                return Mock(stdout="Apple M4\n", returncode=0)
            elif cmd[0] == 'system_profiler':
                return Mock(stdout="Model Name: Apple Silicon Mac\n", returncode=0)

        mock_run.side_effect = side_effect
        mock_exists.return_value = False

        config = ServerConfig.auto_detect()

        # Should use override machine type and timeout
        assert config.machine_type == "low-memory"
        assert config.idle_timeout_seconds == 90

    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('subprocess.run')
    def test_auto_detect_graceful_degradation(self, mock_run, mock_exists, mock_makedirs):
        """Test auto-detection handles failures gracefully."""
        # All detections fail
        mock_run.side_effect = subprocess.CalledProcessError(1, 'cmd')
        mock_exists.return_value = False

        config = ServerConfig.auto_detect()

        # Should fall back to safe defaults (32GB fallback)
        assert config.total_ram_gb == 32
        assert config.chip_model == "Unknown"
        assert config.model_name == "Unknown Mac"
        assert config.machine_type == "unknown-32gb"
        assert config.idle_timeout_seconds > 0  # Has some timeout


class TestConfigToDict:
    """Test configuration serialization."""

    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('src.server_config.subprocess.run')
    def test_to_dict(self, mock_run, mock_exists, mock_makedirs):
        """Test configuration converts to dict correctly."""
        # Mock Air detection
        def side_effect(cmd, **kwargs):
            if 'memsize' in str(cmd):
                return Mock(stdout="34359738368\n", returncode=0)
            elif 'brand_string' in str(cmd):
                return Mock(stdout="Apple M4\n", returncode=0)
            elif 'system_profiler' in str(cmd):
                return Mock(stdout="Model Name: Apple Silicon Mac\n", returncode=0)
            return Mock(stdout="", returncode=0)

        mock_run.side_effect = side_effect
        mock_exists.return_value = False

        config = ServerConfig.auto_detect()
        config_dict = config.to_dict()

        assert config_dict["machine_type"] == "medium-memory"
        assert config_dict["total_ram_gb"] == 32
        assert config_dict["chip_model"] == "Apple M4"
        assert config_dict["model_name"] == "Apple Silicon Mac"
        assert config_dict["main_port"] == 11436
        assert config_dict["idle_timeout_seconds"] == 180
