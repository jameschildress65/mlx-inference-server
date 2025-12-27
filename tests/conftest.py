"""
Pytest Configuration and Shared Fixtures

Provides common fixtures and mocks for all tests.
"""

import pytest
import argparse
from unittest.mock import MagicMock, Mock
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def mock_args():
    """
    Mock command-line arguments for MLX server.

    Returns:
        argparse.Namespace with default MLX server arguments
    """
    args = argparse.Namespace()
    args.model = None
    args.adapter_path = None
    args.draft_model = None
    args.trust_remote_code = False
    args.chat_template = None
    args.use_default_chat_template = False
    args.temp = 0.0
    args.top_p = 1.0
    args.top_k = 0  # Changed from -1 to avoid validation error
    args.min_p = 0.0
    args.max_tokens = 512
    args.num_draft_tokens = 3
    args.log_level = "INFO"
    return args


@pytest.fixture
def mock_mlx_metal(mocker):
    """
    Mock MLX Metal API calls.

    Updated for MLX 0.30.0+ which uses mx.get_*_memory() instead of
    mx.metal.get_*_memory() (deprecated but still functional).

    Returns:
        Dictionary of mocked functions
    """
    mocks = {
        # New API (MLX 0.30.0+)
        'get_active_memory': mocker.patch('mlx.core.get_active_memory', return_value=0),
        'get_cache_memory': mocker.patch('mlx.core.get_cache_memory', return_value=0),
        'get_peak_memory': mocker.patch('mlx.core.get_peak_memory', return_value=0),
        # Metal-specific functions (still used)
        'clear_cache': mocker.patch('mlx.core.metal.clear_cache'),
        'is_available': mocker.patch('mlx.core.metal.is_available', return_value=True),
        # Mock eval for memory cleanup tests
        'eval': mocker.patch('mlx.core.eval'),
    }
    return mocks


@pytest.fixture
def mock_model_provider(mocker):
    """
    Mock ExtendedModelProvider instance.

    Returns:
        MagicMock configured as ExtendedModelProvider
    """
    provider = MagicMock()
    provider.is_loaded.return_value = False
    provider.model = None
    provider.tokenizer = None
    provider.model_key = None
    provider.unload.return_value = {
        'memory_before_gb': 0.0,
        'memory_after_gb': 0.0,
        'memory_freed_gb': 0.0,
        'cache_cleared_gb': 0.0,
        'model_name': None
    }
    provider.get_memory_stats.return_value = {
        'active_memory_gb': 0.0,
        'cache_memory_gb': 0.0,
        'peak_memory_gb': 0.0,
        'model_loaded': False,
        'model_name': None,
        'uptime_seconds': None
    }
    return provider


@pytest.fixture
def mock_request_tracker():
    """
    Mock RequestTracker instance.

    Returns:
        MagicMock configured as RequestTracker
    """
    tracker = MagicMock()
    tracker.get_idle_time.return_value = 0
    tracker.get_last_activity.return_value = 0
    tracker.has_active_requests.return_value = False
    tracker.get_stats.return_value = {
        'total_requests': 0,
        'uptime_seconds': 0,
        'last_activity_seconds_ago': 0
    }
    return tracker


@pytest.fixture
def mock_idle_monitor():
    """
    Mock IdleMonitor instance.

    Returns:
        MagicMock configured as IdleMonitor
    """
    monitor = MagicMock()
    monitor.idle_timeout = 600
    monitor.check_interval = 30
    return monitor


@pytest.fixture
def temp_cache_dir(tmp_path):
    """
    Create temporary cache directory for testing.

    Returns:
        Path to temporary cache directory
    """
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def temp_log_dir(tmp_path):
    """
    Create temporary log directory for testing.

    Returns:
        Path to temporary log directory
    """
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    return log_dir


@pytest.fixture
def mock_t9_drive(mocker, tmp_path):
    """
    Mock presence of T9 drive for Studio detection.

    Args:
        mocker: pytest-mock fixture
        tmp_path: pytest tmp_path fixture

    Returns:
        Path that will be detected as T9
    """
    t9_path = tmp_path / "T9"
    t9_path.mkdir()

    def mock_exists(path):
        return str(path) == "/Volumes/T9" or str(path) == str(t9_path)

    mocker.patch('os.path.exists', side_effect=mock_exists)
    return t9_path


@pytest.fixture
def no_t9_drive(mocker):
    """
    Mock absence of T9 drive for Air detection.

    Args:
        mocker: pytest-mock fixture
    """
    mocker.patch('os.path.exists', return_value=False)


@pytest.fixture(autouse=True)
def reset_logging():
    """
    Reset logging configuration between tests.

    Prevents log handler conflicts between tests.
    """
    import logging

    # Remove all handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    yield

    # Clean up after test
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
