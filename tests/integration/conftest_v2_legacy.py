"""
Integration Test Fixtures

Provides fixtures for running actual server instance and making HTTP requests.
"""

import pytest
import threading
import time
import requests
import os
import tempfile
from pathlib import Path
import sys
import argparse
from http.server import ThreadingHTTPServer
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.server_config import ServerConfig
from src.extended_model_provider import ExtendedModelProvider
from src.request_tracker import RequestTracker
from src.idle_monitor import IdleMonitor
from src.admin_handler import AdminHandler
from mlx_lm.server import APIHandler
from mlx_server_extended import ExtendedAPIHandler


# Test model to use (very small for fast tests)
# Using mlx-community/quantized-gemma-2b-it which is ~2GB
TEST_MODEL = os.environ.get("MLX_TEST_MODEL", "mlx-community/quantized-gemma-2b-it")

# Skip model loading tests if set
SKIP_MODEL_TESTS = os.environ.get("MLX_SKIP_MODEL_TESTS", "0") == "1"


@pytest.fixture(scope="session")
def test_ports():
    """
    Provide test ports for main and admin APIs.

    Uses different ports than production to avoid conflicts.
    """
    return {
        "main_port": 11438,  # Production uses 11436
        "admin_port": 11439   # Production uses 11437
    }


@pytest.fixture(scope="session")
def test_dirs(tmp_path_factory):
    """Create temporary directories for testing."""
    base_dir = tmp_path_factory.mktemp("mlx_test")

    log_dir = base_dir / "logs"
    cache_dir = base_dir / "cache"

    log_dir.mkdir()
    cache_dir.mkdir()

    return {
        "log_dir": str(log_dir),
        "cache_dir": str(cache_dir)
    }


@pytest.fixture(scope="session")
def server_instance(test_ports, test_dirs):
    """
    Start actual MLX server instance for integration testing.

    Yields:
        dict with server info and shutdown function
    """
    # Set environment variables for test
    os.environ['MLX_LOG_DIR'] = test_dirs['log_dir']
    os.environ['HF_HOME'] = test_dirs['cache_dir']
    os.environ['MLX_IDLE_TIMEOUT'] = '300'  # 5 min for tests

    # Setup logging
    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise during tests
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create CLI args
    args = argparse.Namespace(
        model=None,  # No pre-load
        adapter_path=None,
        draft_model=None,
        trust_remote_code=False,
        temp=0.0,
        top_p=1.0,
        top_k=0,  # 0 = disabled (not -1, which is invalid)
        min_p=0.0,
        max_tokens=100,  # Short responses for tests
        num_draft_tokens=3,
        chat_template="",
        use_default_chat_template=False
    )

    # Create components
    model_provider = ExtendedModelProvider(args)
    request_tracker = RequestTracker()

    # Create idle monitor with short timeout
    idle_monitor = IdleMonitor(
        model_provider=model_provider,
        request_tracker=request_tracker,
        idle_timeout_seconds=300  # 5 min
    )
    idle_monitor.start()

    # Create servers
    main_server = ThreadingHTTPServer(
        ("127.0.0.1", test_ports["main_port"]),
        lambda *args, **kwargs: ExtendedAPIHandler(
            model_provider,
            request_tracker,
            *args,
            **kwargs
        )
    )

    admin_server = ThreadingHTTPServer(
        ("127.0.0.1", test_ports["admin_port"]),
        lambda *args, **kwargs: AdminHandler(
            model_provider,
            request_tracker,
            idle_monitor,
            *args,
            **kwargs
        )
    )

    # Start servers in threads
    main_thread = threading.Thread(
        target=main_server.serve_forever,
        daemon=True,
        name="TestMainServer"
    )
    admin_thread = threading.Thread(
        target=admin_server.serve_forever,
        daemon=True,
        name="TestAdminServer"
    )

    main_thread.start()
    admin_thread.start()

    # Wait for servers to be ready
    time.sleep(0.5)

    # Verify servers are running
    max_retries = 10
    for i in range(max_retries):
        try:
            response = requests.get(
                f"http://127.0.0.1:{test_ports['admin_port']}/admin/health",
                timeout=2
            )
            if response.status_code == 200:
                break
        except requests.exceptions.RequestException:
            if i == max_retries - 1:
                raise RuntimeError("Server failed to start")
            time.sleep(0.5)

    # Yield server info
    server_info = {
        "main_port": test_ports["main_port"],
        "admin_port": test_ports["admin_port"],
        "main_url": f"http://127.0.0.1:{test_ports['main_port']}",
        "admin_url": f"http://127.0.0.1:{test_ports['admin_port']}/admin",
        "model_provider": model_provider,
        "request_tracker": request_tracker,
        "idle_monitor": idle_monitor
    }

    yield server_info

    # Cleanup
    idle_monitor.stop()

    # Unload model if loaded
    if model_provider.is_loaded():
        model_provider.unload()

    # Shutdown servers
    main_server.shutdown()
    admin_server.shutdown()


@pytest.fixture
def admin_client(server_instance):
    """
    Provide HTTP client for admin API.

    Returns:
        dict with convenience methods for admin API calls
    """
    base_url = server_instance["admin_url"]

    def get(endpoint, **kwargs):
        """GET request to admin API."""
        return requests.get(f"{base_url}/{endpoint}", **kwargs)

    def post(endpoint, **kwargs):
        """POST request to admin API."""
        return requests.post(f"{base_url}/{endpoint}", **kwargs)

    def put(endpoint, **kwargs):
        """PUT request to admin API."""
        return requests.put(f"{base_url}/{endpoint}", **kwargs)

    return {
        "get": get,
        "post": post,
        "put": put,
        "base_url": base_url
    }


@pytest.fixture
def main_client(server_instance):
    """
    Provide HTTP client for main API (chat completions).

    Returns:
        dict with convenience methods for main API calls
    """
    base_url = server_instance["main_url"]

    def chat_completion(messages, model=TEST_MODEL, **kwargs):
        """
        Make chat completion request.

        Args:
            messages: List of message dicts
            model: Model to use (default: TEST_MODEL)
            **kwargs: Additional parameters
        """
        payload = {
            "model": model,
            "messages": messages,
            **kwargs
        }
        return requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=60  # Long timeout for model loading
        )

    return {
        "chat_completion": chat_completion,
        "base_url": base_url
    }


@pytest.fixture
def clean_model(server_instance):
    """
    Ensure model is unloaded before test.

    Use this fixture when test needs to start with clean state.
    """
    model_provider = server_instance["model_provider"]

    # Unload if loaded
    if model_provider.is_loaded():
        model_provider.unload()

    yield

    # Cleanup after test
    if model_provider.is_loaded():
        model_provider.unload()


@pytest.fixture
def skip_if_no_model():
    """Skip test if model loading is disabled."""
    if SKIP_MODEL_TESTS:
        pytest.skip("Model loading tests disabled (MLX_SKIP_MODEL_TESTS=1)")
