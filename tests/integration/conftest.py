"""
V3 Integration Test Fixtures

Provides fixtures for testing V3 FastAPI architecture with actual server instances.
This replaces the V2 http.server-based fixtures with FastAPI + uvicorn.
"""

import pytest
import threading
import time
import os
import sys
from pathlib import Path
from typing import Dict, Any
import uvicorn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.server_config import ServerConfig
from src.orchestrator.worker_manager import WorkerManager
from src.orchestrator.api import create_app, create_admin_app
from src.orchestrator.idle_monitor import IdleMonitor


# Test model to use (very small for fast tests)
TEST_MODEL = os.environ.get("MLX_TEST_MODEL", "mlx-community/Qwen2.5-0.5B-Instruct-4bit")

# Skip model loading tests if set
SKIP_MODEL_TESTS = os.environ.get("MLX_SKIP_MODEL_TESTS", "0") == "1"


@pytest.fixture(scope="session")
def test_ports_v3():
    """
    Provide test ports for V3 main and admin APIs.

    Uses different ports than V2 tests and production to avoid conflicts.
    """
    return {
        "main_port": 11460,  # V2 uses 11438, production uses 11440
        "admin_port": 11461   # V2 uses 11439, production uses 11441
    }


@pytest.fixture(scope="session")
def test_dirs_v3(tmp_path_factory):
    """Create temporary directories for V3 testing."""
    base_dir = tmp_path_factory.mktemp("mlx_test_v3")

    log_dir = base_dir / "logs"
    cache_dir = base_dir / "cache"

    log_dir.mkdir()
    cache_dir.mkdir()

    return {
        "log_dir": str(log_dir),
        "cache_dir": str(cache_dir),
        "base_dir": str(base_dir)
    }


@pytest.fixture(scope="session")
def server_config_v3(test_ports_v3, test_dirs_v3):
    """Create ServerConfig for V3 tests."""
    # Use actual HF cache (not test cache) for model downloads
    # This prevents re-downloading models for every test run
    actual_cache = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))

    return ServerConfig(
        main_port=test_ports_v3["main_port"],
        admin_port=test_ports_v3["admin_port"],
        host="127.0.0.1",
        idle_timeout_seconds=300,  # 5 min for tests
        request_timeout_seconds=120,  # 2 min for tests
        model_load_timeout_seconds=300,  # Phase 2 field
        max_concurrent_requests=10,  # Phase 2.1 field
        memory_threshold_gb=28,
        cache_dir=actual_cache,  # Use real cache
        log_dir=test_dirs_v3["log_dir"],
        machine_type="test-machine",
        total_ram_gb=32,
        chip_model="Test Chip",
        model_name=TEST_MODEL,
        use_shared_memory=True,  # Phase 2 feature
        rate_limit_enabled=False,  # P1: Rate limiting disabled for tests
        rate_limit_rpm=60,
        rate_limit_burst=10,
        graceful_shutdown_timeout=60  # P2: Graceful shutdown timeout
    )


class UvicornServer:
    """Uvicorn server wrapper for testing."""

    def __init__(self, app, host: str, port: int):
        self.app = app
        self.host = host
        self.port = port
        self.server = None
        self.thread = None

    def start(self):
        """Start server in background thread."""
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="warning",  # Reduce noise
            access_log=False
        )
        self.server = uvicorn.Server(config)

        self.thread = threading.Thread(
            target=self.server.run,
            daemon=True,
            name=f"TestUvicorn-{self.port}"
        )
        self.thread.start()

        # Wait for server to be ready
        max_retries = 20
        for i in range(max_retries):
            try:
                import requests
                response = requests.get(f"http://{self.host}:{self.port}/health", timeout=1)
                if response.status_code == 200:
                    return
            except Exception:
                if i == max_retries - 1:
                    raise RuntimeError(f"Server on port {self.port} failed to start")
                time.sleep(0.5)

    def stop(self):
        """Stop server."""
        if self.server:
            self.server.should_exit = True
            if self.thread:
                self.thread.join(timeout=5)


@pytest.fixture(scope="session")
def v3_server_instance(server_config_v3, test_ports_v3):
    """
    Start actual V3 MLX server instance for integration testing.

    This fixture:
    - Creates WorkerManager (process isolation architecture)
    - Starts IdleMonitor
    - Starts FastAPI main and admin servers with uvicorn
    - Yields server info for tests
    - Cleans up on teardown

    Yields:
        dict with server info and components
    """
    # Create worker manager (V3 architecture)
    worker_manager = WorkerManager(server_config_v3)

    # Start idle monitor
    idle_monitor = IdleMonitor(
        worker_manager=worker_manager,
        idle_timeout_seconds=server_config_v3.idle_timeout_seconds,
        check_interval_seconds=30
    )
    idle_monitor.start()

    # Create FastAPI apps
    main_app = create_app(server_config_v3, worker_manager)
    admin_app = create_admin_app(server_config_v3, worker_manager)

    # Start servers
    main_server = UvicornServer(
        main_app,
        server_config_v3.host,
        test_ports_v3["main_port"]
    )
    admin_server = UvicornServer(
        admin_app,
        server_config_v3.host,
        test_ports_v3["admin_port"]
    )

    main_server.start()
    admin_server.start()

    # Yield server info
    server_info = {
        "main_port": test_ports_v3["main_port"],
        "admin_port": test_ports_v3["admin_port"],
        "main_url": f"http://127.0.0.1:{test_ports_v3['main_port']}",
        "admin_url": f"http://127.0.0.1:{test_ports_v3['admin_port']}/admin",
        "worker_manager": worker_manager,
        "idle_monitor": idle_monitor,
        "config": server_config_v3,
        "main_server": main_server,
        "admin_server": admin_server
    }

    yield server_info

    # Cleanup
    idle_monitor.stop()

    # Shutdown worker if running
    worker_manager.shutdown()

    # Stop servers
    main_server.stop()
    admin_server.stop()


@pytest.fixture
def v3_admin_client(v3_server_instance):
    """
    Provide HTTP client for V3 admin API.

    Returns:
        dict with convenience methods for admin API calls
    """
    import requests

    base_url = v3_server_instance["admin_url"]

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
def v3_main_client(v3_server_instance):
    """
    Provide HTTP client for V3 main API (chat completions).

    Returns:
        dict with convenience methods for main API calls
    """
    import requests

    base_url = v3_server_instance["main_url"]

    def chat_completion(messages, model=TEST_MODEL, **kwargs):
        """
        Make chat completion request.

        Args:
            messages: List of message dicts
            model: Model to use (default: TEST_MODEL)
            **kwargs: Additional parameters (stream, max_tokens, etc.)
        """
        payload = {
            "model": model,
            "messages": messages,
            **kwargs
        }
        return requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=120  # Long timeout for model loading + inference
        )

    def completion(prompt, model=TEST_MODEL, **kwargs):
        """
        Make completion request.

        Args:
            prompt: Text prompt
            model: Model to use (default: TEST_MODEL)
            **kwargs: Additional parameters (stream, max_tokens, etc.)
        """
        payload = {
            "model": model,
            "prompt": prompt,
            **kwargs
        }
        return requests.post(
            f"{base_url}/v1/completions",
            json=payload,
            timeout=120
        )

    return {
        "chat_completion": chat_completion,
        "completion": completion,
        "base_url": base_url
    }


@pytest.fixture
def v3_clean_model(v3_server_instance):
    """
    Ensure model is unloaded before test.

    Use this fixture when test needs to start with clean state.
    """
    worker_manager = v3_server_instance["worker_manager"]

    # Unload if loaded
    try:
        status = worker_manager.get_status()
        if status.get("model_loaded"):
            worker_manager.unload_model()
            time.sleep(1)  # Wait for unload to complete
    except Exception:
        pass  # No model loaded

    yield

    # Cleanup after test
    try:
        status = worker_manager.get_status()
        if status.get("model_loaded"):
            worker_manager.unload_model()
    except Exception:
        pass


@pytest.fixture
def skip_if_no_model_v3():
    """Skip test if model loading is disabled."""
    if SKIP_MODEL_TESTS:
        pytest.skip("Model loading tests disabled (MLX_SKIP_MODEL_TESTS=1)")
