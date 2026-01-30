#!/usr/bin/env python3
"""
MLX Inference Server - Main Entry Point

Production-grade LLM inference server with process isolation for Apple Silicon.

Developed with Claude Code, leveraging Apple's MLX framework for optimized
inference on M-series chips. Features POSIX semaphores for cross-process
synchronization and atomic operations for crash safety.
"""

import sys
import os
import logging
import signal
import threading
import asyncio
from pathlib import Path

import uvicorn
import setproctitle

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config.server_config import ServerConfig
from src.orchestrator.worker_manager import WorkerManager
from src.orchestrator.api import create_app, create_admin_app
from src.orchestrator.idle_monitor import IdleMonitor
from src.orchestrator.shutdown_manager import ShutdownManager


def setup_logging(config: ServerConfig):
    """Configure logging."""
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "mlx_inference_server.log"),
            logging.StreamHandler()
        ]
    )


def run_server_thread(app, host: str, port: int, server_name: str):
    """Run a FastAPI server in a background thread (for admin API)."""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting {server_name} on {host}:{port}")

    # Use uvicorn.run() for background thread - signal handling not needed
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )


def run_main_server(app, host: str, port: int):
    """
    Run the main FastAPI server without uvicorn signal handler override.

    P2 Fix: uvicorn.run() overrides signal handlers, breaking graceful shutdown.
    Use programmatic server control to preserve our ShutdownManager handlers.
    """
    logger = logging.getLogger(__name__)

    # Create uvicorn config without installing signal handlers
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info"
    )
    server = uvicorn.Server(config)

    # Disable uvicorn's signal handler installation
    server.install_signal_handlers = lambda: None

    # Run the server (blocking)
    logger.info(f"Starting Main API on {host}:{port}")
    asyncio.run(server.serve())


def main():
    """Main entry point."""
    # Set process name for easy identification in ps/top
    setproctitle.setproctitle("mlx-inference-server")

    # Detect configuration
    config = ServerConfig.auto_detect()

    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)

    logger.info("=" * 70)
    logger.info("MLX Inference Server - Process Isolation Architecture")
    logger.info("=" * 70)

    # ============================================================
    # CRITICAL: Initialize registry FIRST (kills orphaned workers)
    # ============================================================
    # robust worker lifecycle management: This MUST happen before
    # any worker creation to ensure orphans from crashed servers are cleaned up.
    from src.orchestrator.process_registry import get_registry
    logger.info("Initializing ProcessRegistry (orphan cleanup)...")
    registry = get_registry()  # This triggers orphan cleanup on first call
    logger.info("ProcessRegistry ready - orphaned workers cleaned up")
    logger.info(f"Version: 1.0.0")
    logger.info(f"Machine: {config.machine_type}")
    logger.info(f"Total RAM: {config.total_ram_gb} GB")
    logger.info(f"Chip: {config.chip_model}")
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Main API Port: {config.main_port}")
    logger.info(f"Admin API Port: {config.admin_port}")
    logger.info(f"Cache Dir: {config.cache_dir}")
    logger.info(f"Idle Timeout: {config.idle_timeout_seconds}s")
    logger.info("=" * 70)

    # Initialize vision auto-resize limits (v3.1.0)
    from src.orchestrator.image_utils import detect_system_limits
    logger.info("Detecting vision processing limits...")
    detect_system_limits()
    logger.info("Vision processing initialized")

    # Create worker manager
    worker_manager = WorkerManager(config)

    # Start idle monitor
    idle_monitor = IdleMonitor(
        worker_manager=worker_manager,
        idle_timeout_seconds=config.idle_timeout_seconds,
        check_interval_seconds=30
    )
    idle_monitor.start()
    logger.info(f"Idle monitor started (timeout: {config.idle_timeout_seconds}s)")

    # P2: Setup graceful shutdown (replaces direct signal handlers)
    shutdown_manager = ShutdownManager(
        worker_manager=worker_manager,
        idle_monitor=idle_monitor,
        drain_timeout=config.graceful_shutdown_timeout
    )
    shutdown_manager.install_handlers()

    # Create FastAPI apps
    main_app = create_app(config, worker_manager)
    admin_app = create_admin_app(config, worker_manager)

    logger.info("Starting servers...")

    # Start admin server in background thread
    admin_thread = threading.Thread(
        target=run_server_thread,
        args=(admin_app, config.host, config.admin_port, "Admin API"),
        daemon=True
    )
    admin_thread.start()

    # Start main server in foreground (blocking)
    logger.info(f"Main API ready at http://{config.host}:{config.main_port}")
    logger.info(f"Admin API ready at http://{config.host}:{config.admin_port}")
    logger.info(f"Graceful shutdown timeout: {config.graceful_shutdown_timeout}s")
    logger.info("Server started successfully!")

    # P2: Use programmatic server to preserve ShutdownManager signal handlers
    # (uvicorn.run() would override our handlers)
    run_main_server(main_app, config.host, config.main_port)


if __name__ == "__main__":
    main()
