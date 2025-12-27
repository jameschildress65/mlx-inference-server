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
from pathlib import Path

import uvicorn
import setproctitle

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config.server_config import ServerConfig
from src.orchestrator.worker_manager import WorkerManager
from src.orchestrator.api import create_app, create_admin_app
from src.orchestrator.idle_monitor import IdleMonitor


# Global references for signal handling
worker_manager_global = None
idle_monitor_global = None
main_server_thread = None
admin_server_thread = None


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


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger = logging.getLogger(__name__)
    logger.info(f"Received signal {signum}, shutting down gracefully...")

    if idle_monitor_global:
        logger.info("Stopping idle monitor...")
        idle_monitor_global.stop()

    if worker_manager_global:
        logger.info("Shutting down worker manager...")
        worker_manager_global.shutdown()

    logger.info("Shutdown complete")
    sys.exit(0)


def run_server(app, host: str, port: int, server_name: str):
    """Run a FastAPI server in a thread."""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting {server_name} on {host}:{port}")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )


def main():
    """Main entry point."""
    global worker_manager_global, idle_monitor_global, main_server_thread, admin_server_thread

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

    # Create worker manager
    worker_manager = WorkerManager(config)
    worker_manager_global = worker_manager

    # Start idle monitor
    idle_monitor = IdleMonitor(
        worker_manager=worker_manager,
        idle_timeout_seconds=config.idle_timeout_seconds,
        check_interval_seconds=30
    )
    idle_monitor.start()
    idle_monitor_global = idle_monitor
    logger.info(f"Idle monitor started (timeout: {config.idle_timeout_seconds}s)")

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Create FastAPI apps
    main_app = create_app(config, worker_manager)
    admin_app = create_admin_app(config, worker_manager)

    logger.info("Starting servers...")

    # Start admin server in background thread
    admin_thread = threading.Thread(
        target=run_server,
        args=(admin_app, config.host, config.admin_port, "Admin API"),
        daemon=True
    )
    admin_thread.start()
    admin_server_thread = admin_thread

    # Start main server in foreground (blocking)
    logger.info(f"Main API ready at http://{config.host}:{config.main_port}")
    logger.info(f"Admin API ready at http://{config.host}:{config.admin_port}")
    logger.info("Server started successfully!")

    try:
        run_server(main_app, config.host, config.main_port, "Main API")
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        signal_handler(signal.SIGINT, None)


if __name__ == "__main__":
    main()
