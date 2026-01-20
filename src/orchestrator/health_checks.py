"""Health check utilities for MLX Server.

Provides component-level health checks for GPU, memory, disk, and worker processes.
Used by /health and /ready endpoints for monitoring and orchestration.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Check MLX Metal availability for GPU health checks
try:
    import mlx.core as mx
    _HAS_MLX_METAL = hasattr(mx, 'metal')
except ImportError:
    mx = None
    _HAS_MLX_METAL = False


def check_gpu_health() -> Dict[str, Any]:
    """Check GPU availability and status.

    Returns:
        Dict with 'available' (bool), 'backend' (str), and optional 'error'
    """
    try:
        if _HAS_MLX_METAL:
            return {
                "available": True,
                "backend": "mlx.metal"
            }
        else:
            return {
                "available": False,
                "backend": "cpu_fallback"
            }
    except Exception as e:
        logger.debug(f"GPU health check failed: {e}")
        return {
            "available": False,
            "error": str(e)
        }


def check_memory_health() -> Dict[str, Any]:
    """Check system memory usage.

    Returns:
        Dict with 'percent_used', 'healthy', 'available_gb', 'total_gb'
    """
    import psutil
    mem = psutil.virtual_memory()
    return {
        "percent_used": mem.percent,
        "healthy": mem.percent < 90,
        "available_gb": mem.available / (1024**3),
        "total_gb": mem.total / (1024**3)
    }


def check_disk_health() -> Dict[str, Any]:
    """Check disk usage.

    Returns:
        Dict with 'percent_used', 'healthy', 'available_gb', 'total_gb'
    """
    import psutil
    disk = psutil.disk_usage('/')
    return {
        "percent_used": disk.percent,
        "healthy": disk.percent < 90,
        "available_gb": disk.free / (1024**3),
        "total_gb": disk.total / (1024**3)
    }


def check_worker_health(worker_manager) -> Dict[str, Any]:
    """Check worker process health.

    Args:
        worker_manager: WorkerManager instance

    Returns:
        Dict with 'alive', 'status', 'model_loaded', 'model_name', and optional 'error'
    """
    try:
        health = worker_manager.health_check()
        status = worker_manager.get_status()
        return {
            "alive": health["healthy"],
            "status": health["status"],
            "model_loaded": status["model_loaded"],
            "model_name": status["model_name"] if status["model_loaded"] else None
        }
    except Exception as e:
        logger.error(f"Worker health check failed: {e}")
        return {
            "alive": False,
            "status": "error",
            "error": str(e)
        }
