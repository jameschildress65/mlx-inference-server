"""Memory tracking utilities."""

import logging

logger = logging.getLogger(__name__)


def get_memory_usage_gb() -> float:
    """
    Get current MLX memory usage in GB.

    Returns:
        Memory usage in GB

    Note:
        Uses mlx.get_active_memory() to get Metal GPU memory.
    """
    try:
        import mlx.core as mx

        # Get active memory from Metal
        active_memory_bytes = mx.get_active_memory()
        memory_gb = active_memory_bytes / (1024**3)

        return round(memory_gb, 4)

    except ImportError:
        logger.warning("mlx not available, returning 0.0")
        return 0.0

    except Exception as e:
        logger.error(f"Failed to get memory usage: {e}")
        return 0.0
