"""Utilities for POSIX semaphore naming and management.

This module provides deterministic semaphore name derivation used by both
SharedMemoryBridge (creation) and ProcessRegistry (cleanup). The derivation
MUST be identical across all callers to ensure proper IPC resource management.
"""

import hashlib
from typing import Tuple


def derive_semaphore_names(shm_name: str) -> Tuple[str, str]:
    """
    Derive POSIX semaphore names from shared memory name.

    Uses SHA256 hash truncated to 16 hex characters (64 bits) for uniqueness.
    Returns names for request and response semaphores.

    CRITICAL CONTRACT:
    - This derivation MUST be deterministic and identical across all callers
      (SharedMemoryBridge, ProcessRegistry) to ensure proper cleanup of
      orphaned resources.
    - shm_name is typically from SecureSharedMemoryManager (format: mlx_<26_hex>)

    Name format constraints (macOS POSIX):
    - Maximum 31 characters including leading /
    - Format: /r<hash> and /s<hash> = 18 chars each (safe under limit)

    Args:
        shm_name: Shared memory segment name (must be non-empty string)

    Returns:
        Tuple of (request_sem_name, response_sem_name)
        e.g., ("/r1234567890abcdef", "/s1234567890abcdef")

    Raises:
        ValueError: If shm_name is None or empty
        TypeError: If shm_name is not a string

    Example:
        >>> req_sem, resp_sem = derive_semaphore_names("mlx_test123")
        >>> req_sem.startswith("/r")
        True
        >>> len(req_sem)
        18
    """
    if not isinstance(shm_name, str):
        raise TypeError(f"shm_name must be str, got {type(shm_name).__name__}")

    if not shm_name:
        raise ValueError("shm_name must be non-empty string")

    name_hash = hashlib.sha256(shm_name.encode()).hexdigest()[:16]
    return (f"/r{name_hash}", f"/s{name_hash}")
