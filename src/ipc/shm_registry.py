"""Shared Memory Registry - Track active shared memory segments.

Provides thread-safe and process-safe tracking of active shared memory segments
for cleanup on startup (handling crashes) and coordination between processes.

Uses file-based locking (fcntl.flock) for synchronization.
"""

import os
import json
import fcntl
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class SharedMemoryRegistry:
    """
    Registry for tracking active shared memory segments.

    Thread-safe and process-safe using file-based locking.
    Stores segment metadata in JSON file for crash recovery.

    Registry location: ~/.mlx-server/active_shm.json

    Format:
    {
        "version": 1,
        "segments": [
            {
                "name": "mlx_a3f2c1d9e4b5a6c7...",
                "pid": 12345,
                "created_at": "2025-12-25T12:00:00Z",
                "size": 8388608
            }
        ]
    }
    """

    REGISTRY_VERSION = 1
    DEFAULT_DIR = Path.home() / ".mlx-server"
    DEFAULT_FILE = DEFAULT_DIR / "active_shm.json"

    def __init__(self, registry_path: Optional[Path] = None):
        """
        Initialize registry.

        Args:
            registry_path: Path to registry file (default: ~/.mlx-server/active_shm.json)
        """
        self.registry_path = registry_path or self.DEFAULT_FILE
        self._ensure_registry_exists()

    def _ensure_registry_exists(self):
        """Create registry directory and file if they don't exist."""
        # Create directory
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

        # Create file with initial structure if doesn't exist
        if not self.registry_path.exists():
            initial_data = {
                "version": self.REGISTRY_VERSION,
                "segments": []
            }
            with open(self.registry_path, 'w') as f:
                json.dump(initial_data, f, indent=2)
            logger.debug(f"Created registry file: {self.registry_path}")

    def register(self, name: str, pid: int, size: int) -> None:
        """
        Register a new shared memory segment.

        Thread-safe and process-safe using atomic file operations with fcntl locking.
        Uses low-level POSIX operations to guarantee atomicity of read-modify-write.

        Args:
            name: Shared memory segment name (e.g., "mlx_a3f2c1d9...")
            pid: Process ID that created the segment
            size: Size of segment in bytes

        Example:
            registry.register("mlx_a3f2c1d9...", os.getpid(), 8388608)
        """
        # Ensure registry file exists
        self._ensure_registry_exists()

        # Open file descriptor with low-level POSIX operations
        fd = os.open(str(self.registry_path), os.O_RDWR | os.O_CREAT, 0o600)
        try:
            # Acquire exclusive lock (blocks until available)
            fcntl.flock(fd, fcntl.LOCK_EX)

            # Read entire file atomically (lock held)
            content = os.read(fd, 1024 * 1024)  # 1MB max
            if content:
                try:
                    data = json.loads(content.decode('utf-8'))
                except json.JSONDecodeError as e:
                    logger.error(f"Corrupted registry file: {e}")
                    data = {
                        "version": self.REGISTRY_VERSION,
                        "segments": []
                    }
            else:
                # Empty file
                data = {
                    "version": self.REGISTRY_VERSION,
                    "segments": []
                }

            # Check if already registered (idempotent)
            for segment in data["segments"]:
                if segment["name"] == name:
                    logger.debug(f"Segment already registered: {name}")
                    return

            # Add new segment
            segment = {
                "name": name,
                "pid": pid,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "size": size
            }
            data["segments"].append(segment)

            # PRODUCTION: Atomic write using temp file + rename
            # This prevents corruption if crash happens during write
            json_bytes = json.dumps(data, indent=2).encode('utf-8')
            temp_path = f"{self.registry_path}.{os.getpid()}.tmp"

            try:
                # Write to temp file first (CRASH SAFE - original file intact)
                with open(temp_path, 'wb') as tmp:
                    tmp.write(json_bytes)
                    tmp.flush()
                    os.fsync(tmp.fileno())  # Ensure written to disk

                # Atomic rename (POSIX guarantees atomicity)
                # Either old file intact OR new file intact (no corruption window)
                os.rename(temp_path, self.registry_path)

                logger.debug(f"Registered segment: {name} (pid={pid}, size={size})")
            finally:
                # Cleanup temp file if rename failed
                try:
                    os.unlink(temp_path)
                except FileNotFoundError:
                    pass  # Already renamed successfully

        finally:
            # Release lock and close (lock released automatically on close, but explicit is safer)
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def unregister(self, name: str) -> bool:
        """
        Unregister a shared memory segment.

        Thread-safe and process-safe using atomic file operations with fcntl locking.
        Uses low-level POSIX operations to guarantee atomicity of read-modify-write.

        Args:
            name: Shared memory segment name

        Returns:
            True if segment was found and removed, False if not found

        Example:
            registry.unregister("mlx_a3f2c1d9...")
        """
        # Ensure registry file exists
        self._ensure_registry_exists()

        # Open file descriptor with low-level POSIX operations
        fd = os.open(str(self.registry_path), os.O_RDWR | os.O_CREAT, 0o600)
        try:
            # Acquire exclusive lock (blocks until available)
            fcntl.flock(fd, fcntl.LOCK_EX)

            # Read entire file atomically (lock held)
            content = os.read(fd, 1024 * 1024)  # 1MB max
            if content:
                try:
                    data = json.loads(content.decode('utf-8'))
                except json.JSONDecodeError as e:
                    logger.error(f"Corrupted registry file: {e}")
                    return False
            else:
                # Empty file
                return False

            # Find and remove segment
            original_count = len(data["segments"])
            data["segments"] = [
                seg for seg in data["segments"]
                if seg["name"] != name
            ]

            if len(data["segments"]) < original_count:
                # PRODUCTION: Atomic write using temp file + rename
                # This prevents corruption if crash happens during write
                json_bytes = json.dumps(data, indent=2).encode('utf-8')
                temp_path = f"{self.registry_path}.{os.getpid()}.tmp"

                try:
                    # Write to temp file first (CRASH SAFE - original file intact)
                    with open(temp_path, 'wb') as tmp:
                        tmp.write(json_bytes)
                        tmp.flush()
                        os.fsync(tmp.fileno())  # Ensure written to disk

                    # Atomic rename (POSIX guarantees atomicity)
                    # Either old file intact OR new file intact (no corruption window)
                    os.rename(temp_path, self.registry_path)

                    logger.debug(f"Unregistered segment: {name}")
                    return True
                finally:
                    # Cleanup temp file if rename failed
                    try:
                        os.unlink(temp_path)
                    except FileNotFoundError:
                        pass  # Already renamed successfully
            else:
                logger.debug(f"Segment not found in registry: {name}")
                return False

        finally:
            # Release lock and close
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def list_active(self) -> List[Dict]:
        """
        List all registered segments.

        Thread-safe read using atomic file operations with fcntl locking.

        Returns:
            List of segment metadata dictionaries

        Example:
            segments = registry.list_active()
            for seg in segments:
                print(f"{seg['name']}: {seg['pid']}")
        """
        # Ensure registry file exists
        self._ensure_registry_exists()

        # Open file descriptor for read-only with low-level POSIX operations
        fd = os.open(str(self.registry_path), os.O_RDONLY)
        try:
            # Acquire shared lock (multiple readers allowed)
            fcntl.flock(fd, fcntl.LOCK_SH)

            # Read entire file
            content = os.read(fd, 1024 * 1024)  # 1MB max
            if content:
                try:
                    data = json.loads(content.decode('utf-8'))
                    return data.get("segments", []).copy()
                except json.JSONDecodeError as e:
                    logger.error(f"Corrupted registry file: {e}")
                    return []
            else:
                return []

        finally:
            # Release lock and close
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def get_segment(self, name: str) -> Optional[Dict]:
        """
        Get metadata for a specific segment.

        Args:
            name: Shared memory segment name

        Returns:
            Segment metadata dict or None if not found
        """
        segments = self.list_active()
        for segment in segments:
            if segment["name"] == name:
                return segment
        return None

    def cleanup_stale(self) -> List[str]:
        """
        Remove entries for processes that no longer exist.

        Thread-safe and process-safe using atomic file operations with fcntl locking.
        Uses low-level POSIX operations to guarantee atomicity of read-modify-write.

        This does NOT clean up the shared memory itself, just the registry entries.
        Use this before attempting to clean up actual shared memory segments.

        Returns:
            List of segment names that were removed

        Example:
            stale = registry.cleanup_stale()
            print(f"Removed {len(stale)} stale registry entries")
        """
        removed = []

        # Ensure registry file exists
        self._ensure_registry_exists()

        # Open file descriptor with low-level POSIX operations
        fd = os.open(str(self.registry_path), os.O_RDWR | os.O_CREAT, 0o600)
        try:
            # Acquire exclusive lock (blocks until available)
            fcntl.flock(fd, fcntl.LOCK_EX)

            # Read entire file atomically (lock held)
            content = os.read(fd, 1024 * 1024)  # 1MB max
            if content:
                try:
                    data = json.loads(content.decode('utf-8'))
                except json.JSONDecodeError as e:
                    logger.error(f"Corrupted registry file: {e}")
                    return removed
            else:
                # Empty file
                return removed

            # Filter out segments for dead processes
            alive_segments = []
            for segment in data["segments"]:
                pid = segment["pid"]

                # Check if process exists
                if self._is_process_alive(pid):
                    alive_segments.append(segment)
                else:
                    removed.append(segment["name"])
                    logger.debug(
                        f"Stale registry entry (pid {pid} dead): {segment['name']}"
                    )

            if removed:
                data["segments"] = alive_segments

                # PRODUCTION: Atomic write using temp file + rename
                # This prevents corruption if crash happens during write
                json_bytes = json.dumps(data, indent=2).encode('utf-8')
                temp_path = f"{self.registry_path}.{os.getpid()}.tmp"

                try:
                    # Write to temp file first (CRASH SAFE - original file intact)
                    with open(temp_path, 'wb') as tmp:
                        tmp.write(json_bytes)
                        tmp.flush()
                        os.fsync(tmp.fileno())  # Ensure written to disk

                    # Atomic rename (POSIX guarantees atomicity)
                    # Either old file intact OR new file intact (no corruption window)
                    os.rename(temp_path, self.registry_path)

                    logger.info(f"Cleaned up {len(removed)} stale registry entries")
                finally:
                    # Cleanup temp file if rename failed
                    try:
                        os.unlink(temp_path)
                    except FileNotFoundError:
                        pass  # Already renamed successfully

        finally:
            # Release lock and close
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

        return removed

    @staticmethod
    def _is_process_alive(pid: int) -> bool:
        """
        Check if a process is alive.

        Args:
            pid: Process ID to check

        Returns:
            True if process exists, False otherwise

        Note:
            Uses os.kill(pid, 0) which works on Unix-like systems.
            Returns True for zombie processes (acceptable for our use case).
        """
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            # Process exists but we don't have permission to signal it
            return True

    def clear(self) -> int:
        """
        Clear all entries from registry.

        Thread-safe and process-safe using atomic file operations with fcntl locking.
        Uses low-level POSIX operations to guarantee atomicity of read-modify-write.

        WARNING: This does not clean up actual shared memory segments.
        Only use for testing or when you know segments are already cleaned up.

        Returns:
            Number of entries removed
        """
        # Ensure registry file exists
        self._ensure_registry_exists()

        # Open file descriptor with low-level POSIX operations
        fd = os.open(str(self.registry_path), os.O_RDWR | os.O_CREAT, 0o600)
        try:
            # Acquire exclusive lock (blocks until available)
            fcntl.flock(fd, fcntl.LOCK_EX)

            # Read entire file atomically (lock held)
            content = os.read(fd, 1024 * 1024)  # 1MB max
            if content:
                try:
                    data = json.loads(content.decode('utf-8'))
                except json.JSONDecodeError:
                    data = {"version": self.REGISTRY_VERSION, "segments": []}
            else:
                data = {"version": self.REGISTRY_VERSION, "segments": []}

            count = len(data["segments"])
            data["segments"] = []

            # PRODUCTION: Atomic write using temp file + rename
            # This prevents corruption if crash happens during write
            json_bytes = json.dumps(data, indent=2).encode('utf-8')
            temp_path = f"{self.registry_path}.{os.getpid()}.tmp"

            try:
                # Write to temp file first (CRASH SAFE - original file intact)
                with open(temp_path, 'wb') as tmp:
                    tmp.write(json_bytes)
                    tmp.flush()
                    os.fsync(tmp.fileno())  # Ensure written to disk

                # Atomic rename (POSIX guarantees atomicity)
                # Either old file intact OR new file intact (no corruption window)
                os.rename(temp_path, self.registry_path)

                logger.warning(f"Cleared {count} entries from registry")
                return count
            finally:
                # Cleanup temp file if rename failed
                try:
                    os.unlink(temp_path)
                except FileNotFoundError:
                    pass  # Already renamed successfully

        finally:
            # Release lock and close
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def __repr__(self) -> str:
        """String representation."""
        segments = self.list_active()
        return f"SharedMemoryRegistry(path={self.registry_path}, segments={len(segments)})"
