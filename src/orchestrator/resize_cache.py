"""Image resize cache with TTL for vision processing.

Caches resized images to avoid re-processing the same image multiple times
during multi-turn conversations or repeated requests.
"""

import time
import hashlib
import logging
from collections import OrderedDict
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class TimedResizeCache:
    """LRU cache with TTL for resized images.

    Features:
    - Size-limited (max N entries)
    - Time-limited (TTL in seconds)
    - LRU eviction when full
    - Lazy cleanup on access (vision requests only)

    Thread-safe: Not required (single worker process, async orchestrator)
    """

    def __init__(self, max_size: int = 20, ttl_seconds: int = 3600):
        """Initialize cache.

        Args:
            max_size: Maximum number of cached images (default: 20)
            ttl_seconds: Time to live in seconds (default: 3600 = 1 hour)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = OrderedDict()  # {hash: (resized_bytes, timestamp)}
        self._hits = 0
        self._misses = 0

        logger.info(
            f"Resize cache initialized: max_size={max_size}, "
            f"ttl={ttl_seconds}s ({ttl_seconds/3600:.1f}h)"
        )

    def get(self, image_bytes: bytes) -> Optional[bytes]:
        """Get cached resized image if exists and not expired.

        Also triggers cleanup of all expired entries.

        Args:
            image_bytes: Original image bytes

        Returns:
            Resized image bytes if cached and valid, None otherwise
        """
        # Cleanup expired entries (only runs on vision requests)
        self._cleanup_expired()

        # Check for this image
        key = self._hash_image(image_bytes)

        if key in self.cache:
            resized, timestamp = self.cache[key]
            age = time.time() - timestamp

            # Check if still valid
            if age < self.ttl_seconds:
                # Mark as recently used (LRU)
                self.cache.move_to_end(key)
                self._hits += 1

                logger.info(
                    f"Resize cache HIT: {key[:8]}... "
                    f"(age: {int(age)}s, saved resize operation)"
                )
                return resized
            else:
                # Expired - will be cleaned up, return miss
                logger.debug(f"Resize cache EXPIRED: {key[:8]}... (age: {int(age)}s)")

        self._misses += 1
        return None

    def put(self, image_bytes: bytes, resized_bytes: bytes) -> None:
        """Store resized image in cache.

        Args:
            image_bytes: Original image bytes
            resized_bytes: Resized image bytes
        """
        key = self._hash_image(image_bytes)

        # Add to cache with timestamp
        self.cache[key] = (resized_bytes, time.time())
        self.cache.move_to_end(key)

        # Evict oldest if over size limit (LRU)
        if len(self.cache) > self.max_size:
            oldest_key, (oldest_data, oldest_ts) = self.cache.popitem(last=False)
            age = time.time() - oldest_ts
            logger.debug(
                f"Resize cache EVICTED (size limit): {oldest_key[:8]}... "
                f"(age: {int(age)}s)"
            )

        logger.debug(
            f"Resize cache PUT: {key[:8]}... "
            f"(cache size: {len(self.cache)}/{self.max_size})"
        )

    def _cleanup_expired(self) -> None:
        """Remove all expired entries from cache.

        Called on every get() to keep memory clean.
        Only runs during vision requests (not text requests).
        """
        now = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if now - timestamp >= self.ttl_seconds
        ]

        for key in expired_keys:
            del self.cache[key]

        if expired_keys:
            logger.info(
                f"Resize cache cleanup: removed {len(expired_keys)} expired entries "
                f"(cache size: {len(self.cache)}/{self.max_size})"
            )

    def _hash_image(self, image_bytes: bytes) -> str:
        """Generate hash of image for cache key.

        Args:
            image_bytes: Image bytes to hash

        Returns:
            Hex digest of MD5 hash
        """
        return hashlib.md5(image_bytes).hexdigest()

    def stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0

        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_percent": round(hit_rate, 1)
        }

    def clear(self) -> None:
        """Clear all cached entries."""
        count = len(self.cache)
        self.cache.clear()
        logger.info(f"Resize cache cleared: {count} entries removed")
