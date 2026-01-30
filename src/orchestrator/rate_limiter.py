"""Rate limiting for MLX Server.

Simple sliding window rate limiter for home lab use.
Prevents accidental self-DoS from batch scripts.

P1 Implementation Notes:
- Global counter (not per-IP) - single user home lab
- Thread-safe with lock
- Optional - disabled by default for home lab
- Returns HTTP 429 with Retry-After header when exceeded
"""

import time
import threading
import logging
from typing import Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    requests_per_minute: int = 60
    burst_size: int = 10
    enabled: bool = False  # Disabled by default for home lab


class RateLimiter:
    """
    Simple sliding window rate limiter.

    Thread-safe implementation using token bucket algorithm.
    Tokens refill at rate of (requests_per_minute / 60) per second.

    P1 Fix: Includes threading lock to prevent race conditions.
    """

    def __init__(self, config: RateLimitConfig):
        self.config = config

        # P1 Fix: Validate configuration to prevent ZeroDivisionError
        if config.enabled:
            if config.requests_per_minute <= 0:
                raise ValueError("requests_per_minute must be positive when rate limiting is enabled")
            if config.burst_size < 1:
                raise ValueError("burst_size must be at least 1 when rate limiting is enabled")

        self.rate_per_second = config.requests_per_minute / 60.0 if config.requests_per_minute > 0 else 1.0
        self.max_tokens = float(max(1, config.burst_size))
        self.tokens = float(max(1, config.burst_size))  # Start with full bucket
        self.last_update = time.time()
        self._lock = threading.Lock()

    def check(self) -> Tuple[bool, float]:
        """
        Check if request is allowed.

        Returns:
            Tuple of (allowed, retry_after_seconds)
            - allowed: True if request is permitted
            - retry_after: Seconds until next token available (0 if allowed)
        """
        if not self.config.enabled:
            return (True, 0.0)

        with self._lock:
            now = time.time()

            # Handle clock skew (system time went backward)
            elapsed = max(0.0, now - self.last_update)

            # Refill tokens based on elapsed time
            self.tokens = min(
                self.max_tokens,
                self.tokens + elapsed * self.rate_per_second
            )
            self.last_update = now

            # Check if we have a token to consume
            if self.tokens >= 1.0 - 1e-9:  # Float safety
                self.tokens -= 1.0
                return (True, 0.0)
            else:
                # Calculate time until next token
                tokens_needed = 1.0 - self.tokens
                retry_after = tokens_needed / self.rate_per_second
                logger.warning(
                    f"Rate limit exceeded - retry after {retry_after:.1f}s "
                    f"(tokens={self.tokens:.2f}, rate={self.config.requests_per_minute}/min)"
                )
                return (False, retry_after)

    def get_status(self) -> dict:
        """Get current rate limiter status for admin endpoint."""
        with self._lock:
            return {
                "enabled": self.config.enabled,
                "requests_per_minute": self.config.requests_per_minute,
                "burst_size": self.config.burst_size,
                "current_tokens": round(self.tokens, 2),
                "max_tokens": self.max_tokens
            }
