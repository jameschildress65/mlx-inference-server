"""
Unit tests for P1 Rate Limiter.

Tests token bucket algorithm, thread safety, and configuration.
"""

import pytest
import time
import threading
from src.orchestrator.rate_limiter import RateLimiter, RateLimitConfig


class TestRateLimiterBasic:
    """Test basic rate limiter functionality."""

    def test_disabled_always_allows(self):
        """Disabled rate limiter always allows requests."""
        config = RateLimitConfig(enabled=False)
        limiter = RateLimiter(config)

        for _ in range(100):
            allowed, retry_after = limiter.check()
            assert allowed is True
            assert retry_after == 0.0

    def test_enabled_allows_burst(self):
        """Enabled limiter allows burst_size requests immediately."""
        config = RateLimitConfig(
            requests_per_minute=60,
            burst_size=5,
            enabled=True
        )
        limiter = RateLimiter(config)

        # Should allow exactly burst_size requests
        for i in range(5):
            allowed, retry_after = limiter.check()
            assert allowed is True, f"Request {i} should be allowed"
            assert retry_after == 0.0

        # Next request should be rate limited
        allowed, retry_after = limiter.check()
        assert allowed is False
        assert retry_after > 0

    def test_tokens_refill_over_time(self):
        """Tokens refill at the configured rate."""
        config = RateLimitConfig(
            requests_per_minute=60,  # 1 per second
            burst_size=1,
            enabled=True
        )
        limiter = RateLimiter(config)

        # Consume the only token
        allowed, _ = limiter.check()
        assert allowed is True

        # Immediately should be rate limited
        allowed, retry_after = limiter.check()
        assert allowed is False
        assert retry_after > 0
        assert retry_after <= 1.0  # Should be ~1 second

        # Wait for refill
        time.sleep(1.1)

        # Should be allowed again
        allowed, _ = limiter.check()
        assert allowed is True

    def test_retry_after_calculation(self):
        """Retry-After header value is calculated correctly."""
        config = RateLimitConfig(
            requests_per_minute=60,  # 1 per second
            burst_size=1,
            enabled=True
        )
        limiter = RateLimiter(config)

        # Consume token
        limiter.check()

        # Check retry_after
        allowed, retry_after = limiter.check()
        assert allowed is False
        # Should be close to 1 second (within margin for test execution)
        assert 0.9 <= retry_after <= 1.1


class TestRateLimiterStatus:
    """Test rate limiter status reporting."""

    def test_get_status_disabled(self):
        """Status shows disabled state."""
        config = RateLimitConfig(enabled=False)
        limiter = RateLimiter(config)
        status = limiter.get_status()

        assert status["enabled"] is False
        assert "requests_per_minute" in status
        assert "burst_size" in status
        assert "current_tokens" in status

    def test_get_status_enabled(self):
        """Status shows enabled state with token count."""
        config = RateLimitConfig(
            requests_per_minute=60,
            burst_size=10,
            enabled=True
        )
        limiter = RateLimiter(config)
        status = limiter.get_status()

        assert status["enabled"] is True
        assert status["requests_per_minute"] == 60
        assert status["burst_size"] == 10
        assert status["current_tokens"] == 10.0
        assert status["max_tokens"] == 10.0

    def test_status_reflects_consumption(self):
        """Status reflects token consumption."""
        config = RateLimitConfig(
            requests_per_minute=60,
            burst_size=10,
            enabled=True
        )
        limiter = RateLimiter(config)

        # Consume 3 tokens
        limiter.check()
        limiter.check()
        limiter.check()

        status = limiter.get_status()
        assert status["current_tokens"] == 7.0


class TestRateLimiterThreadSafety:
    """Test thread safety of rate limiter."""

    def test_concurrent_access(self):
        """Rate limiter handles concurrent access correctly."""
        config = RateLimitConfig(
            requests_per_minute=6000,  # 100 per second
            burst_size=100,
            enabled=True
        )
        limiter = RateLimiter(config)

        allowed_count = 0
        denied_count = 0
        lock = threading.Lock()

        def make_requests():
            nonlocal allowed_count, denied_count
            for _ in range(20):
                allowed, _ = limiter.check()
                with lock:
                    if allowed:
                        allowed_count += 1
                    else:
                        denied_count += 1

        # 10 threads, 20 requests each = 200 total
        threads = [threading.Thread(target=make_requests) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have allowed exactly burst_size (100) requests
        # Some may have been allowed due to time passing, so allow margin
        assert allowed_count >= 100
        assert allowed_count <= 110  # Small margin for time elapsed
        assert allowed_count + denied_count == 200


class TestRateLimitConfig:
    """Test rate limit configuration."""

    def test_default_values(self):
        """Default config has sensible defaults."""
        config = RateLimitConfig()
        assert config.requests_per_minute == 60
        assert config.burst_size == 10
        assert config.enabled is False  # Disabled by default for home lab

    def test_custom_values(self):
        """Custom config values are respected."""
        config = RateLimitConfig(
            requests_per_minute=120,
            burst_size=20,
            enabled=True
        )
        assert config.requests_per_minute == 120
        assert config.burst_size == 20
        assert config.enabled is True


class TestRateLimiterValidation:
    """Test rate limiter input validation (Opus review fixes)."""

    def test_zero_rpm_when_enabled_raises(self):
        """Zero RPM with enabled=True raises ValueError."""
        config = RateLimitConfig(
            requests_per_minute=0,
            burst_size=10,
            enabled=True
        )
        with pytest.raises(ValueError, match="requests_per_minute must be positive"):
            RateLimiter(config)

    def test_negative_rpm_when_enabled_raises(self):
        """Negative RPM with enabled=True raises ValueError."""
        config = RateLimitConfig(
            requests_per_minute=-10,
            burst_size=10,
            enabled=True
        )
        with pytest.raises(ValueError, match="requests_per_minute must be positive"):
            RateLimiter(config)

    def test_zero_burst_when_enabled_raises(self):
        """Zero burst_size with enabled=True raises ValueError."""
        config = RateLimitConfig(
            requests_per_minute=60,
            burst_size=0,
            enabled=True
        )
        with pytest.raises(ValueError, match="burst_size must be at least 1"):
            RateLimiter(config)

    def test_zero_rpm_when_disabled_ok(self):
        """Zero RPM is OK when rate limiting is disabled."""
        config = RateLimitConfig(
            requests_per_minute=0,
            burst_size=0,
            enabled=False
        )
        # Should not raise - disabled limiter doesn't validate
        limiter = RateLimiter(config)
        allowed, _ = limiter.check()
        assert allowed is True  # Disabled always allows
