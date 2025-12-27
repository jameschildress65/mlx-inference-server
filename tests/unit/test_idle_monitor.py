"""
Unit Tests for IdleMonitor

Tests idle timeout monitoring and auto-unload.
"""

import pytest
import time
import threading
from src.idle_monitor import IdleMonitor


class TestIdleMonitor:
    """Test suite for IdleMonitor"""

    def test_init(self, mock_model_provider, mock_request_tracker):
        """Test monitor initialization"""
        monitor = IdleMonitor(
            model_provider=mock_model_provider,
            request_tracker=mock_request_tracker,
            idle_timeout_seconds=60,
            check_interval_seconds=5
        )

        assert monitor.idle_timeout == 60
        assert monitor.check_interval == 5
        assert monitor.daemon is True
        assert monitor.name == "IdleMonitor"

    def test_auto_unload_on_timeout(self, mock_model_provider, mock_request_tracker):
        """Test auto-unload triggers after timeout"""
        # Configure mocks
        mock_request_tracker.get_idle_time.return_value = 70  # >60s idle
        mock_model_provider.is_loaded.return_value = True
        mock_model_provider.unload.return_value = {
            'model_name': 'test-model',
            'memory_freed_gb': 18.2
        }

        monitor = IdleMonitor(
            model_provider=mock_model_provider,
            request_tracker=mock_request_tracker,
            idle_timeout_seconds=60,
            check_interval_seconds=1  # Fast check for testing
        )

        # Start monitor
        monitor.start()
        time.sleep(2)  # Let it run one check
        monitor.stop()
        monitor.join(timeout=2)

        # Verify unload was called
        mock_model_provider.unload.assert_called()

    def test_no_unload_when_active(self, mock_model_provider, mock_request_tracker):
        """Test no unload when server is active"""
        # Configure mocks - idle time below threshold
        mock_request_tracker.get_idle_time.return_value = 30  # <60s idle
        mock_model_provider.is_loaded.return_value = True

        monitor = IdleMonitor(
            mock_model_provider,
            mock_request_tracker,
            idle_timeout_seconds=60,
            check_interval_seconds=1
        )

        monitor.start()
        time.sleep(2)
        monitor.stop()
        monitor.join(timeout=2)

        # Verify unload was NOT called
        mock_model_provider.unload.assert_not_called()

    def test_no_unload_when_no_model(self, mock_model_provider, mock_request_tracker):
        """Test no unload when no model loaded"""
        # Configure mocks - idle but no model
        mock_request_tracker.get_idle_time.return_value = 70
        mock_model_provider.is_loaded.return_value = False

        monitor = IdleMonitor(
            mock_model_provider,
            mock_request_tracker,
            idle_timeout_seconds=60,
            check_interval_seconds=1
        )

        monitor.start()
        time.sleep(2)
        monitor.stop()
        monitor.join(timeout=2)

        # Should check but not unload
        mock_model_provider.is_loaded.assert_called()
        mock_model_provider.unload.assert_not_called()

    def test_update_timeout(self, mock_model_provider, mock_request_tracker):
        """Test dynamic timeout update"""
        monitor = IdleMonitor(
            mock_model_provider,
            mock_request_tracker,
            idle_timeout_seconds=60,
            check_interval_seconds=1
        )

        assert monitor.idle_timeout == 60

        # Update timeout
        monitor.update_timeout(300)
        assert monitor.idle_timeout == 300

    def test_update_timeout_invalid(self, mock_model_provider, mock_request_tracker):
        """Test update_timeout rejects invalid values"""
        monitor = IdleMonitor(
            mock_model_provider,
            mock_request_tracker,
            idle_timeout_seconds=60,
            check_interval_seconds=1
        )

        # Should raise ValueError for non-positive timeout
        with pytest.raises(ValueError, match="Timeout must be positive"):
            monitor.update_timeout(0)

        with pytest.raises(ValueError):
            monitor.update_timeout(-100)

    def test_stop_gracefully(self, mock_model_provider, mock_request_tracker):
        """Test graceful stop"""
        monitor = IdleMonitor(
            mock_model_provider,
            mock_request_tracker,
            idle_timeout_seconds=60,
            check_interval_seconds=1
        )

        monitor.start()
        assert monitor.is_alive()

        monitor.stop()
        monitor.join(timeout=3)

        # Should have stopped
        assert not monitor.is_alive()

    def test_handles_unload_exception(self, mock_model_provider, mock_request_tracker, caplog):
        """Test monitor handles exceptions during unload gracefully"""
        # Configure mocks - unload raises exception
        mock_request_tracker.get_idle_time.return_value = 70
        mock_model_provider.is_loaded.return_value = True
        mock_model_provider.unload.side_effect = Exception("Unload failed")

        monitor = IdleMonitor(
            mock_model_provider,
            mock_request_tracker,
            idle_timeout_seconds=60,
            check_interval_seconds=1
        )

        monitor.start()
        time.sleep(2)
        monitor.stop()
        monitor.join(timeout=2)

        # Monitor should still be alive and functional despite exception
        # (It logs error but continues)

    def test_daemon_thread(self, mock_model_provider, mock_request_tracker):
        """Test monitor is daemon thread"""
        monitor = IdleMonitor(
            mock_model_provider,
            mock_request_tracker,
            idle_timeout_seconds=60,
            check_interval_seconds=1
        )

        # Should be daemon (won't prevent program exit)
        assert monitor.daemon is True

    def test_check_interval_respected(self, mock_model_provider, mock_request_tracker):
        """Test that check interval is respected"""
        check_count = []

        def count_check(*args):
            check_count.append(time.time())
            return 0  # Not idle

        mock_request_tracker.get_idle_time.side_effect = count_check

        monitor = IdleMonitor(
            mock_model_provider,
            mock_request_tracker,
            idle_timeout_seconds=60,
            check_interval_seconds=0.5  # 500ms
        )

        monitor.start()
        time.sleep(2.5)
        monitor.stop()
        monitor.join(timeout=2)

        # Should have checked ~5 times (2.5 / 0.5)
        # Allow some tolerance for timing
        assert len(check_count) >= 4
        assert len(check_count) <= 7

    def test_multiple_unload_cycles(self, mock_model_provider, mock_request_tracker):
        """Test multiple load/unload cycles"""
        # Alternate between idle and active
        idle_values = [70, 30, 70, 30, 70]  # Over threshold, under, over, under, over
        idle_index = [0]

        def get_idle_time():
            val = idle_values[idle_index[0] % len(idle_values)]
            idle_index[0] += 1
            return val

        mock_request_tracker.get_idle_time.side_effect = get_idle_time
        mock_model_provider.is_loaded.return_value = True
        mock_model_provider.unload.return_value = {'model_name': 'test', 'memory_freed_gb': 1.0}

        monitor = IdleMonitor(
            mock_model_provider,
            mock_request_tracker,
            idle_timeout_seconds=60,
            check_interval_seconds=0.2
        )

        monitor.start()
        time.sleep(2)
        monitor.stop()
        monitor.join(timeout=2)

        # Should have triggered unload multiple times
        assert mock_model_provider.unload.call_count >= 2

    def test_thread_name(self, mock_model_provider, mock_request_tracker):
        """Test thread has correct name"""
        monitor = IdleMonitor(
            mock_model_provider,
            mock_request_tracker,
            idle_timeout_seconds=60,
            check_interval_seconds=1
        )

        assert monitor.name == "IdleMonitor"

    def test_no_cpu_spin_with_active_requests(self, mock_model_provider, mock_request_tracker):
        """Test thread sleeps properly even when requests are active (CPU bug fix)"""
        # This tests the fix for the 100% CPU bug where has_active_requests()
        # returning True would cause continue without sleep, spinning the CPU
        check_count = []

        def count_check():
            check_count.append(time.time())
            return True  # Always active

        mock_request_tracker.has_active_requests.side_effect = count_check
        mock_model_provider.is_loaded.return_value = True

        monitor = IdleMonitor(
            mock_model_provider,
            mock_request_tracker,
            idle_timeout_seconds=60,
            check_interval_seconds=0.5  # 500ms
        )

        monitor.start()
        time.sleep(2.5)  # Run for 2.5 seconds
        monitor.stop()
        monitor.join(timeout=2)

        # Should have checked ~5 times (2.5 / 0.5 = 5)
        # NOT hundreds of times (which would indicate CPU spinning)
        # Allow tolerance for timing: 4-7 checks
        assert len(check_count) >= 4, f"Too few checks: {len(check_count)}"
        assert len(check_count) <= 7, f"Too many checks (CPU spin): {len(check_count)}"

        # Verify get_idle_time was NOT called (skipped because has_active=True)
        mock_request_tracker.get_idle_time.assert_not_called()

        # Verify unload was NOT called
        mock_model_provider.unload.assert_not_called()
