"""
Unit Tests for RequestTracker

Tests thread-safe request activity tracking.
"""

import pytest
import time
import threading
from src.request_tracker import RequestTracker


class TestRequestTracker:
    """Test suite for RequestTracker"""

    def test_init(self):
        """Test initialization"""
        tracker = RequestTracker()

        assert tracker._request_count == 0
        assert tracker.get_idle_time() >= 0
        assert tracker.get_idle_time() < 1  # Should be very recent

    def test_update_activity(self):
        """Test activity update resets idle time"""
        tracker = RequestTracker()

        # Wait a bit
        time.sleep(0.1)
        initial_idle = tracker.get_idle_time()
        assert initial_idle >= 0.1

        # Update activity
        tracker.update()
        updated_idle = tracker.get_idle_time()

        # Idle time should be reset
        assert updated_idle < initial_idle
        assert updated_idle < 0.01  # Should be very recent

    def test_request_count_increments(self):
        """Test request counter increments"""
        tracker = RequestTracker()

        assert tracker._request_count == 0

        tracker.update()
        assert tracker._request_count == 1

        tracker.update()
        tracker.update()
        assert tracker._request_count == 3

    def test_get_last_activity(self):
        """Test get_last_activity returns timestamp"""
        tracker = RequestTracker()

        timestamp = tracker.get_last_activity()
        assert isinstance(timestamp, float)
        assert timestamp > 0

        # Should be recent (within last second)
        current_time = time.time()
        assert abs(current_time - timestamp) < 1.0

    def test_get_stats(self):
        """Test statistics retrieval"""
        tracker = RequestTracker()

        tracker.update()
        tracker.update()

        time.sleep(0.1)

        stats = tracker.get_stats()

        assert isinstance(stats, dict)
        assert 'total_requests' in stats
        assert 'uptime_seconds' in stats
        assert 'last_activity_seconds_ago' in stats

        assert stats['total_requests'] == 2
        assert stats['uptime_seconds'] > 0
        assert stats['last_activity_seconds_ago'] >= 0.1

    def test_reset(self):
        """Test reset functionality"""
        tracker = RequestTracker()

        # Generate some activity
        tracker.update()
        tracker.update()
        tracker.update()

        time.sleep(0.1)

        # Reset
        tracker.reset()

        # Should be reset
        assert tracker._request_count == 0
        assert tracker.get_idle_time() < 0.01

        stats = tracker.get_stats()
        assert stats['total_requests'] == 0

    def test_thread_safety_update(self):
        """Test concurrent updates are thread-safe"""
        tracker = RequestTracker()

        def update_many():
            for _ in range(100):
                tracker.update()

        # Spawn 5 threads, each doing 100 updates
        threads = [threading.Thread(target=update_many) for _ in range(5)]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # Should have exactly 500 requests
        assert tracker._request_count == 500

    def test_thread_safety_get_stats(self):
        """Test get_stats is thread-safe during updates"""
        tracker = RequestTracker()
        results = []

        def update_continuously():
            for _ in range(50):
                tracker.update()
                time.sleep(0.001)

        def read_stats():
            for _ in range(50):
                stats = tracker.get_stats()
                results.append(stats['total_requests'])
                time.sleep(0.001)

        # One thread updating, one thread reading
        update_thread = threading.Thread(target=update_continuously)
        read_thread = threading.Thread(target=read_stats)

        update_thread.start()
        read_thread.start()

        update_thread.join()
        read_thread.join()

        # Should never crash or return invalid data
        assert len(results) == 50
        assert all(isinstance(r, int) for r in results)
        assert all(r >= 0 for r in results)

    def test_idle_time_increases(self):
        """Test idle time increases over time"""
        tracker = RequestTracker()

        tracker.update()

        idle1 = tracker.get_idle_time()
        time.sleep(0.1)
        idle2 = tracker.get_idle_time()
        time.sleep(0.1)
        idle3 = tracker.get_idle_time()

        assert idle2 > idle1
        assert idle3 > idle2

    def test_uptime_increases(self):
        """Test uptime increases over time"""
        tracker = RequestTracker()

        time.sleep(0.1)

        stats1 = tracker.get_stats()
        time.sleep(0.1)
        stats2 = tracker.get_stats()

        assert stats2['uptime_seconds'] > stats1['uptime_seconds']
        assert stats2['uptime_seconds'] >= 0.2

    # ====================================================================
    # Active Request Tracking Tests (Fix for idle-monitor race condition)
    # ====================================================================

    def test_begin_end_request(self):
        """Test basic begin/end request tracking"""
        tracker = RequestTracker()

        assert tracker.has_active_requests() is False
        assert tracker._active_requests == 0

        # Begin request
        tracker.begin_request()
        assert tracker.has_active_requests() is True
        assert tracker._active_requests == 1

        # End request
        tracker.end_request()
        assert tracker.has_active_requests() is False
        assert tracker._active_requests == 0

    def test_active_request_count(self):
        """Test active request counting"""
        tracker = RequestTracker()

        stats = tracker.get_stats()
        assert stats['active_requests'] == 0

        # Start 3 requests
        tracker.begin_request()
        tracker.begin_request()
        tracker.begin_request()

        stats = tracker.get_stats()
        assert stats['active_requests'] == 3
        assert stats['total_requests'] == 3

        # End 1 request
        tracker.end_request()
        stats = tracker.get_stats()
        assert stats['active_requests'] == 2

        # End remaining
        tracker.end_request()
        tracker.end_request()
        stats = tracker.get_stats()
        assert stats['active_requests'] == 0

    def test_idle_time_zero_during_active(self):
        """Test that idle time is 0 during active requests"""
        tracker = RequestTracker()

        # Start request
        tracker.begin_request()

        # Even after waiting, idle time should be 0 while request is active
        time.sleep(0.2)
        assert tracker.get_idle_time() == 0.0

        # End request
        tracker.end_request()

        # Now idle time should start counting
        time.sleep(0.1)
        idle_time = tracker.get_idle_time()
        assert idle_time > 0
        assert idle_time >= 0.1

    def test_idle_time_after_completion(self):
        """Test that idle time starts from last completion, not last start"""
        tracker = RequestTracker()

        # Start request
        tracker.begin_request()

        # Simulate long-running request (like generation)
        time.sleep(0.2)

        # Idle time should still be 0 during request
        assert tracker.get_idle_time() == 0.0

        # End request
        tracker.end_request()

        # Idle time should now start from this completion time
        time.sleep(0.1)
        idle_time = tracker.get_idle_time()

        # Should be ~0.1s (time since completion), NOT ~0.3s (time since start)
        assert idle_time >= 0.1
        assert idle_time < 0.2  # Should not include the 0.2s request duration

    def test_multiple_active_requests(self):
        """Test multiple concurrent active requests"""
        tracker = RequestTracker()

        # Start multiple requests
        tracker.begin_request()  # Request 1
        time.sleep(0.05)
        tracker.begin_request()  # Request 2
        time.sleep(0.05)
        tracker.begin_request()  # Request 3

        assert tracker._active_requests == 3
        assert tracker.get_idle_time() == 0.0

        # End requests in different order
        tracker.end_request()  # End one
        assert tracker._active_requests == 2
        assert tracker.get_idle_time() == 0.0  # Still active

        tracker.end_request()  # End another
        assert tracker._active_requests == 1
        assert tracker.get_idle_time() == 0.0  # Still active

        tracker.end_request()  # End last
        assert tracker._active_requests == 0

        # Now idle time should count
        time.sleep(0.1)
        assert tracker.get_idle_time() > 0

    def test_thread_safety_active_requests(self):
        """Test thread safety of begin/end request"""
        tracker = RequestTracker()

        def simulate_request():
            tracker.begin_request()
            time.sleep(0.01)  # Simulate request processing
            tracker.end_request()

        # Spawn 10 threads simulating requests
        threads = [threading.Thread(target=simulate_request) for _ in range(10)]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # All requests should be complete
        assert tracker._active_requests == 0
        assert tracker._request_count == 10

    def test_has_active_requests(self):
        """Test has_active_requests() method"""
        tracker = RequestTracker()

        assert tracker.has_active_requests() is False

        tracker.begin_request()
        assert tracker.has_active_requests() is True

        tracker.begin_request()
        assert tracker.has_active_requests() is True

        tracker.end_request()
        assert tracker.has_active_requests() is True

        tracker.end_request()
        assert tracker.has_active_requests() is False

    def test_end_request_underflow_protection(self):
        """Test that end_request handles underflow gracefully"""
        tracker = RequestTracker()

        # Call end_request without begin_request
        tracker.end_request()

        # Should not go negative
        assert tracker._active_requests == 0
        assert tracker.has_active_requests() is False

        # Call multiple times
        tracker.end_request()
        tracker.end_request()

        # Should still be 0
        assert tracker._active_requests == 0

    def test_reset_clears_active_requests(self):
        """Test that reset clears active request count"""
        tracker = RequestTracker()

        # Start some requests
        tracker.begin_request()
        tracker.begin_request()
        assert tracker._active_requests == 2

        # Reset
        tracker.reset()

        # Active requests should be cleared
        assert tracker._active_requests == 0
        assert tracker.has_active_requests() is False

    def test_stats_includes_new_fields(self):
        """Test that get_stats includes new fields"""
        tracker = RequestTracker()

        tracker.begin_request()
        time.sleep(0.1)
        tracker.end_request()
        time.sleep(0.05)

        stats = tracker.get_stats()

        # Check new fields exist
        assert 'active_requests' in stats
        assert 'last_completion_seconds_ago' in stats

        # Verify values
        assert stats['active_requests'] == 0
        assert stats['last_completion_seconds_ago'] >= 0.05
        assert stats['last_completion_seconds_ago'] < 0.2
