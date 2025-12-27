"""
Unit Tests for ExtendedModelProvider

Tests model loading, unloading, and memory management.
"""

import pytest
import threading
import time
from unittest.mock import MagicMock, patch
from src.extended_model_provider import ExtendedModelProvider


class TestExtendedModelProvider:
    """Test suite for ExtendedModelProvider"""

    def test_init(self, mock_args):
        """Test initialization"""
        provider = ExtendedModelProvider(mock_args)

        assert provider.model is None
        assert provider.tokenizer is None
        assert provider._lock is not None
        assert provider._load_time is None
        assert provider._memory_at_load is None

    def test_load_model_updates_tracking(self, mock_args, mock_mlx_metal, mocker):
        """Test that loading model updates memory tracking"""
        # Mock the parent ModelProvider.load method to prevent real HuggingFace calls
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        def mock_load_side_effect(model_path, adapter_path=None, draft_model_path=None):
            # Simulate parent class behavior: set instance variables
            provider.model = mock_model
            provider.tokenizer = mock_tokenizer
            provider.model_key = (model_path, adapter_path, draft_model_path)
            return (mock_model, mock_tokenizer)

        mock_parent_load = mocker.patch('mlx_lm.server.ModelProvider.load', side_effect=mock_load_side_effect)

        # Mock Metal memory
        mock_mlx_metal['get_active_memory'].return_value = 18_000_000_000  # 18GB

        provider = ExtendedModelProvider(mock_args)

        # Load model
        model, tokenizer = provider.load("mlx-community/test-model")

        # Verify parent load was called
        mock_parent_load.assert_called_once_with("mlx-community/test-model", None, None)

        # Verify model is set
        assert provider.model is not None
        assert provider.tokenizer is not None
        assert provider._load_time is not None
        assert provider._memory_at_load == 18_000_000_000

    def test_unload_model_clears_memory(self, mock_args, mock_mlx_metal):
        """Test that unloading model clears Metal cache"""
        provider = ExtendedModelProvider(mock_args)

        # Set up loaded state
        provider.model = MagicMock()
        provider.tokenizer = MagicMock()
        provider.model_key = ("test-model", None, None)
        provider._load_time = time.time()

        # Mock memory before/after
        mock_mlx_metal['get_active_memory'].side_effect = [
            18_000_000_000,  # before unload
            150_000_000      # after unload
        ]
        mock_mlx_metal['get_cache_memory'].side_effect = [
            500_000_000,  # before clear
            0             # after clear
        ]

        # Unload
        stats = provider.unload()

        # Verify cleanup was called
        mock_mlx_metal['clear_cache'].assert_called_once()

        # Verify state is reset
        assert provider.model is None
        assert provider.tokenizer is None
        assert provider.model_key is None
        assert provider._load_time is None

        # Verify stats
        assert stats['memory_freed_gb'] > 0
        assert stats['model_name'] == "test-model"
        assert stats['cache_cleared_gb'] > 0

    def test_unload_no_model(self, mock_args):
        """Test unload when no model loaded"""
        provider = ExtendedModelProvider(mock_args)

        stats = provider.unload()

        assert stats['memory_freed_gb'] == 0.0
        assert stats['model_name'] is None
        assert provider.model is None

    def test_get_memory_stats(self, mock_args, mock_mlx_metal):
        """Test memory stats retrieval"""
        mock_mlx_metal['get_active_memory'].return_value = 18_000_000_000
        mock_mlx_metal['get_cache_memory'].return_value = 500_000_000
        mock_mlx_metal['get_peak_memory'].return_value = 19_000_000_000

        provider = ExtendedModelProvider(mock_args)
        provider.model = MagicMock()
        provider.model_key = ("test-model", None, None)
        provider._load_time = time.time() - 100  # 100 seconds ago

        stats = provider.get_memory_stats()

        assert isinstance(stats['active_memory_gb'], float)
        assert stats['active_memory_gb'] > 0
        assert stats['model_loaded'] is True
        assert stats['model_name'] == "test-model"
        assert stats['uptime_seconds'] >= 100

    def test_get_memory_stats_no_model(self, mock_args, mock_mlx_metal):
        """Test memory stats when no model loaded"""
        mock_mlx_metal['get_active_memory'].return_value = 1000000  # 1MB

        provider = ExtendedModelProvider(mock_args)

        stats = provider.get_memory_stats()

        assert stats['model_loaded'] is False
        assert stats['model_name'] is None
        assert stats['uptime_seconds'] is None

    def test_is_loaded(self, mock_args):
        """Test is_loaded check"""
        provider = ExtendedModelProvider(mock_args)

        # Initially not loaded
        assert provider.is_loaded() is False

        # Set model
        provider.model = MagicMock()
        assert provider.is_loaded() is True

        # Clear model
        provider.model = None
        assert provider.is_loaded() is False

    def test_get_model_info(self, mock_args):
        """Test get_model_info returns correct information"""
        provider = ExtendedModelProvider(mock_args)

        # No model loaded
        assert provider.get_model_info() is None

        # Set model info
        provider.model_key = ("mlx-community/phi-4", "/path/to/adapter", None)
        provider.model = MagicMock()

        info = provider.get_model_info()

        assert info is not None
        assert info['model_path'] == "mlx-community/phi-4"
        assert info['adapter_path'] == "/path/to/adapter"
        assert info['draft_model_path'] is None

    def test_thread_safety_load(self, mock_args, mock_mlx_metal, mocker):
        """Test thread-safe loading"""
        def mock_load_side_effect(model_path, adapter_path=None, draft_model_path=None):
            # Simulate parent class behavior: set instance variables
            provider.model = MagicMock()
            provider.tokenizer = MagicMock()
            provider.model_key = (model_path, adapter_path, draft_model_path)
            return (provider.model, provider.tokenizer)

        # Mock the parent ModelProvider.load method to prevent real HuggingFace calls
        mock_parent_load = mocker.patch('mlx_lm.server.ModelProvider.load', side_effect=mock_load_side_effect)
        mock_mlx_metal['get_active_memory'].return_value = 18_000_000_000

        provider = ExtendedModelProvider(mock_args)
        results = []

        def load_model():
            try:
                provider.load("test-model")
                results.append("success")
            except Exception as e:
                results.append(f"error: {e}")

        # Spawn multiple threads trying to load
        threads = [threading.Thread(target=load_model) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should handle concurrent loads gracefully (via lock)
        # At least one should succeed
        assert any(r == "success" for r in results)

    def test_thread_safety_unload(self, mock_args, mock_mlx_metal):
        """Test thread-safe unloading"""
        provider = ExtendedModelProvider(mock_args)
        provider.model = MagicMock()
        provider.tokenizer = MagicMock()
        provider.model_key = ("test", None, None)

        mock_mlx_metal['get_active_memory'].return_value = 1000
        mock_mlx_metal['get_cache_memory'].return_value = 0

        results = []

        def unload_model():
            try:
                stats = provider.unload()
                results.append(("success", stats['model_name']))
            except Exception as e:
                results.append(("error", str(e)))

        # Spawn multiple threads trying to unload
        threads = [threading.Thread(target=unload_model) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should handle gracefully - first unloads, rest see no model
        assert len(results) == 3
        # At least one should successfully unload
        success_count = sum(1 for r in results if r[0] == "success")
        assert success_count >= 1

    def test_memory_calculation_accuracy(self, mock_args, mock_mlx_metal):
        """Test memory freed calculation is accurate"""
        provider = ExtendedModelProvider(mock_args)
        provider.model = MagicMock()
        provider.tokenizer = MagicMock()
        provider.model_key = ("test", None, None)

        # Set specific memory values
        memory_before = 20_000_000_000  # 20GB
        memory_after = 1_000_000_000     # 1GB
        cache_before = 500_000_000       # 500MB
        cache_after = 0                   # 0MB

        mock_mlx_metal['get_active_memory'].side_effect = [memory_before, memory_after]
        mock_mlx_metal['get_cache_memory'].side_effect = [cache_before, cache_after]

        stats = provider.unload()

        # Check calculations
        expected_freed = (memory_before - memory_after) / (1024**3)
        expected_cache = (cache_before - cache_after) / (1024**3)

        assert abs(stats['memory_freed_gb'] - expected_freed) < 0.01
        assert abs(stats['cache_cleared_gb'] - expected_cache) < 0.01
        assert stats['memory_before_gb'] == pytest.approx(memory_before / (1024**3))
        assert stats['memory_after_gb'] == pytest.approx(memory_after / (1024**3))

    def test_unload_with_draft_model(self, mock_args, mock_mlx_metal):
        """Test unloading when draft model is present"""
        provider = ExtendedModelProvider(mock_args)
        provider.model = MagicMock()
        provider.tokenizer = MagicMock()
        provider.draft_model = MagicMock()
        provider.model_key = ("test", None, "draft")

        mock_mlx_metal['get_active_memory'].side_effect = [1000, 100]
        mock_mlx_metal['get_cache_memory'].side_effect = [0, 0]

        provider.unload()

        # Verify draft model was also cleared
        assert provider.draft_model is None

    def test_load_time_tracking(self, mock_args, mock_mlx_metal, mocker):
        """Test that load time is tracked correctly"""
        # Simulate slow load
        def slow_load(model_path, adapter_path=None, draft_model_path=None):
            time.sleep(0.1)
            # Simulate parent class behavior: set instance variables
            provider.model = MagicMock()
            provider.tokenizer = MagicMock()
            provider.model_key = (model_path, adapter_path, draft_model_path)
            return (provider.model, provider.tokenizer)

        # Mock the parent ModelProvider.load method with slow load
        mock_parent_load = mocker.patch('mlx_lm.server.ModelProvider.load', side_effect=slow_load)
        mock_mlx_metal['get_active_memory'].return_value = 1000

        provider = ExtendedModelProvider(mock_args)

        start_time = time.time()
        provider.load("test-model")
        end_time = time.time()

        # Verify load time is recorded and reasonable
        assert provider._load_time is not None
        assert provider._load_time >= start_time
        assert provider._load_time <= end_time

    def test_uptime_calculation(self, mock_args):
        """Test uptime calculation in stats"""
        provider = ExtendedModelProvider(mock_args)
        provider.model = MagicMock()
        provider.model_key = ("test", None, None)
        provider._load_time = time.time() - 50  # 50 seconds ago

        stats = provider.get_memory_stats()

        assert stats['uptime_seconds'] >= 50
        assert stats['uptime_seconds'] < 51  # Should be close to 50

    def test_eval_called_before_model_deletion(self, mock_args, mock_mlx_metal):
        """
        Test that mx.eval() is called on models before deletion.

        This is critical for proper memory cleanup in MLX 0.30.0+.
        Without eval, the Load primitive holds references and prevents
        memory freeing.

        See: https://github.com/ml-explore/mlx/pull/1702
        """
        provider = ExtendedModelProvider(mock_args)

        # Set up loaded state
        mock_model = MagicMock()
        mock_draft_model = MagicMock()
        provider.model = mock_model
        provider.tokenizer = MagicMock()
        provider.draft_model = mock_draft_model
        provider.model_key = ("test-model", None, "draft")
        provider._load_time = time.time()

        # Mock memory values
        mock_mlx_metal['get_active_memory'].side_effect = [
            5_000_000_000,  # before unload
            150_000_000     # after unload
        ]
        mock_mlx_metal['get_cache_memory'].side_effect = [0, 0]

        # Unload
        stats = provider.unload()

        # CRITICAL: Verify mx.eval() was called on both models before deletion
        # This ensures proper memory release
        assert mock_mlx_metal['eval'].call_count == 2
        mock_mlx_metal['eval'].assert_any_call(mock_model)
        mock_mlx_metal['eval'].assert_any_call(mock_draft_model)

        # Verify memory was freed
        assert stats['memory_freed_gb'] > 0

        # Verify clear_cache was called after eval
        mock_mlx_metal['clear_cache'].assert_called_once()

    def test_eval_not_called_on_none_draft_model(self, mock_args, mock_mlx_metal):
        """Test that mx.eval() is only called on non-None draft models"""
        provider = ExtendedModelProvider(mock_args)

        # Set up loaded state WITHOUT draft model
        mock_model = MagicMock()
        provider.model = mock_model
        provider.tokenizer = MagicMock()
        provider.draft_model = None  # No draft model
        provider.model_key = ("test-model", None, None)
        provider._load_time = time.time()

        # Mock memory values
        mock_mlx_metal['get_active_memory'].side_effect = [1000, 100]
        mock_mlx_metal['get_cache_memory'].side_effect = [0, 0]

        # Unload
        provider.unload()

        # Verify mx.eval() was called only once (on main model, not draft)
        assert mock_mlx_metal['eval'].call_count == 1
        mock_mlx_metal['eval'].assert_called_once_with(mock_model)
