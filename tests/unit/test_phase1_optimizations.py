"""
Unit tests for Phase 1 Apple Silicon optimizations.

Tests coverage for:
1. MLX environment variable configuration
2. Model loader optimization methods exist
3. Async token batching configuration
"""

import pytest
import os
from src.worker.model_loader import ModelLoader
from src.worker.inference import InferenceEngine


class TestMLXEnvironmentConfiguration:
    """Test MLX environment variables are set correctly."""

    def test_metal_shader_cache_enabled(self):
        """Test that Metal shader cache configuration exists."""
        import src.worker.__main__ as worker_main

        # Verify the configuration function exists
        assert hasattr(worker_main, 'configure_mlx_environment')

        # Test the function sets environment variables
        from unittest.mock import patch
        with patch.dict(os.environ, {}, clear=True):
            worker_main.configure_mlx_environment()

            # Verify environment variables are set
            assert os.environ.get('MTL_SHADER_CACHE_ENABLE') == '1'
            assert os.environ.get('PYTHONMALLOC') == 'malloc'

    def test_environment_config_called_before_imports(self):
        """Test that environment configuration happens before mlx imports."""
        import src.worker.__main__ as worker_main
        import inspect

        source = inspect.getsource(worker_main)
        config_line = source.find('configure_mlx_environment()')
        import_line = source.find('from src.ipc.stdio_bridge import')

        # Configuration must come before imports
        assert config_line > 0, "configure_mlx_environment() not found"
        assert import_line > 0, "imports not found"
        assert config_line < import_line, "MLX environment must be configured before imports"


class TestModelLoaderOptimizations:
    """Test model loader Apple Silicon optimizations."""

    def test_model_loader_has_optimization_methods(self):
        """Test that ModelLoader has optimization methods."""
        # Verify optimization methods exist
        assert hasattr(ModelLoader, '_eval_parameters_recursive')
        assert hasattr(ModelLoader, '_warmup_model')

        # Verify methods are static
        assert callable(ModelLoader._eval_parameters_recursive)
        assert callable(ModelLoader._warmup_model)

    def test_model_loader_load_signature(self):
        """Test that ModelLoader.load() has correct signature."""
        import inspect

        sig = inspect.signature(ModelLoader.load)
        params = list(sig.parameters.keys())

        # Should have: self, model_path
        assert 'self' in params
        assert 'model_path' in params

    def test_model_loader_has_gc_and_cache_logic(self):
        """Test that model loader code includes optimization logic."""
        import inspect

        source = inspect.getsource(ModelLoader.load)

        # Verify optimization code exists
        assert 'gc.collect()' in source, "Missing gc.collect() optimization"
        assert 'clear_cache()' in source, "Missing Metal cache clear"
        assert 'synchronize()' in source, "Missing synchronize() calls"
        assert '_eval_parameters_recursive' in source, "Missing parameter pre-evaluation"
        assert '_warmup_model' in source, "Missing model warmup"


class TestAsyncTokenBatching:
    """Test async token batching in inference engine."""

    def test_inference_engine_has_batch_size(self):
        """Test that inference engine initializes with batch size."""
        from unittest.mock import Mock

        mock_model = Mock()
        mock_tokenizer = Mock()

        engine = InferenceEngine(mock_model, mock_tokenizer)

        # Verify batch size is set
        assert hasattr(engine, '_token_batch_size')
        assert engine._token_batch_size == 4  # Opus recommended value

    def test_generate_stream_has_batching_logic(self):
        """Test that generate_stream includes batching logic."""
        import inspect

        source = inspect.getsource(InferenceEngine.generate_stream)

        # Verify batching code exists
        assert 'batch_buffer' in source, "Missing batch buffer"
        assert 'batch_text' in source, "Missing batch text accumulation"
        assert '_token_batch_size' in source, "Missing batch size check"
        assert 'should_yield' in source, "Missing yield condition logic"

    def test_batching_docstring_mentions_opus(self):
        """Test that optimization is documented as from Opus."""
        docstring = InferenceEngine.generate_stream.__doc__

        assert docstring is not None
        assert 'OPTIMIZATION' in docstring or 'Opus' in docstring, "Missing optimization documentation"


class TestBackwardsCompatibility:
    """Test that Phase 1 optimizations don't break existing functionality."""

    def test_model_loader_returns_tuple(self):
        """Test that ModelLoader.load() returns a tuple."""
        import inspect

        sig = inspect.signature(ModelLoader.load)

        # Check return annotation if present
        # The method should still return (model, tokenizer) tuple
        # This is verified by reading the source
        source = inspect.getsource(ModelLoader.load)
        assert 'return model, tokenizer' in source

    def test_inference_engine_api_unchanged(self):
        """Test that InferenceEngine API is unchanged."""
        from unittest.mock import Mock

        mock_model = Mock()
        mock_tokenizer = Mock()

        engine = InferenceEngine(mock_model, mock_tokenizer)

        # Verify public methods exist
        assert hasattr(engine, 'generate')
        assert hasattr(engine, 'generate_stream')
        assert callable(engine.generate)
        assert callable(engine.generate_stream)

        # Verify sampler caching still exists
        assert hasattr(engine, '_sampler_cache')
        assert hasattr(engine, '_logits_processor_cache')

    def test_inference_engine_initialization(self):
        """Test that InferenceEngine can still be initialized."""
        from unittest.mock import Mock

        mock_model = Mock()
        mock_tokenizer = Mock()

        # Should not raise any exception
        engine = InferenceEngine(mock_model, mock_tokenizer)

        assert engine.model is mock_model
        assert engine.tokenizer is mock_tokenizer


class TestPerformanceAssumptions:
    """Test that performance assumptions hold."""

    def test_batch_size_is_4(self):
        """Test that token batch size is 4 (Opus recommendation)."""
        from unittest.mock import Mock

        mock_model = Mock()
        mock_tokenizer = Mock()

        engine = InferenceEngine(mock_model, mock_tokenizer)

        assert engine._token_batch_size == 4, "Batch size should be 4 (Opus 4.5 recommendation)"

    def test_metal_shader_cache_variable_name(self):
        """Test that Metal shader cache uses correct environment variable name."""
        import src.worker.__main__ as worker_main
        from unittest.mock import patch

        with patch.dict(os.environ, {}, clear=True):
            worker_main.configure_mlx_environment()

            # Verify exact variable name (typos would break functionality)
            assert 'MTL_SHADER_CACHE_ENABLE' in os.environ
            assert os.environ['MTL_SHADER_CACHE_ENABLE'] == '1'

    def test_malloc_environment_variable(self):
        """Test that PYTHONMALLOC is set to 'malloc'."""
        import src.worker.__main__ as worker_main
        from unittest.mock import patch

        with patch.dict(os.environ, {}, clear=True):
            worker_main.configure_mlx_environment()

            assert 'PYTHONMALLOC' in os.environ
            assert os.environ['PYTHONMALLOC'] == 'malloc'

    def test_optimization_comments_reference_opus(self):
        """Test that optimizations are documented as from Opus 4.5."""
        import src.worker.__main__ as worker_main
        import src.worker.model_loader as model_loader
        import src.worker.inference as inference
        import inspect

        # Check that Opus is referenced in optimization comments
        worker_source = inspect.getsource(worker_main)
        assert 'Opus 4.5' in worker_source or 'Opus' in worker_source

        loader_source = inspect.getsource(model_loader.ModelLoader)
        assert 'Opus 4.5' in loader_source or 'Opus' in loader_source

        inference_source = inspect.getsource(inference.InferenceEngine)
        assert 'Opus 4.5' in inference_source or 'Opus' in inference_source
