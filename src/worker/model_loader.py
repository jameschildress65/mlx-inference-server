"""ModelLoader - Loads MLX models from HuggingFace."""

import logging
import gc
import time
from typing import Tuple, Any

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Loads MLX models using mlx_lm with Apple Silicon optimizations.

    Based on Opus 4.5 performance review recommendations:
    - Clear Metal cache before load (clean slate)
    - Pre-evaluate parameters hierarchically (force memory allocation)
    - Warm up with dummy forward pass (compile Metal shaders)
    - Clear warmup from KV cache (clean state for inference)
    """

    def load(self, model_path: str) -> Tuple[Any, Any]:
        """
        Load MLX model and tokenizer with Apple Silicon optimizations.

        Args:
            model_path: HuggingFace model path (e.g., "mlx-community/Qwen2.5-7B-Instruct-4bit")

        Returns:
            Tuple of (model, tokenizer)

        Raises:
            Exception: If model loading fails
        """
        try:
            import mlx.core as mx
            from mlx_lm import load

            logger.info(f"Loading model from: {model_path}")

            # OPTIMIZATION 1: Clear Metal cache before load (Opus 4.5 recommendation)
            # Ensures clean GPU state, prevents fragmentation from previous loads
            gc.collect()  # Python garbage collection
            mx.metal.clear_cache()  # Metal GPU memory cache
            mx.synchronize()  # Wait for GPU to finish any pending operations

            # Track memory and timing
            start_memory = mx.metal.get_active_memory() / (1024**3)  # GB
            start_time = time.perf_counter()

            # Load model and tokenizer
            model, tokenizer = load(model_path)

            # OPTIMIZATION 2: Pre-evaluate ALL parameters hierarchically (Opus 4.5)
            # Forces immediate memory allocation rather than lazy allocation on first inference
            # This prevents first-request latency spike and ensures stable memory usage
            logger.debug("Pre-evaluating model parameters (forcing memory allocation)...")
            self._eval_parameters_recursive(model, mx)
            mx.synchronize()  # Wait for all evaluations to complete

            # OPTIMIZATION 3: Warm up with dummy forward pass (Opus 4.5)
            # Compiles Metal shaders on first use - do this during load, not first request
            logger.debug("Warming up inference path (compiling Metal shaders)...")
            self._warmup_model(model, tokenizer, mx)
            mx.synchronize()

            # Calculate load stats
            load_time = time.perf_counter() - start_time
            memory_used = (mx.metal.get_active_memory() / (1024**3)) - start_memory

            logger.info(
                f"Model loaded successfully: {model_path} "
                f"({memory_used:.2f} GB in {load_time:.1f}s)"
            )

            return model, tokenizer

        except ImportError as e:
            logger.error(f"mlx_lm not installed: {e}")
            raise Exception(f"mlx_lm not available: {e}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            raise Exception(f"Model load failed: {e}")

    @staticmethod
    def _eval_parameters_recursive(module: Any, mx: Any) -> None:
        """
        Recursively evaluate all model parameters to force memory allocation.

        This prevents lazy allocation and ensures stable first-request performance.
        Opus 4.5: "Evaluate parameters hierarchically for better memory layout"
        """
        # Evaluate this module's parameters
        if hasattr(module, 'parameters'):
            params = module.parameters()
            if isinstance(params, dict):
                for param in params.values():
                    mx.eval(param)

        # Recursively evaluate children
        if hasattr(module, 'children'):
            children = module.children()
            if isinstance(children, dict):
                for child in children.values():
                    ModelLoader._eval_parameters_recursive(child, mx)

    @staticmethod
    def _warmup_model(model: Any, tokenizer: Any, mx: Any) -> None:
        """
        Run a dummy forward pass to compile Metal shaders.

        First inference compiles shaders which adds ~500ms-1s latency.
        Do this during model load so first user request is fast.

        Opus 4.5: "Warm up the inference path - compile Metal shaders during load"
        """
        try:
            # Tokenize a short warmup string
            warmup_text = "Hello"
            warmup_tokens = tokenizer.encode(warmup_text)

            # Handle different tokenizer output formats
            if isinstance(warmup_tokens, list):
                warmup_ids = mx.array([warmup_tokens])
            else:
                warmup_ids = mx.array([[warmup_tokens]])

            # Forward pass to trigger shader compilation
            # Different models have different forward() signatures
            try:
                _ = model(warmup_ids)
            except TypeError:
                # Some models need different args
                try:
                    _ = model(warmup_ids, cache=None)
                except Exception:
                    pass  # If warmup fails, continue anyway

            mx.synchronize()

            # OPTIMIZATION 4: Clear warmup from KV cache (Opus 4.5)
            # Ensure model starts fresh for actual inference
            if hasattr(model, 'reset_cache'):
                model.reset_cache()
            if hasattr(model, 'cache') and model.cache is not None:
                if hasattr(model.cache, 'clear'):
                    model.cache.clear()

        except Exception as e:
            # Warmup failure is not critical - log and continue
            logger.debug(f"Model warmup failed (non-critical): {e}")
