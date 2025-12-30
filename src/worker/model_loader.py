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

    Phase 3: Supports both text-only and vision models (Qwen2-VL, etc.)
    """

    @staticmethod
    def detect_model_capabilities(model_path: str) -> dict:
        """
        Detect model capabilities from model path.

        Args:
            model_path: HuggingFace model path

        Returns:
            Dict with capabilities: {"text": bool, "vision": bool, "detection_method": str}
        """
        vision_patterns = [
            "qwen2-vl",
            "qwen2.5-vl",
            "-vl-",
            "llava",
            "idefics"
        ]

        model_lower = model_path.lower()
        is_vision = any(pattern in model_lower for pattern in vision_patterns)

        return {
            "text": True,  # All models support text
            "vision": is_vision,
            "detection_method": "model_name"
        }

    def load(self, model_path: str) -> Tuple[Any, Any]:
        """
        Load MLX model and tokenizer with Apple Silicon optimizations.

        Routes to vision or text loader based on model capabilities.

        Args:
            model_path: HuggingFace model path (e.g., "mlx-community/Qwen2.5-7B-Instruct-4bit")

        Returns:
            Tuple of (model, tokenizer/processor)

        Raises:
            Exception: If model loading fails
        """
        # Phase 3: Detect model capabilities
        capabilities = self.detect_model_capabilities(model_path)

        if capabilities["vision"]:
            logger.info(f"Detected vision model: {model_path}")
            return self._load_vision_model(model_path)
        else:
            logger.info(f"Detected text-only model: {model_path}")
            return self._load_text_model(model_path)

    def _load_text_model(self, model_path: str) -> Tuple[Any, Any]:
        """
        Load text-only MLX model (original implementation).

        Args:
            model_path: HuggingFace model path

        Returns:
            Tuple of (model, tokenizer)
        """
        try:
            import mlx.core as mx
            from mlx_lm import load

            logger.info(f"Loading text model from: {model_path}")

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

    def _load_vision_model(self, model_path: str) -> Tuple[Any, Any]:
        """
        Load vision model using mlx-vlm.

        Phase 3: Vision model loading with same Metal optimizations as text models.

        Args:
            model_path: HuggingFace model path for vision model

        Returns:
            Tuple of (model, processor)

        Raises:
            Exception: If mlx-vlm not installed or model loading fails
        """
        try:
            import mlx.core as mx

            # Check mlx-vlm availability
            try:
                from mlx_vlm import load as vlm_load
                from mlx_vlm import generate as vlm_generate
            except ImportError:
                raise Exception(
                    "mlx-vlm is required for vision models. "
                    "Install: pip install mlx-vlm pillow"
                )

            logger.info(f"Loading vision model from: {model_path}")

            # OPTIMIZATION 1: Clear Metal cache before load (same as text models)
            gc.collect()
            mx.metal.clear_cache()
            mx.synchronize()

            start_memory = mx.metal.get_active_memory() / (1024**3)
            start_time = time.perf_counter()

            # Load vision model and processor
            model, processor = vlm_load(model_path)

            # OPTIMIZATION 2: Pre-evaluate parameters (same as text models)
            logger.debug("Pre-evaluating vision model parameters...")
            self._eval_parameters_recursive(model, mx)
            mx.synchronize()

            # OPTIMIZATION 3: Warm up vision model
            logger.debug("Warming up vision inference path...")
            self._warmup_vision_model(model, processor, mx)
            mx.synchronize()

            # Calculate load stats
            load_time = time.perf_counter() - start_time
            memory_used = (mx.metal.get_active_memory() / (1024**3)) - start_memory

            logger.info(
                f"Vision model loaded successfully: {model_path} "
                f"({memory_used:.2f} GB in {load_time:.1f}s)"
            )

            return model, processor

        except ImportError as e:
            logger.error(f"mlx-vlm not installed: {e}")
            raise Exception(f"mlx-vlm not available: {e}")

        except Exception as e:
            logger.error(f"Failed to load vision model: {e}", exc_info=True)
            raise Exception(f"Vision model load failed: {e}")

    @staticmethod
    def _warmup_vision_model(model: Any, processor: Any, mx: Any) -> None:
        """
        Warm up vision model with dummy inference.

        Similar to text model warmup but with image input.

        Args:
            model: Vision model
            processor: Vision processor
            mx: MLX module
        """
        try:
            from PIL import Image
            import io

            # Create tiny dummy image (1x1 red pixel)
            dummy_img = Image.new('RGB', (1, 1), color='red')

            # Dummy prompt
            dummy_prompt = "Hello"

            # Try vision inference warmup
            # Note: mlx-vlm handles preprocessing internally
            try:
                from mlx_vlm.prompt_utils import apply_chat_template
                from mlx_vlm import generate as vlm_generate

                # Apply chat template for vision
                # Get model config if available
                config = model.config if hasattr(model, 'config') else {}
                formatted_prompt = apply_chat_template(
                    processor, config, dummy_prompt, num_images=1
                )

                # Dummy generation (just to compile shaders)
                _ = vlm_generate(
                    model, processor,
                    formatted_prompt,
                    image=dummy_img,
                    max_tokens=1,
                    verbose=False
                )

            except Exception as e:
                # Warmup failure is non-critical
                logger.debug(f"Vision warmup failed (non-critical): {e}")

            mx.synchronize()

        except Exception as e:
            logger.debug(f"Vision warmup failed (non-critical): {e}")
