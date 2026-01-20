"""InferenceEngine - Handles MLX model inference with backend abstraction.

Phase 3: Supports both text-only and vision models via backend routing.
"""

import logging
import base64
import io
from typing import Any, Dict, Iterator, Optional, List
from abc import ABC, abstractmethod

# Opus 4.5 High Priority Fix H3: Set PIL bomb protection at module load (not in method)
try:
    from PIL import Image
    # Set decompression bomb protection BEFORE any image operations
    # 50 megapixels = ~7071x7071 image = reasonable limit for vision models
    Image.MAX_IMAGE_PIXELS = 50_000_000
except ImportError:
    # PIL not available (text-only worker)
    Image = None

# Opus Improvement 1: Import MLX at module level for cache clearing
try:
    import mlx.core as mx
    _HAS_MLX_METAL = hasattr(mx, 'metal')
except ImportError:
    mx = None
    _HAS_MLX_METAL = False

logger = logging.getLogger(__name__)


# Opus Improvement 2: Extract cache clearing to helper function
def _clear_mlx_cache_safe(context: str = "inference"):
    """Clear MLX GPU cache, handling errors gracefully.

    Recommended by Opus 4.5 for memory management under sustained load.
    Prevents GPU memory fragmentation that causes worker crashes.

    Args:
        context: Description for logging (e.g., "text inference", "vision inference")
    """
    if _HAS_MLX_METAL:
        try:
            mx.metal.clear_cache()
            logger.debug(f"Cleared MLX GPU cache after {context}")
        except Exception as e:
            logger.debug(f"Could not clear MLX cache: {e}")


class InferenceBackend(ABC):
    """
    Abstract base class for inference backends.

    Backends handle model-specific inference logic (text-only vs vision).
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        images: Optional[List] = None
    ) -> Dict[str, Any]:
        """Generate completion."""
        pass

    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        images: Optional[List] = None
    ) -> Iterator[Dict[str, Any]]:
        """Generate streaming completion."""
        pass


class TextInferenceBackend(InferenceBackend):
    """
    Text-only inference backend using mlx_lm.

    Original implementation from Phase 1/2 with Opus 4.5 optimizations.
    """

    def __init__(self, model: Any, tokenizer: Any):
        """
        Initialize text inference backend.

        Args:
            model: Loaded MLX text model
            tokenizer: Loaded tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer
        # Phase 1.2: Cache samplers to avoid recreating on every request
        self._sampler_cache: Dict[tuple, Any] = {}
        self._logits_processor_cache: Dict[float, Any] = {}
        # OPTIMIZATION: Token batching for streaming (Opus 4.5)
        self._token_batch_size = 4  # Optimal for M4

    def _get_cached_sampler(self, temperature: float, top_p: float):
        """Get cached sampler or create new one."""
        key = (temperature, top_p)
        if key not in self._sampler_cache:
            from mlx_lm.sample_utils import make_sampler
            logger.debug(f"Creating new sampler: temp={temperature}, top_p={top_p}")
            self._sampler_cache[key] = make_sampler(temp=temperature, top_p=top_p)
        else:
            logger.debug(f"Using cached sampler: temp={temperature}, top_p={top_p}")
        return self._sampler_cache[key]

    def _get_cached_logits_processor(self, repetition_penalty: float):
        """Get cached logits processor or create new one."""
        if repetition_penalty == 1.0:
            return None

        if repetition_penalty not in self._logits_processor_cache:
            from mlx_lm.sample_utils import make_logits_processors
            logger.debug(f"Creating new logits processor: rep_penalty={repetition_penalty}")
            self._logits_processor_cache[repetition_penalty] = make_logits_processors(
                repetition_penalty=repetition_penalty
            )
        else:
            logger.debug(f"Using cached logits processor: rep_penalty={repetition_penalty}")
        return self._logits_processor_cache[repetition_penalty]

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 1.0,
        repetition_penalty: float = 1.1,
        images: Optional[List] = None
    ) -> Dict[str, Any]:
        """
        Generate text completion.

        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            repetition_penalty: Penalty for repeating tokens
            images: Ignored for text-only backend

        Returns:
            Dict with 'text', 'tokens', 'finish_reason'
        """
        if images:
            raise ValueError("Text-only model cannot process images")

        try:
            from mlx_lm import stream_generate

            logger.debug(f"Generating completion (max_tokens={max_tokens}, temp={temperature})")

            # Phase 1.2: Use cached samplers
            sampler = self._get_cached_sampler(temperature, top_p)
            logits_processors = self._get_cached_logits_processor(repetition_penalty)

            # Use stream_generate internally for accurate token counting
            text_chunks = []
            response_obj = None
            for resp in stream_generate(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                sampler=sampler,
                logits_processors=logits_processors
            ):
                text_chunks.append(resp.text)
                response_obj = resp

            full_text = "".join(text_chunks)
            logger.debug(f"Generated {response_obj.generation_tokens} tokens at {response_obj.generation_tps:.1f} tok/s")

            # MLX Optimization (Phase 1): Clear GPU cache after text inference
            _clear_mlx_cache_safe("text inference")

            return {
                "text": full_text,
                "tokens": response_obj.generation_tokens,
                "finish_reason": response_obj.finish_reason or "stop"
            }

        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            raise Exception(f"Inference failed: {e}")

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 1.0,
        repetition_penalty: float = 1.1,
        images: Optional[List] = None
    ) -> Iterator[Dict[str, Any]]:
        """
        Generate streaming text completion with async batching.

        Yields:
            Dict chunks with 'text', 'token', 'done', 'finish_reason'
        """
        if images:
            raise ValueError("Text-only model cannot process images")

        try:
            from mlx_lm import stream_generate

            logger.debug(f"Generating streaming completion (max_tokens={max_tokens})")

            sampler = self._get_cached_sampler(temperature, top_p)
            logits_processors = self._get_cached_logits_processor(repetition_penalty)

            # OPTIMIZATION: Accumulate tokens for batching
            batch_buffer = []
            batch_text = []

            for response in stream_generate(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                sampler=sampler,
                logits_processors=logits_processors
            ):
                batch_buffer.append(response)
                batch_text.append(response.text)

                should_yield = (
                    len(batch_buffer) >= self._token_batch_size or
                    response.finish_reason is not None
                )

                if should_yield:
                    combined_text = "".join(batch_text)
                    last_response = batch_buffer[-1]

                    chunk = {
                        "text": combined_text,
                        "token": last_response.token,
                        "done": last_response.finish_reason is not None,
                        "finish_reason": last_response.finish_reason,
                        "generation_tokens": last_response.generation_tokens,
                        "generation_tps": last_response.generation_tps
                    }
                    yield chunk

                    batch_buffer.clear()
                    batch_text.clear()

                if response.finish_reason is not None:
                    logger.debug(f"Stream complete: {response.generation_tokens} tokens")
                    break

            # MLX Optimization (Phase 1): Clear GPU cache after streaming completes
            _clear_mlx_cache_safe("text streaming")

        except Exception as e:
            logger.error(f"Streaming generation failed: {e}", exc_info=True)
            raise Exception(f"Streaming inference failed: {e}")


class VisionInferenceBackend(InferenceBackend):
    """
    Vision inference backend using mlx-vlm.

    Phase 3: Handles vision models (Qwen2-VL, LLaVA, etc.)
    """

    def __init__(self, model: Any, processor: Any):
        """
        Initialize vision inference backend.

        Args:
            model: Loaded MLX vision model
            processor: Loaded vision processor

        Raises:
            ValueError: If model or processor validation fails
        """
        self.model = model
        self.processor = processor

        # M7: Health check for vision backend
        # Validate model has expected attributes
        if not hasattr(model, 'config'):
            logger.warning("Vision model missing 'config' attribute")

        # Verify processor is usable
        try:
            # Check for tokenizer (most vision processors have this)
            if hasattr(processor, 'tokenizer'):
                _ = processor.tokenizer
            # Verify processor callable
            if not callable(getattr(processor, '__call__', None)):
                raise ValueError("Vision processor is not callable")
        except Exception as e:
            raise ValueError(f"Invalid vision processor: {e}")

    def _decode_images(self, image_data_list: List) -> List:
        """
        Decode ImageData objects to PIL images with security protections.

        Security Features:
        - PIL decompression bomb protection (set at module load - Opus H3)
        - Base64 bomb protection (size limits before decode)
        - Image count limits

        Args:
            image_data_list: List of ImageData from IPC

        Returns:
            List of PIL Image objects

        Raises:
            ValueError: If security limits exceeded or invalid image
        """
        # Security limits (Opus recommendations)
        MAX_IMAGES = 5
        MAX_BASE64_SIZE = 10 * 1024 * 1024  # 10MB

        # Check image count limit
        if len(image_data_list) > MAX_IMAGES:
            raise ValueError(
                f"Too many images: {len(image_data_list)} exceeds limit of {MAX_IMAGES}"
            )

        # PIL.Image.MAX_IMAGE_PIXELS already set at module load (Opus H3)

        images = []
        for idx, img_data in enumerate(image_data_list):
            try:
                if img_data.type == 'inline':
                    # Base64 bomb protection: check encoded size before decode
                    encoded_size = len(img_data.data)
                    if encoded_size > MAX_BASE64_SIZE * 4 / 3:  # Base64 expands by ~1.33x
                        raise ValueError(
                            f"Image {idx}: base64 data too large "
                            f"({encoded_size / 1024 / 1024:.1f}MB exceeds {MAX_BASE64_SIZE / 1024 / 1024:.0f}MB limit)"
                        )

                    # Decode base64
                    image_bytes = base64.b64decode(img_data.data)

                    # Check decoded size
                    decoded_size = len(image_bytes)
                    if decoded_size > MAX_BASE64_SIZE:
                        raise ValueError(
                            f"Image {idx}: decoded size too large "
                            f"({decoded_size / 1024 / 1024:.1f}MB exceeds {MAX_BASE64_SIZE / 1024 / 1024:.0f}MB limit)"
                        )

                    # Open image with PIL (protected by MAX_IMAGE_PIXELS)
                    pil_img = Image.open(io.BytesIO(image_bytes))

                    # Verify image loaded successfully
                    pil_img.verify()

                    # Re-open after verify (verify() consumes the file)
                    pil_img = Image.open(io.BytesIO(image_bytes))

                    images.append(pil_img)

                elif img_data.type == 'shmem':
                    # Read from shared memory
                    # TODO Phase 4: Implement shared memory reading in worker
                    raise NotImplementedError("Shared memory images not yet supported in worker")
                else:
                    raise ValueError(f"Image {idx}: Unknown image type: {img_data.type}")

            except ValueError:
                # Re-raise our validation errors
                raise
            except Exception as e:
                # Wrap other errors with context
                raise ValueError(f"Image {idx}: Failed to decode - {str(e)}")

        return images

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 1.0,
        repetition_penalty: float = 1.1,
        images: Optional[List] = None
    ) -> Dict[str, Any]:
        """
        Generate vision completion with comprehensive error handling.

        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            repetition_penalty: Penalty for repeating tokens
            images: List of ImageData objects

        Returns:
            Dict with 'text', 'tokens', 'finish_reason'

        Raises:
            ValueError: Invalid input (missing images, security limits exceeded, invalid image format)
            MemoryError: Insufficient memory for model or images
            Exception: Other inference failures
        """
        try:
            from mlx_vlm import generate as vlm_generate
            from mlx_vlm.prompt_utils import apply_chat_template
        except ImportError as e:
            logger.error("mlx-vlm not installed for vision model")
            raise EnvironmentError(
                "Vision models require mlx-vlm. "
                "Install with: venv-vision/bin/pip install mlx-vlm pillow"
            ) from e

        # Input validation
        if not images:
            raise ValueError("Vision model requires at least one image")

        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        logger.debug(f"Generating vision completion with {len(images)} images (max_tokens={max_tokens})")

        try:
            # Decode images from IPC format (includes security checks)
            pil_images = self._decode_images(images)

        except ValueError as e:
            # Security limit exceeded or invalid image
            logger.warning(f"Image validation failed: {e}")
            raise  # Re-raise with original message

        except MemoryError as e:
            logger.error(f"Out of memory decoding images")
            raise MemoryError(
                f"Insufficient memory to decode {len(images)} images. "
                f"Try reducing image count or size."
            ) from e

        try:
            # Apply vision chat template
            config = self.model.config if hasattr(self.model, 'config') else {}
            formatted_prompt = apply_chat_template(
                self.processor, config, prompt, num_images=len(pil_images)
            )

        except Exception as e:
            logger.error(f"Chat template formatting failed: {e}", exc_info=True)
            raise ValueError(f"Failed to format vision prompt: {e}") from e

        try:
            # Generate (mlx-vlm expects images as list, even for single image)
            output = vlm_generate(
                self.model,
                self.processor,
                formatted_prompt,
                image=pil_images,  # Always pass as list
                max_tokens=max_tokens,
                temp=temperature,
                verbose=False
            )

        except MemoryError as e:
            logger.error(f"Out of memory during vision inference")
            raise MemoryError(
                f"Insufficient memory for vision inference. "
                f"Try using a smaller model or reducing max_tokens."
            ) from e

        except Exception as e:
            logger.error(f"Vision model inference failed: {e}", exc_info=True)
            raise Exception(f"Vision model generation failed: {e}") from e

        # Parse output (mlx-vlm may return GenerationResult object or string)
        # Handle both cases for API compatibility
        if hasattr(output, 'text'):
            # GenerationResult object (newer mlx-vlm)
            output_text = output.text if output.text else ""
        elif isinstance(output, str):
            # Direct string (older mlx-vlm)
            output_text = output
        else:
            logger.warning(f"Unexpected output type: {type(output)}")
            output_text = str(output) if output else ""

        if not output_text:
            logger.warning("Vision model returned empty output")
            output_text = ""

        # Estimate tokens (rough approximation)
        tokens = len(output_text.split())

        logger.debug(f"Generated vision response: {len(output_text)} chars, ~{tokens} tokens")

        # MLX Optimization (v3.1.0): Clear GPU cache after vision inference
        _clear_mlx_cache_safe("vision inference")

        return {
            "text": output_text,
            "tokens": tokens,
            "finish_reason": "stop"
        }

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 1.0,
        repetition_penalty: float = 1.1,
        images: Optional[List] = None
    ) -> Iterator[Dict[str, Any]]:
        """
        Generate streaming vision completion.

        Note: mlx-vlm doesn't support streaming yet, so we generate full response
        and yield it as a single chunk.

        Yields:
            Dict chunks with 'text', 'token', 'done', 'finish_reason'
        """
        logger.debug("Vision streaming (non-streamed): generating full response")

        # Generate full response
        result = self.generate(prompt, max_tokens, temperature, top_p, repetition_penalty, images)

        # Yield as single chunk
        yield {
            "text": result["text"],
            "token": 0,  # Not available
            "done": True,
            "finish_reason": result["finish_reason"],
            "generation_tokens": result["tokens"],
            "generation_tps": 0.0  # Not tracked for vision
        }


class InferenceEngine:
    """
    Inference engine that routes to appropriate backend (text or vision).

    Phase 3: Acts as a router between TextInferenceBackend and VisionInferenceBackend.
    """

    def __init__(self, model: Any, tokenizer_or_processor: Any, model_type: str = "text"):
        """
        Initialize inference engine with appropriate backend.

        Args:
            model: Loaded MLX model
            tokenizer_or_processor: Tokenizer (text) or Processor (vision)
            model_type: "text" or "vision"
        """
        self.model_type = model_type

        if model_type == "vision":
            logger.info("Initializing VisionInferenceBackend")
            self.backend = VisionInferenceBackend(model, tokenizer_or_processor)
        else:
            logger.info("Initializing TextInferenceBackend")
            self.backend = TextInferenceBackend(model, tokenizer_or_processor)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 1.0,
        repetition_penalty: float = 1.1,
        images: Optional[List] = None
    ) -> Dict[str, Any]:
        """
        Generate completion using appropriate backend.

        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            repetition_penalty: Penalty for repeating tokens
            images: Optional list of ImageData (for vision models)

        Returns:
            Dict with 'text', 'tokens', 'finish_reason'
        """
        return self.backend.generate(
            prompt, max_tokens, temperature, top_p, repetition_penalty, images
        )

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 1.0,
        repetition_penalty: float = 1.1,
        images: Optional[List] = None
    ) -> Iterator[Dict[str, Any]]:
        """
        Generate streaming completion using appropriate backend.

        Yields:
            Dict chunks with 'text', 'token', 'done', 'finish_reason'
        """
        yield from self.backend.generate_stream(
            prompt, max_tokens, temperature, top_p, repetition_penalty, images
        )
