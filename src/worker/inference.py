"""InferenceEngine - Handles MLX model inference."""

import logging
from typing import Any, Dict, Iterator

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Handles MLX model inference (generation) with Apple Silicon optimizations.

    Based on Opus 4.5 performance review:
    - Async token batching (prevents GPU stalls from synchronous IPC)
    - Cached samplers and logits processors
    - Optimized for Metal GPU pipeline
    """

    def __init__(self, model: Any, tokenizer: Any):
        """
        Initialize inference engine.

        Args:
            model: Loaded MLX model
            tokenizer: Loaded tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer
        # Phase 1.2: Cache samplers to avoid recreating on every request
        self._sampler_cache: Dict[tuple, Any] = {}
        self._logits_processor_cache: Dict[float, Any] = {}

        # OPTIMIZATION: Token batching for streaming (Opus 4.5)
        # Batch this many tokens before yielding to prevent GPU stalls
        # from synchronous flush() on each token
        self._token_batch_size = 4  # Optimal for M4 according to Opus

    def _get_cached_sampler(self, temperature: float, top_p: float):
        """Get cached sampler or create new one.

        Args:
            temperature: Sampling temperature
            top_p: Top-p sampling parameter

        Returns:
            Sampler instance
        """
        key = (temperature, top_p)
        if key not in self._sampler_cache:
            from mlx_lm.sample_utils import make_sampler
            logger.debug(f"Creating new sampler: temp={temperature}, top_p={top_p}")
            self._sampler_cache[key] = make_sampler(temp=temperature, top_p=top_p)
        else:
            logger.debug(f"Using cached sampler: temp={temperature}, top_p={top_p}")
        return self._sampler_cache[key]

    def _get_cached_logits_processor(self, repetition_penalty: float):
        """Get cached logits processor or create new one.

        Args:
            repetition_penalty: Repetition penalty value

        Returns:
            Logits processor or None if penalty is 1.0
        """
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
        repetition_penalty: float = 1.1
    ) -> Dict[str, Any]:
        """
        Generate completion for prompt.

        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            repetition_penalty: Penalty for repeating tokens (1.0 = no penalty, >1.0 = penalize)

        Returns:
            Dict with 'text', 'tokens', 'finish_reason'
        """
        try:
            from mlx_lm import stream_generate

            logger.debug(f"Generating completion (max_tokens={max_tokens}, temp={temperature}, rep_penalty={repetition_penalty})")

            # Phase 1.2: Use cached samplers
            sampler = self._get_cached_sampler(temperature, top_p)
            logits_processors = self._get_cached_logits_processor(repetition_penalty)

            # Use stream_generate internally for accurate token counting
            # Phase 1.3: Use list+join instead of string concatenation (faster for many iterations)
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
                text_chunks.append(resp.text)  # Collect chunks
                response_obj = resp  # Keep last response for metadata

            full_text = "".join(text_chunks)  # Join once at end
            logger.debug(f"Generated {response_obj.generation_tokens} tokens at {response_obj.generation_tps:.1f} tok/s")

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
        repetition_penalty: float = 1.1
    ) -> Iterator[Dict[str, Any]]:
        """
        Generate completion for prompt (streaming with async batching).

        OPTIMIZATION (Opus 4.5): Batch tokens before yielding to prevent GPU stalls.
        Instead of yielding each token immediately (which triggers synchronous flush()),
        we accumulate tokens and yield in batches. This keeps the GPU pipeline full.

        Expected gain: 3-4% performance improvement

        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            repetition_penalty: Penalty for repeating tokens (1.0 = no penalty, >1.0 = penalize)

        Yields:
            Dict chunks with 'text', 'token', 'done', 'finish_reason'
        """
        try:
            from mlx_lm import stream_generate

            logger.debug(f"Generating streaming completion (max_tokens={max_tokens}, temp={temperature}, rep_penalty={repetition_penalty})")

            # Phase 1.2: Use cached samplers
            sampler = self._get_cached_sampler(temperature, top_p)
            logits_processors = self._get_cached_logits_processor(repetition_penalty)

            # OPTIMIZATION: Accumulate tokens for batching
            batch_buffer = []
            batch_text = []

            # Stream generation
            for response in stream_generate(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                sampler=sampler,
                logits_processors=logits_processors
            ):
                # Accumulate this token
                batch_buffer.append(response)
                batch_text.append(response.text)

                # Check if we should yield this batch
                should_yield = (
                    len(batch_buffer) >= self._token_batch_size or  # Batch full
                    response.finish_reason is not None  # Generation complete
                )

                if should_yield:
                    # Yield batched chunk
                    # Combine all text from this batch
                    combined_text = "".join(batch_text)
                    last_response = batch_buffer[-1]

                    chunk = {
                        "text": combined_text,
                        "token": last_response.token,  # Last token in batch
                        "done": last_response.finish_reason is not None,
                        "finish_reason": last_response.finish_reason,
                        "generation_tokens": last_response.generation_tokens,
                        "generation_tps": last_response.generation_tps
                    }
                    yield chunk

                    # Clear batch
                    batch_buffer.clear()
                    batch_text.clear()

                # Break if done
                if response.finish_reason is not None:
                    logger.debug(f"Stream complete: {response.generation_tokens} tokens, reason: {response.finish_reason}")
                    break

        except Exception as e:
            logger.error(f"Streaming generation failed: {e}", exc_info=True)
            raise Exception(f"Streaming inference failed: {e}")
