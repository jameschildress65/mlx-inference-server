"""FastAPI application for MLX Server V3."""

import logging
import json
from functools import lru_cache
from typing import Dict, Any, Optional, Union, Literal
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from .worker_manager import WorkerManager, NoModelLoadedError, WorkerError
from ..ipc.messages import CompletionRequest as IPCCompletionRequest
from ..ipc.shared_memory_bridge import WorkerCommunicationError
# StdioBridge no longer used directly - WorkerManager handles IPC abstraction
from ..config.server_config import ServerConfig
from .image_utils import prepare_images, ImageProcessingError

logger = logging.getLogger(__name__)

# Phase 1.1: Tokenizer cache - avoid reloading tokenizer on every request
# Opus 4.5 High Priority Fix H1: Bounded cache (was unbounded dict)
@lru_cache(maxsize=5)
def get_tokenizer(model_path: str):
    """Get cached tokenizer or load if not cached.

    Security: Opus 4.5 High Priority Fix H1
    - Bounded cache prevents memory leak from many different models
    - LRU eviction: keeps last 5 tokenizers, evicts least recently used
    - Each tokenizer ~50-200MB, so 5 = ~250MB-1GB max

    Args:
        model_path: HuggingFace model path (e.g. 'mlx-community/Qwen2.5-32B-Instruct-4bit')

    Returns:
        AutoTokenizer instance
    """
    from transformers import AutoTokenizer
    logger.info(f"Loading tokenizer for {model_path}")
    return AutoTokenizer.from_pretrained(model_path)


def clear_tokenizer_cache():
    """Clear tokenizer cache (call on model unload).

    Security: Opus 4.5 High Priority Fix H1
    Allows explicit cache cleanup when models are unloaded.
    """
    get_tokenizer.cache_clear()
    logger.info("Tokenizer cache cleared")


# Request/Response Models (OpenAI-compatible)

class CompletionRequest(BaseModel):
    """OpenAI-compatible completion request."""
    model: str
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 1.0
    repetition_penalty: float = 1.1
    stream: bool = False


class CompletionResponse(BaseModel):
    """OpenAI-compatible completion response."""
    id: str = "cmpl-mlx"
    object: str = "text_completion"
    created: int
    model: str
    choices: list
    usage: Dict[str, int]


# Vision/Multimodal Support - Content Block Models

class TextContent(BaseModel):
    """Text content block for multimodal messages."""
    type: Literal["text"] = "text"
    text: str


class ImageUrlContent(BaseModel):
    """Image URL content block for multimodal messages."""
    type: Literal["image_url"] = "image_url"
    image_url: dict  # {"url": "data:image/jpeg;base64,..." or "https://..."}


# Union type for content blocks
ContentBlock = Union[TextContent, ImageUrlContent]


class ChatMessage(BaseModel):
    """Chat message with optional multimodal content.

    Supports both text-only (backward compatible) and multimodal formats:
    - Text-only: content = "Hello"
    - Multimodal: content = [{"type": "text", "text": "..."}, {"type": "image_url", ...}]
    """
    role: str
    content: Union[str, list[ContentBlock]]  # Backward compatible with text-only


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str
    messages: list[ChatMessage]
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 1.0
    repetition_penalty: float = 1.1
    stream: bool = False


# API Application

def create_app(config: ServerConfig, worker_manager: WorkerManager) -> FastAPI:
    """
    Create FastAPI application.

    Args:
        config: Server configuration
        worker_manager: WorkerManager instance

    Returns:
        FastAPI application
    """
    app = FastAPI(
        title="MLX Server V3",
        description="Production-grade LLM inference server with process isolation",
        version="3.0.0-alpha"
    )

    # OpenAI-compatible error formatting
    def format_openai_error(message: str, error_type: str = "server_error") -> Dict[str, Any]:
        """
        Format error in OpenAI-compatible structure.

        Args:
            message: Error message
            error_type: Error type (server_error, invalid_request_error, authentication_error)

        Returns:
            OpenAI-compatible error object
        """
        return {
            "error": {
                "message": message,
                "type": error_type,
                "param": None,
                "code": None
            }
        }

    # Custom exception handler for OpenAI compatibility
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request, exc: HTTPException):
        """Convert HTTPException to OpenAI-compatible error format."""
        # Map HTTP status codes to OpenAI error types
        error_type = "server_error"
        if exc.status_code == 400:
            error_type = "invalid_request_error"
        elif exc.status_code in (401, 403):
            error_type = "authentication_error"

        return JSONResponse(
            status_code=exc.status_code,
            content=format_openai_error(exc.detail, error_type)
        )

    # Health check
    @app.get("/health")
    async def health_check():
        """Basic health check."""
        return {"status": "healthy", "version": "3.0.0-alpha"}

    # V1 Completions endpoint
    @app.post("/v1/completions")
    async def completions(request: CompletionRequest):
        """
        OpenAI-compatible completions endpoint.

        Args:
            request: Completion request

        Returns:
            Completion response
        """
        try:
            # Check if model is loaded, if not load it
            status = worker_manager.get_status()
            if not status["model_loaded"] or status["model_name"] != request.model:
                logger.info(f"Loading model on-demand: {request.model}")
                load_result = worker_manager.load_model(request.model)
                logger.info(f"Model loaded: {load_result.model_name} ({load_result.memory_gb:.2f} GB)")

            # Create IPC request
            ipc_request = IPCCompletionRequest(
                model=request.model,
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
                stream=request.stream
            )

            # Generate
            if request.stream:
                # Streaming
                async def generate_stream():
                    """Generate SSE stream for text completions."""
                    try:
                        # Use WorkerManager's streaming method (handles SharedMemoryBridge/StdioBridge)
                        for chunk in worker_manager.generate_stream(ipc_request):
                            if chunk.type == "stream_chunk":
                                # Format as SSE for text completions (not chat)
                                data = {
                                    "id": "cmpl-mlx-v3",
                                    "object": "text_completion",
                                    "created": int(__import__('time').time()),
                                    "model": request.model,
                                    "choices": [{
                                        "index": 0,
                                        "text": chunk.text if chunk.text else "",
                                        "finish_reason": chunk.finish_reason
                                    }]
                                }
                                yield f"data: {json.dumps(data)}\n\n"

                                if chunk.done:
                                    yield "data: [DONE]\n\n"
                                    break
                            elif chunk.type == "error":
                                yield f"data: {json.dumps(format_openai_error(chunk.message))}\n\n"
                                break
                    except Exception as e:
                        logger.error(f"Streaming error: {e}")
                        yield f"data: {json.dumps(format_openai_error(str(e)))}\n\n"

                return StreamingResponse(generate_stream(), media_type="text/event-stream")
            else:
                # Non-streaming
                result = worker_manager.generate(ipc_request)

                # Count prompt tokens using tokenizer
                try:
                    tokenizer = get_tokenizer(request.model)
                    prompt_tokens = len(tokenizer.encode(request.prompt))
                except Exception as e:
                    # Can't load tokenizer (e.g., test mock), default to 0
                    logger.warning(f"Could not count prompt tokens: {e}")
                    prompt_tokens = 0
                completion_tokens = result["tokens"]
                total_tokens = prompt_tokens + completion_tokens

                # Format response (OpenAI-compatible)
                import time
                return {
                    "id": "cmpl-mlx-v3",
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{
                        "text": result["text"],
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": result["finish_reason"]
                    }],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens
                    }
                }

        except HTTPException:
            # Re-raise HTTP exceptions (like 501 for streaming)
            raise
        except NoModelLoadedError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except WorkerCommunicationError as e:
            # Buffer full - system overloaded (backpressure)
            logger.warning(f"Backpressure triggered: {e}")
            raise HTTPException(status_code=503, detail=str(e))
        except WorkerError as e:
            logger.error(f"Worker error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    # V1 Chat completions endpoint
    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        """
        OpenAI-compatible chat completions endpoint.

        Args:
            request: Chat completion request

        Returns:
            Chat completion response
        """
        # Convert chat messages to prompt using model's chat template
        logger.info(f"Chat completion request: stream={request.stream}, messages={len(request.messages)}")
        try:
            # Get cached tokenizer (Phase 1.1 optimization)
            tokenizer = get_tokenizer(request.model)

            # Convert messages to dicts for apply_chat_template
            messages_dict = [{"role": msg.role, "content": msg.content} for msg in request.messages]

            # Apply chat template (tokenize=False returns string, not tokens)
            prompt = tokenizer.apply_chat_template(
                messages_dict,
                tokenize=False,
                add_generation_prompt=True
            )
            logger.info(f"Chat template applied, prompt length: {len(prompt)} chars, stream={request.stream}")
            logger.debug(f"Generated prompt: {prompt[:200]}...")
        except Exception as e:
            logger.warning(f"Failed to apply chat template for {request.model}: {e}")
            logger.warning("Falling back to simple concatenation")
            # Fallback to simple concatenation if template fails
            prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in request.messages])

        try:
            # Check if model is loaded, if not load it
            status = worker_manager.get_status()
            if not status["model_loaded"] or status["model_name"] != request.model:
                logger.info(f"Loading model on-demand: {request.model}")
                load_result = worker_manager.load_model(request.model)
                logger.info(f"Model loaded: {load_result.model_name} ({load_result.memory_gb:.2f} GB)")

            # Phase 2: Image preprocessing for vision/multimodal requests
            images = None
            has_images = any(
                isinstance(msg.content, list) and
                any(hasattr(block, 'type') and block.type == 'image_url' for block in msg.content)
                for msg in request.messages
            )

            if has_images:
                logger.info("Vision/multimodal request detected - preprocessing images")
                try:
                    # Collect all content blocks from all messages
                    all_content_blocks = []
                    for msg in request.messages:
                        if isinstance(msg.content, list):
                            all_content_blocks.extend(msg.content)

                    # Get bridge for large images (if using shared memory)
                    bridge = worker_manager.bridge if hasattr(worker_manager, 'bridge') else None

                    # Preprocess images
                    images = await prepare_images(all_content_blocks, bridge=bridge)
                    logger.info(f"Preprocessed {len(images)} images for request")
                except ImportError as e:
                    # Pillow/mlx-vlm not installed
                    logger.warning(f"Missing dependency for vision: {e}")
                    raise HTTPException(status_code=400, detail=str(e))
                except ImageProcessingError as e:
                    logger.warning(f"Image preprocessing failed: {e}")
                    raise HTTPException(status_code=400, detail=f"Image processing error: {e}")

            # Create IPC request
            ipc_request = IPCCompletionRequest(
                model=request.model,
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
                stream=request.stream,
                images=images  # Pass preprocessed images (None for text-only)
            )

            # Generate
            if request.stream:
                # Streaming - chat format
                async def generate_stream():
                    """Generate SSE stream for chat completions."""
                    try:
                        # Use WorkerManager's streaming method (handles SharedMemoryBridge/StdioBridge)
                        for chunk in worker_manager.generate_stream(ipc_request):
                            if chunk.type == "stream_chunk":
                                # Format as SSE for chat completions (use delta.content)
                                data = {
                                    "id": "chatcmpl-mlx-v3",
                                    "object": "chat.completion.chunk",
                                    "created": int(__import__('time').time()),
                                    "model": request.model,
                                    "choices": [{
                                        "index": 0,
                                        "delta": {"content": chunk.text} if chunk.text else {},
                                        "finish_reason": chunk.finish_reason
                                    }]
                                }
                                yield f"data: {json.dumps(data)}\n\n"

                                if chunk.done:
                                    yield "data: [DONE]\n\n"
                                    break
                            elif chunk.type == "error":
                                yield f"data: {json.dumps(format_openai_error(chunk.message))}\n\n"
                                break
                    except Exception as e:
                        logger.error(f"Streaming error: {e}")
                        yield f"data: {json.dumps(format_openai_error(str(e)))}\n\n"

                return StreamingResponse(generate_stream(), media_type="text/event-stream")
            else:
                # Non-streaming
                import time
                start_time = time.time()
                result = worker_manager.generate(ipc_request)
                generation_time = time.time() - start_time

                # Convert to chat format
                response_text = result["text"]
                completion_tokens = result["tokens"]

                # Count prompt tokens using tokenizer (reload if template failed)
                try:
                    prompt_tokens = len(tokenizer.encode(prompt))
                except NameError:
                    # Tokenizer not defined (template failed), try to reload it
                    try:
                        tokenizer = get_tokenizer(request.model)
                        prompt_tokens = len(tokenizer.encode(prompt))
                    except Exception as e:
                        # Can't load tokenizer (e.g., test mock), default to 0
                        logger.warning(f"Could not count prompt tokens: {e}")
                        prompt_tokens = 0
                total_tokens = prompt_tokens + completion_tokens

                tokens_per_sec = completion_tokens / generation_time if generation_time > 0 else 0
                logger.info(f"Chat completion: {len(response_text)} chars, {completion_tokens} tokens in {generation_time:.1f}s ({tokens_per_sec:.1f} tok/s)")
                logger.debug(f"Response content: {response_text[:200]}...")
                return {
                    "id": "chatcmpl-mlx-v3",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response_text
                        },
                        "finish_reason": result["finish_reason"]
                    }],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                        "tokens_per_sec": round(tokens_per_sec, 2)
                    }
                }

        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except NoModelLoadedError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except WorkerCommunicationError as e:
            # Buffer full - system overloaded (backpressure)
            logger.warning(f"Backpressure triggered: {e}")
            raise HTTPException(status_code=503, detail=str(e))
        except WorkerError as e:
            logger.error(f"Worker error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    # Models list endpoint
    @app.get("/v1/models")
    async def list_models():
        """List available models from HuggingFace cache."""
        import os
        from pathlib import Path

        models = []

        # Check HF_HOME or default cache location
        hf_home = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        hub_path = Path(hf_home) / "hub"

        if hub_path.exists():
            # Scan for model directories (format: models--org--model-name)
            for item in hub_path.iterdir():
                if item.is_dir() and item.name.startswith("models--"):
                    # Convert models--mlx-community--Model-Name to mlx-community/Model-Name
                    parts = item.name.split("--", 2)
                    if len(parts) == 3:
                        model_id = f"{parts[1]}/{parts[2]}"
                        # Only include mlx-community models
                        if parts[1] == "mlx-community":
                            models.append({
                                "id": model_id,
                                "object": "model",
                                "created": int(item.stat().st_mtime),
                                "owned_by": parts[1]
                            })

        # Sort by creation time (most recent first)
        models.sort(key=lambda x: x["created"], reverse=True)

        return {
            "object": "list",
            "data": models
        }

    return app


def create_admin_app(config: ServerConfig, worker_manager: WorkerManager) -> FastAPI:
    """
    Create admin API application.

    Args:
        config: Server configuration
        worker_manager: WorkerManager instance

    Returns:
        FastAPI application for admin endpoints
    """
    app = FastAPI(
        title="MLX Server V3 - Admin API",
        description="Administrative endpoints for model management",
        version="3.0.0-alpha"
    )

    @app.get("/admin/health")
    async def admin_health():
        """Admin health check with worker status."""
        health = worker_manager.health_check()
        return {
            "status": "healthy" if health["healthy"] else "degraded",
            "worker_status": health["status"],
            "version": "3.0.0-alpha"
        }

    @app.get("/admin/status")
    async def admin_status():
        """Get server and worker status."""
        status = worker_manager.get_status()
        health = worker_manager.health_check()

        return {
            "status": "running",
            "version": "3.0.0-alpha",
            "ports": {
                "main": config.main_port,
                "admin": config.admin_port
            },
            "model": {
                "loaded": status["model_loaded"],
                "name": status["model_name"],
                "memory_gb": status["memory_gb"]
            },
            "worker": {
                "healthy": health["healthy"],
                "status": health["status"]
            },
            "config": {
                "machine_type": config.machine_type,
                "total_ram_gb": config.total_ram_gb,
                "idle_timeout_seconds": config.idle_timeout_seconds
            }
        }

    @app.get("/admin/capabilities")
    async def admin_capabilities():
        """
        Get model capabilities (text/vision support).

        Returns:
            Model capabilities and mlx-vlm availability
        """
        status = worker_manager.get_status()
        model_name = status["model_name"]

        # Detect vision capability from model name
        vision_patterns = ["qwen2-vl", "qwen2.5-vl", "-vl-", "llava", "idefics"]
        is_vision = False
        detection_method = "model_name"

        if model_name:
            model_lower = model_name.lower()
            is_vision = any(pattern in model_lower for pattern in vision_patterns)
        else:
            detection_method = "no_model_loaded"

        # Check mlx-vlm availability
        mlx_vlm_available = False
        try:
            import mlx_vlm
            mlx_vlm_available = True
        except ImportError:
            pass

        return {
            "model": model_name,
            "capabilities": {
                "text": True,  # All models support text
                "vision": is_vision
            },
            "detection_method": detection_method,
            "mlx_vlm_available": mlx_vlm_available,
            "notes": (
                "Vision models require mlx-vlm package. "
                "Install: pip install mlx-vlm pillow"
            ) if is_vision and not mlx_vlm_available else None
        }

    @app.post("/admin/load")
    async def admin_load(model_path: str):
        """
        Load a model.

        Args:
            model_path: HuggingFace model path

        Returns:
            Load result
        """
        try:
            result = worker_manager.load_model(model_path)
            return {
                "status": "success",
                "model_name": result.model_name,
                "memory_gb": result.memory_gb,
                "load_time": result.load_time
            }
        except Exception as e:
            logger.error(f"Load failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/admin/unload")
    async def admin_unload():
        """
        Unload current model (terminate worker).

        Returns:
            Unload result with memory freed
        """
        try:
            result = worker_manager.unload_model()
            return result.to_dict()
        except NoModelLoadedError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Unload failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return app
