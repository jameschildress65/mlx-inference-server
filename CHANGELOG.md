# Changelog

All notable changes to MLX Inference Server will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- **Request parameter logging** - Enhanced debugging and monitoring
  - Log max_tokens, temperature, top_p, repetition_penalty for all requests
  - Helps diagnose timeout issues and client behavior
  - Files changed: `src/orchestrator/api.py`

- **HF_HOME fallback in daemon script** - Robust model cache configuration
  - Daemon script now sets HF_HOME with fallback to `~/.cache/huggingface`
  - Works across manual starts, launchd, cron scenarios
  - Each machine can override via shell config
  - Files changed: `bin/mlx-inference-server-daemon.sh`

### Changed
- **Version updated to 3.1.0** - Proper semantic versioning
  - Updated from development version to production release
  - Files changed: `src/orchestrator/api.py`
  - Tests updated: `tests/unit/test_api_v3.py`

### Fixed
- **Vision API image threshold bug** - Images ≥500KB now work properly
  - Increased INLINE_THRESHOLD from 500KB to 10MB
  - Added public `bridge` property to WorkerManager for image processing
  - Vision API now handles all images up to maximum 10MB size limit
  - Files changed: `src/orchestrator/image_utils.py`, `src/orchestrator/worker_manager.py`
  - Commit: bf47a28

- **Chat template warning for vision models** - Eliminated type mismatch warning
  - Fixed multimodal content handling in chat completions endpoint
  - Now properly extracts text from content blocks before applying chat template
  - No more "can only concatenate str (not 'list') to str" warnings
  - Files changed: `src/orchestrator/api.py`

### Technical Details
**Image Threshold Fix:**
- Previous behavior: Images ≥500KB tried to use shared memory but bridge was inaccessible
- Error message: "Large image requires shared memory bridge, but none provided"
- Root cause: No public property to access internal `_shmem_bridge` in WorkerManager
- Solution: Increased threshold to 10MB (matches max image size), added bridge property
- Impact: Simpler architecture - all typical images use inline base64, no shared memory needed

**Chat Template Fix:**
- Previous behavior: Vision requests with multimodal content triggered type warning
- Cause: Content list passed directly to tokenizer expecting string
- Solution: Preprocess content to extract text before applying chat template
- Impact: Clean logs, no functional change (fallback already worked)

---

## [3.1.1] - 2026-01-05

### Fixed
- **Critical: Orchestrator hang after worker timeout** - Prevents 1200s deadlock
  - Worker timeouts now trigger immediate cleanup instead of stdio fallback
  - Prevents double-timeout scenario (600s shared memory + 600s stdio = 1200s hang)
  - Orchestrator remains responsive after worker timeout, returns error to client
  - New `WorkerTimeoutError` exception type for precise timeout handling
  - Force-kill hung workers with `_force_kill_hung_worker()` method
  - Files changed: `src/orchestrator/worker_manager.py`, `src/ipc/shared_memory_bridge.py`, `src/ipc/stdio_bridge.py`
  - Tests added: `tests/unit/test_worker_timeout_fix.py` (7 unit tests)
  - Commit: f0d1c91

### Technical Details
**Timeout Fix:**
- Previous behavior: Worker timeout → stdio fallback → second 600s timeout → orchestrator deadlock
- Root cause: Exception fallback logic treated timeout as transient IPC error
- Solution: Distinguish `WorkerTimeoutError` from generic `WorkerCommunicationError`
- Timeout handling: Catch timeout specifically, force-kill worker, skip stdio retry
- Non-timeout IPC errors: Still fall back to stdio (preserves resilience for transient errors)
- Impact: Critical production bug eliminated, orchestrator never hangs on worker timeout
- Opus-reviewed and approved approach using exception type hierarchy

---

## [3.0.0] - 2025-01-02

### Added
- **Robust worker lifecycle management** with ProcessRegistry
  - Crash-safe worker tracking in persistent file (`/tmp/mlx-server/worker_registry.json`)
  - Automatic orphan cleanup on server restart
  - Prevents memory leaks from abandoned worker processes
  - File-locking for multi-process safety

- **Vision model support** (Phase 3)
  - Dual-venv architecture: text models (mlx-lm) and vision models (mlx-vlm + PyTorch)
  - Automatic model capability detection
  - OpenAI-compatible multimodal API
  - Image preprocessing with security hardening (SSRF protection, size limits)
  - Support for base64 data URLs, HTTP/HTTPS URLs, and local file paths

- **Shared memory IPC** for large data transfers
  - Efficient image passing between orchestrator and worker
  - POSIX shared memory with file locking
  - Automatic fallback to stdio for compatibility

### Changed
- **Production-ready installation** with automated `install.sh`
  - One-command deployment
  - Comprehensive testing during installation
  - RAM tier detection (16GB/32GB/64GB+)

- **Process naming** for easy identification
  - Workers show as `mlx-server-v3-<id>` in process listings

### Security
- SSRF protection for image URLs (blocks private IP ranges)
- Image size limits (10MB per image, 5 images max)
- Dimension limits to prevent decompression bombs (4096x4096 max)
- Model path validation (HuggingFace org/model format only)
- Bounded tokenizer cache (LRU, max 5 tokenizers)

### Performance
- On-demand model loading with idle timeout
- Tokenizer caching to avoid reloads
- Shared memory for efficient image transfer
- Metal shader cache enabled
- Optimized malloc for large allocations

---

## [2.0.0] - 2024-12-XX

### Added
- Initial release with worker process isolation
- OpenAI-compatible API endpoints
- Text completion and chat completion support
- Streaming responses
- Health check endpoints

---

## [1.0.0] - 2024-11-XX

### Added
- Initial proof-of-concept implementation
- Basic MLX model loading
- Simple HTTP API

---

**Note:** Dates in YYYY-MM-DD format. Unreleased changes are tracked at the top.
