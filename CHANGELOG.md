# Changelog

All notable changes to MLX Inference Server will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

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
