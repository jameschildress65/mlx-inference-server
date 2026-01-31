# Changelog

All notable changes to MLX Inference Server will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

*No unreleased changes.*

---

## [3.2.0] - 2026-01-30

### Added
- **JSON Mode (Structured Output)** - OpenAI-compatible response_format field
  - `response_format: {type: "json_object"}` for generic JSON output
  - `response_format: {type: "json_schema", json_schema: {...}}` for schema-constrained output
  - Uses Outlines library for constrained decoding via logits processing
  - Bounded LRU cache (10 entries) for JSON processors
  - Schema validation: 64KB size limit, 20 levels max depth
  - Files changed: `src/orchestrator/api.py`, `src/ipc/messages.py`, `src/worker/inference.py`
  - Tests: `tests/unit/test_json_mode.py` (27 tests)

- **Priority Request Queue** - Fair scheduling with priority levels
  - Three priority levels: HIGH, NORMAL, LOW (set via X-Priority header)
  - Configurable timeouts per priority (default: 300s/120s/60s)
  - Queue depth limits with reject threshold
  - Prometheus metrics integration
  - Files changed: `src/orchestrator/priority_queue.py`, `src/orchestrator/api.py`
  - Tests: `tests/unit/test_priority_queue.py`, `tests/integration/test_queue_integration.py`

- **Rate Limiting** - Sliding window rate limiter (optional)
  - Global request rate limiting (disabled by default)
  - Configurable RPM and burst size
  - Enable with `MLX_RATE_LIMIT_ENABLED=1`
  - Files changed: `src/orchestrator/rate_limiter.py`

- **Prometheus Metrics** - Production monitoring endpoint
  - `/metrics` endpoint with request counts, latencies, queue stats
  - Model loading metrics, GPU memory tracking
  - Files changed: `src/orchestrator/prometheus_metrics.py`

### Changed
- **Version updated to 3.2.0**

### Fixed
- **Vision backend TypeError** - VisionInferenceBackend now accepts JSON mode params
  - Prevents TypeError on all vision model requests
  - JSON mode params are accepted but ignored (not supported for vision)

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
