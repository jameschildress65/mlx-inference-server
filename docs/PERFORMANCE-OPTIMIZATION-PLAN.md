# Performance Optimization Plan - Apple Silicon
**Branch:** `phase2-performance-apple-silicon`
**Based on:** Opus 4.5 Performance Review (2025-12-25)
**Target:** 20.8 tok/s → 23.5-24.0 tok/s (beat Ollama's 22.1 tok/s)

---

## Current State

**Baseline Performance (Qwen2.5-32B-Instruct-4bit on M4 Max):**
- Native MLX: 24.7 tok/s (0% overhead)
- **MLX Server V3:** 20.8 tok/s (16% overhead) ← CURRENT
- Ollama: 22.1 tok/s (10% overhead) ← TARGET TO BEAT

**Opus Grade:** C+ (functional but leaving Apple Silicon performance on table)

---

## Root Cause Analysis (from Opus 4.5)

**Overhead Breakdown (REVISED):**
| Component | Previous Estimate | Actual Impact | Fix |
|-----------|-------------------|---------------|-----|
| **Synchronous flush() blocking** | 0% | **5-7%** | Async batching |
| **IPC (stdin/stdout)** | 8-10% | **4-5%** | Shared memory |
| **Python↔MLX boundary** | 0% | **2-3%** | Reduce crossings |
| **JSON serialization** | 3-4% | **1-2%** | ✅ Not a problem |
| **API + logging** | 2-3% | **1-2%** | ✅ Acceptable |

**Primary Bottleneck:** Synchronous `flush()` on each token stalls GPU pipeline

---

## Implementation Phases

### Phase 1: Quick Wins (Target: 22+ tok/s, ~4-5% gain)

#### 1.1 MLX Environment Variables ✅
**Files:** `src/worker/__main__.py`
**Expected gain:** 1-2%
**Complexity:** Low

```python
# Set BEFORE importing mlx
os.environ['MTL_SHADER_CACHE_ENABLE'] = '1'
os.environ['PYTHONMALLOC'] = 'malloc'
```

#### 1.2 Model Loading Optimization ✅
**Files:** `src/worker/model_loader.py`
**Expected gain:** Startup time + stability
**Complexity:** Medium

- Clear Metal cache before load
- Pre-evaluate parameters hierarchically
- Warm up with dummy forward pass (compile Metal shaders)
- Clear warmup from KV cache

#### 1.3 Async Token Batching ✅
**Files:** `src/worker/inference.py`
**Expected gain:** 3-4%
**Complexity:** Medium

- Batch 4 tokens before flushing to IPC
- Prevents GPU stalls from synchronous writes
- Maintains streaming UX (chunked responses)

**After Phase 1:** Benchmark → should hit 22+ tok/s

---

### Phase 2: IPC Overhaul (Target: 23.5+ tok/s, ~4-5% additional gain)

#### 2.1 POSIX Shared Memory IPC
**Files:** `src/ipc/shared_memory_bridge.py` (new)
**Expected gain:** 4-5%
**Complexity:** High

- Lock-free ring buffers
- Zero-copy on Apple Silicon unified memory
- Sub-microsecond latency (vs 50-100µs for stdin/stdout)

#### 2.2 Replace StdioBridge
**Files:** `src/orchestrator/worker_manager.py`, `src/worker/__main__.py`
**Expected gain:** (included above)
**Complexity:** Medium

- Migrate from stdin/stdout to shared memory
- Preserve process isolation
- Backward compatibility for rollback

**After Phase 2:** Benchmark → should hit 23.5-24.0 tok/s (match native MLX)

---

## Success Criteria

### Phase 1 Complete
- ✅ Performance: ≥22 tok/s (match/beat Ollama)
- ✅ All existing tests passing
- ✅ No regressions in stability
- ✅ Benchmark data collected

### Phase 2 Complete
- ✅ Performance: ≥23.5 tok/s (approach native MLX)
- ✅ Overhead reduced to <5%
- ✅ Process isolation maintained
- ✅ Production-ready stability

---

## Apple Silicon Optimizations Applied

### Unified Memory Architecture
- Zero-copy IPC via shared memory (Phase 2)
- Direct GPU↔CPU access to same physical pages
- Maximize 400 GB/s bandwidth (M4 Max)

### Metal Framework
- Shader cache enabled
- Pre-compilation on model load
- Optimized memory allocator

### M4-Specific
- Thermal monitoring (MacBook Air)
- Memory pressure detection
- Performance core utilization

---

## Risk Mitigation

### Phase 1 (Low Risk)
- All changes are additive
- Easy rollback if issues
- No architectural changes

### Phase 2 (Medium Risk)
- Major IPC change
- Requires careful testing
- Preserve fallback to stdin/stdout
- Incremental migration path

---

## Testing Strategy

### Benchmark Suite
1. Single request latency
2. Throughput (requests/second)
3. Token generation speed (tok/s)
4. Memory usage
5. Thermal behavior (Air)

### Models to Test
- Qwen2.5-7B-Instruct-4bit (baseline, fast)
- Qwen2.5-32B-Instruct-4bit (primary target)
- Qwen2.5-72B-Instruct-4bit (memory stress)

### Platforms to Test
- M4 Max (128GB) - primary
- M4 Air (32GB) - thermal concerns
- M4 Mini (16GB) - memory pressure

---

## Implementation Log

### 2025-12-25

**Phase 1 Complete ✅**

- ✅ Opus 4.5 performance review completed
- ✅ Branch created: `phase2-performance-apple-silicon`
- ✅ MLX environment variables implemented
- ✅ Model loading optimization implemented
- ✅ Async token batching implemented
- ✅ Benchmarks completed: **22.41 tok/s** (target exceeded!)
- ✅ All Phase 1 optimizations working

**Results:**
- Baseline: 20.8 tok/s (16% overhead)
- Phase 1: **22.41 tok/s (9.3% overhead)**
- Improvement: +7.7% (+1.61 tok/s)
- Status: Beat Ollama (22.1 tok/s) ✅

**Phase 2 Ready:** POSIX shared memory IPC for final 4-5% gain

---

## References

- Opus 4.5 Performance Review: `/tmp/opus-performance-review.md`
- Baseline benchmarks: `/tmp/performance-analysis-final.md`
- Phase 2.1 critical fixes: `docs/PHASE2-IMPLEMENTATION-PLAN.md`

---

**Document Version:** 1.0
**Status:** In Progress - Phase 1
**Next Milestone:** 22+ tok/s (beat Ollama)
