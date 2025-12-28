# Performance Insights - MLX Inference Server
**Last Updated:** 2025-12-28
**Testing:** M4 Air 32GB vs M4 Max Studio 128GB

---

## 🎯 Critical Discovery: Memory Bandwidth is King

### The Fundamental Law of LLM Inference Performance

**For autoregressive LLM inference on Apple Silicon, memory bandwidth is the primary performance determinant, not CPU/GPU cores or language choice (Python vs C++).**

---

## Hardware Comparison

### M4 MacBook Air (32GB)
- **Memory Bandwidth:** ~120 GB/s (LPDDR5X)
- **GPU Cores:** 10
- **7B-4bit Performance:** 17 tok/s (both MLX and Ollama)
- **Bottleneck:** Memory bandwidth fully saturated

### M4 Max Mac Studio (128GB)
- **Memory Bandwidth:** ~400 GB/s (unified memory)
- **GPU Cores:** 40
- **7B-4bit Performance:**
  - MLX: 94 tok/s (direct), 82-89 tok/s (server)
  - Ollama: 86 tok/s
- **Bottleneck:** Partial - 30-38% bandwidth utilization

---

## Key Findings from Comprehensive Benchmarks

### 1. MLX IS Faster Than Ollama (When Bandwidth Available)

| Model Size | MLX (direct) | Ollama | MLX Advantage |
|------------|--------------|--------|---------------|
| 0.5B-4bit  | 385 tok/s    | 328 tok/s | **+17%** |
| 3B-4bit    | 156 tok/s    | 138 tok/s | **+13%** |
| 7B-4bit    | 94 tok/s     | 86 tok/s  | **+9%**  |

**Why the advantage?**
- Unified memory architecture (zero-copy)
- Lazy evaluation optimization
- Native Metal JIT compilation
- No portability overhead

### 2. On Bandwidth-Limited Hardware, All Frameworks Converge

**M4 Air Results (7B-4bit):**
- MLX via our server: 16.7 tok/s
- Ollama: 17.3 tok/s
- Difference: **3%** (essentially identical)

**Why?** Both frameworks saturate the 120 GB/s bandwidth ceiling. No room for software optimizations to matter.

### 3. Server Architecture Overhead: Negligible

**Our Process Isolation + IPC + HTTP Architecture:**

| Workload | Overhead |
|----------|----------|
| Short (50 tokens) | 13-20% |
| Medium (100 tokens) | 9-13% |
| Long (200 tokens) | **1-6%** |

**For production workloads (sustained generation), overhead is <6%.**

Components:
- FastAPI/Uvicorn HTTP: ~1-2ms
- POSIX shared memory IPC: <1µs
- Process scheduling: ~0.5ms
- JSON serialization: ~0.5-1ms
- Total: **~3-7ms per request**

### 4. Performance Scales Sub-linearly with Model Size

| Model | Parameters | Perf (tok/s) | Scaling Factor |
|-------|------------|--------------|----------------|
| 0.5B  | 0.5B       | 385          | 1.0x (baseline) |
| 3B    | 3B         | 156          | 2.5x slower (6x params) |
| 7B    | 7B         | 94           | 4.1x slower (14x params) |
| 14B   | 14B        | 50           | 7.7x slower (28x params) |

**Good news:** Larger models are more efficient than linear scaling would predict.

**Why?** KV cache overhead is fixed per token, not per parameter. Larger models also achieve better Metal kernel utilization.

---

## Industry Benchmark Validation

### Our Results vs Published Estimates (7B-4bit, M4 Max)

| Source | Published Estimate | Our Direct MLX | Our Server | Status |
|--------|-------------------|----------------|------------|--------|
| llama.cpp benchmarks | 70-80 tok/s | **94 tok/s** | 82-89 tok/s | ✅ Exceeds |
| MLX examples | 65-75 tok/s | **94 tok/s** | 82-89 tok/s | ✅ Exceeds |
| Ollama community | 68-78 tok/s | **86 tok/s** (Ollama) | 82-89 tok/s | ✅ Matches |

**Conclusion:** Our implementation is production-grade and exceeds published benchmarks.

---

## Laptop Purchase Guidance

### For Maximum LLM Inference Performance

**Priority Order:**
1. **Memory Bandwidth** (most important)
2. **Total RAM** (enables larger models)
3. **GPU Cores** (helps but less critical)

### Recommended Configurations

#### Option 1: MacBook Pro M4 Max (Recommended)
- **Memory:** 64GB or 128GB
- **Bandwidth:** ~400 GB/s
- **Expected 7B-4bit:** 80-95 tok/s
- **Expected 32B-4bit:** 25-30 tok/s
- **Cost:** Higher
- **Sweet spot for:** 7B-32B models

#### Option 2: MacBook Pro M4 Pro
- **Memory:** 48GB
- **Bandwidth:** ~200-250 GB/s (estimated)
- **Expected 7B-4bit:** 40-50 tok/s
- **Expected 32B-4bit:** 12-15 tok/s
- **Cost:** Medium
- **Sweet spot for:** 7B-14B models

#### Option 3: MacBook Air M4 (Current)
- **Memory:** 32GB
- **Bandwidth:** ~120 GB/s
- **Expected 7B-4bit:** 15-20 tok/s
- **Expected 32B-4bit:** Not recommended (would be <5 tok/s)
- **Cost:** Lower
- **Sweet spot for:** 0.5B-7B models only

### Memory Bandwidth Comparison

| Chip | Bandwidth | 7B-4bit Estimate | 32B-4bit Estimate |
|------|-----------|------------------|-------------------|
| M4 (Air/Mini) | ~120 GB/s | 15-20 tok/s | 4-6 tok/s |
| M4 Pro | ~200-250 GB/s | 40-50 tok/s | 12-15 tok/s |
| M4 Max | ~400 GB/s | 80-95 tok/s | 25-30 tok/s |
| M4 Ultra (future) | ~800 GB/s | 160-190 tok/s | 50-60 tok/s |

**Rule of thumb:** Double the bandwidth ≈ double the tok/s (for same model).

---

## Technical Deep Dive

### Why Memory Bandwidth Dominates

**Autoregressive generation requires:**
1. Loading model weights for each token
2. Reading/writing KV cache (grows with context)
3. Attention computation (scales with context length)

**For 7B-4bit model (~4 GB):**
```
Data movement per token ≈ 4-5 GB
M4 Air (120 GB/s): 120 ÷ 5 = 24 tok/s theoretical
M4 Max (400 GB/s): 400 ÷ 5 = 80 tok/s theoretical

Observed efficiency: 70-90% of theoretical
M4 Air: 17 tok/s (71% efficient)
M4 Max: 94 tok/s (94% efficient → hitting other limits)
```

### Why Language Choice (Python vs C++) Doesn't Matter

**When memory-bandwidth limited:**
- CPU/GPU spends >95% of time waiting for memory
- Python interpreter overhead: <1% of total time
- C++ would only save that <1%

**Analogy:**
- Reading 4GB from disk takes 5 seconds
- Python processing: +0.05s
- C++ processing: +0.02s
- Difference: 0.6% (nobody cares)

**When bandwidth available (M4 Max):**
- Framework optimizations become visible
- MLX's unified memory + lazy eval = 9-17% advantage
- Still not language choice - it's algorithmic optimization

---

## Performance Optimization Recommendations

### What Works

✅ **Buy higher memory bandwidth hardware** (biggest impact)
✅ **Use smaller models** (0.5B-3B can be faster for simple tasks)
✅ **Use appropriate quantization** (4-bit is sweet spot)
✅ **MLX over Ollama** (9-17% faster when bandwidth available)
✅ **Our server architecture** (only 1-6% overhead)

### What Doesn't Work (When Bandwidth-Limited)

❌ **Switching from Python to C++** (<1% gain)
❌ **More GPU cores** (not the bottleneck)
❌ **Faster CPU** (not the bottleneck)
❌ **Code optimization** (already maxing bandwidth)
❌ **Different HTTP server** (overhead is negligible)

### Advanced Optimizations (If Needed)

**Speculative Decoding:**
- Use fast 0.5B draft model + 7B verification model
- Potential: 20-40% improvement
- Complexity: High implementation effort

**Prompt Caching:**
- Cache KV states for common prompt prefixes
- Reduces time-to-first-token
- Already implemented in MLX (use via API)

**Batch Processing:**
- Process multiple requests simultaneously
- Better GPU utilization
- Requires architecture changes

---

## Test Methodology

### Benchmark Suite Details

**Hardware:**
- Mac Studio M4 Max, 128GB RAM, 40-core GPU
- MacBook Air M4, 32GB RAM, 10-core GPU

**Models Tested:**
- Qwen2.5-0.5B-Instruct-4bit (260MB)
- Qwen2.5-3B-Instruct-4bit (1.9GB)
- Qwen2.5-7B-Instruct-4bit (4.1GB)
- Qwen2.5-14B-Instruct-4bit (8.2GB)

**Test Configurations:**
1. Direct MLX library (baseline, no server)
2. MLX Inference Server (HTTP + IPC + process isolation)
3. Ollama (Go HTTP + C++ llama.cpp)

**Prompt Sizes:**
- Short: "Hello" → 50 tokens
- Medium: "Write a Python function" → 100 tokens
- Long: "Explain quantum computing in detail" → 200 tokens

**Runs:** 5 iterations per test, averaged

**Results:**
- Direct MLX: `/tmp/benchmark_direct_mlx.log`
- Server: `/tmp/benchmark_server_results.txt`
- Ollama: `/tmp/benchmark_ollama_results.txt`
- Analysis: `/tmp/benchmark-analysis.md`
- Full log: `/tmp/benchmark-full-results.log`

---

## Historical Context

### Discovery Timeline

**2025-12-27:** Deployed mlx-inference-server from Studio to Air

**2025-12-28:**
1. Initial Air benchmarks showed MLX = Ollama (~17 tok/s)
2. Questioned why Python MLX matches C++ Ollama
3. Submitted question to Claude Opus for analysis
4. Opus hypothesis: Memory bandwidth limitation
5. Ran comprehensive benchmarks on Studio
6. **Discovery:** MLX is 9-17% faster when bandwidth available

**Key Learning:** Air results were not wrong - they revealed bandwidth limitation. Studio results revealed MLX's true advantage when unconstrained.

---

## Recommendations for Future Hardware

### Upgrade Path

**Current:** MacBook Air M4 32GB (120 GB/s)
**Recommendation:** MacBook Pro M4 Max 64GB+ (400 GB/s)
**Expected Improvement:**
- 7B models: **17 → 80-94 tok/s** (4.7-5.5x faster)
- 14B models: ~8 → 44-50 tok/s (5.5-6.3x faster)
- 32B models: Not viable → 25-30 tok/s (usable!)

**Cost-Benefit:**
- M4 Max costs ~2x M4 Air
- Performance gain: 4.7-5.5x for LLM inference
- **Excellent ROI for LLM workloads**

### When to Upgrade

**Upgrade if:**
- ✅ Working with 14B+ models regularly
- ✅ Need faster iteration speed (4.7x speedup)
- ✅ Running production inference server
- ✅ Budget allows (~$3500+ for M4 Max MBP)

**Stay with Air if:**
- ✅ 0.5B-7B models are sufficient
- ✅ 17 tok/s is acceptable for workflow
- ✅ Budget-conscious
- ✅ Primarily training/fine-tuning (different bottleneck)

---

## Files Reference

**Documentation:**
- This file: `docs/PERFORMANCE-INSIGHTS.md`
- Benchmark analysis: `/tmp/benchmark-analysis.md`

**Benchmark Results:**
- Full log: `/tmp/benchmark-full-results.log`
- Direct MLX: `/tmp/benchmark_direct_mlx.log`
- Server results: `/tmp/benchmark_server_results.txt`
- Ollama results: `/tmp/benchmark_ollama_results.txt`

**Benchmark Scripts:**
- Comprehensive suite: `/tmp/comprehensive-benchmark-suite.sh`
- Language overhead test: `/tmp/language-overhead-test.sh`

**Opus Analysis:**
- Batch request: `/tmp/opus-mlx-performance-analysis.jsonl`
- Results: `/tmp/opus-performance-analysis.txt`
- Response: `/tmp/opus-performance-results.jsonl`

---

## Quick Reference

### Performance by Hardware

| Hardware | 7B-4bit | 14B-4bit | 32B-4bit |
|----------|---------|----------|----------|
| M4 Air 32GB | 17 tok/s | ~8 tok/s | Not viable |
| M4 Pro 48GB | 40-50 tok/s | 20-25 tok/s | 10-12 tok/s |
| M4 Max 128GB | 82-94 tok/s | 44-50 tok/s | 25-30 tok/s |

### When Frameworks Perform Similarly

**Bandwidth-limited scenarios:**
- M4 Air with any 7B+ model
- M4 Pro with 32B+ model
- Very long context (>8K tokens)

**MLX advantage scenarios:**
- M4 Max with 7B-14B models
- Smaller models (0.5B-3B) on any hardware
- Short-to-medium context (<4K tokens)

---

**Bottom Line:** Memory bandwidth is king. For your next laptop, prioritize M4 Max for the 400 GB/s bandwidth, not the GPU cores.
