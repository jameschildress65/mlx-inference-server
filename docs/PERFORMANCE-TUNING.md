# MLX Server V3 - Performance Tuning Guide

**Version**: 3.0.0-alpha
**Date**: 2025-12-24

---

## Performance Baselines

### Expected Performance (Qwen2.5-0.5B-Instruct-4bit)

| Hardware | Throughput | First Token | Memory |
|----------|------------|-------------|---------|
| M1 (16GB) | 80-100 tok/s | ~50ms | 260 MB |
| M2 (32GB) | 100-120 tok/s | ~40ms | 260 MB |
| M3 (32GB) | 120-150 tok/s | ~35ms | 260 MB |
| M4 Max (128GB) | 150-180 tok/s | ~30ms | 260 MB |

**V3 Overhead**: ~1-2ms per request (IPC)

---

## Optimization Strategies

### 1. Model Selection

**For Speed**:
- Smaller models (0.5B-1.5B) = faster
- 4-bit quantization = 4x smaller, minimal quality loss

**For Quality**:
- Larger models (7B-14B) = better outputs
- Trade-off: 2-3x slower

**Recommendation**:
- Development: 0.5B-4bit (fast iteration)
- Production: 7B-4bit (quality)
- Heavy workloads: 14B-4bit (if RAM allows)

### 2. Pre-loading

**Problem**: First request triggers model load (2-5s latency)

**Solution**:
```bash
# Pre-load on server start
curl -X POST "http://localhost:11441/admin/load?model_path=mlx-community/Qwen2.5-7B-Instruct-4bit"
```

**Auto-load Script**:
```bash
#!/bin/bash
# auto-load.sh

# Wait for server to start
sleep 5

# Load default model
curl -X POST "http://localhost:11441/admin/load?model_path=mlx-community/Qwen2.5-7B-Instruct-4bit"
```

### 3. Idle Timeout Tuning

**Trade-off**: Memory vs Latency

**Short Timeout (600s)**: Frees memory quickly, but reloads often
**Long Timeout (3600s)**: Keeps model loaded, no reload latency

**Adjust**:
```python
# config/server_config.py
config.idle_timeout_seconds = 1800  # 30 minutes
```

### 4. Request Parameters

**Temperature**:
- Lower (0.1-0.3): Faster, deterministic
- Higher (0.7-1.0): Slower, creative

**Max Tokens**:
- Fewer tokens = faster response
- Use minimum needed

**Optimal**:
```json
{
  "temperature": 0.3,
  "max_tokens": 50,
  "top_p": 1.0
}
```

---

## Benchmarking

### Throughput Test

```bash
#!/bin/bash
# throughput-test.sh

MODEL="mlx-community/Qwen2.5-0.5B-Instruct-4bit"

# Load model
curl -X POST "http://localhost:11441/admin/load?model_path=$MODEL"
sleep 2

# Test 10 requests
TOTAL_TOKENS=0
TOTAL_TIME=0

for i in {1..10}; do
  START=$(date +%s.%N)

  RESPONSE=$(curl -s -X POST http://localhost:11440/v1/completions \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"$MODEL\",
      \"prompt\": \"Test $i:\",
      \"max_tokens\": 50,
      \"temperature\": 0.3
    }")

  END=$(date +%s.%N)
  TIME=$(echo "$END - $START" | bc)

  TOKENS=$(echo $RESPONSE | jq -r '.usage.completion_tokens')

  TOTAL_TOKENS=$((TOTAL_TOKENS + TOKENS))
  TOTAL_TIME=$(echo "$TOTAL_TIME + $TIME" | bc)

  echo "Request $i: $TOKENS tokens in ${TIME}s"
done

THROUGHPUT=$(echo "scale=2; $TOTAL_TOKENS / $TOTAL_TIME" | bc)
echo ""
echo "Total: $TOTAL_TOKENS tokens in ${TOTAL_TIME}s"
echo "Throughput: $THROUGHPUT tokens/sec"
```

### Latency Test

```bash
#!/bin/bash
# latency-test.sh

# Test first-token latency
curl -X POST http://localhost:11440/v1/completions \
  -H "Content-Type: application/json" \
  -w "\nTime: %{time_total}s\n" \
  -d '{
    "model": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    "prompt": "Count to 3:",
    "max_tokens": 10,
    "temperature": 0.1
  }'
```

---

## Bottleneck Analysis

### CPU Bottleneck

**Symptoms**:
- High CPU usage (>80%)
- Slow generation

**Check**:
```bash
top -o CPU
```

**Solutions**:
- Close other applications
- Use smaller model
- Reduce temperature (less sampling)

### Memory Bottleneck

**Symptoms**:
- Swap usage
- Slow performance

**Check**:
```bash
vm_stat | grep "Pages active"
```

**Solutions**:
- Use smaller model
- Close other applications
- Increase RAM (hardware upgrade)

### Disk I/O Bottleneck

**Symptoms**:
- First load slow (only)

**Check**:
```bash
iostat -w 1
```

**Solutions**:
- Use SSD (not HDD)
- Cache models on fast storage
- Pre-download models

---

## Performance Comparison

### V2 vs V3

**Expected**:
- Throughput: Similar (within 5%)
- Latency: V3 +1-2ms (IPC overhead)
- Memory: V3 much better (0 leaks)

**Test**:
```bash
# V2
time ./test-v2.sh

# V3
time ./test-v3.sh

# Compare results
```

---

## Advanced Optimizations

### 1. Model Quantization

**Options**:
- 2-bit: Smallest, lowest quality
- 4-bit: Good balance (recommended)
- 8-bit: Larger, better quality

**Performance**:
- 4-bit vs 8-bit: 2x faster, similar quality

### 2. Batch Size (Future)

**Current**: V3 processes 1 request at a time
**Future**: Phase 4 worker pool support

### 3. Prompt Caching (Future)

**Current**: Each request independent
**Future**: Cache common prefixes

---

## Monitoring Performance

### Key Metrics

```bash
# Throughput (tokens/sec)
grep "tokens" logs/mlx-inference-server.log | awk '{print $X}' | stats

# Latency (seconds)
grep "latency" logs/mlx-inference-server.log | awk '{print $X}' | stats

# Memory usage
while true; do
  ps aux | grep mlx_inference_server | awk '{print $6/1024 " MB"}'
  sleep 10
done
```

### Grafana Dashboard (Future)

**Phase 6**: Prometheus metrics export
- Request rate
- Latency percentiles (p50, p95, p99)
- Throughput
- Memory usage

---

## Performance Troubleshooting

### Issue: Slow Generation

1. Check model size
2. Check CPU usage
3. Check temperature setting
4. Test with smaller model

### Issue: High Latency

1. Pre-load model
2. Reduce max_tokens
3. Lower temperature
4. Check network (if remote)

### Issue: Memory Pressure

1. Use smaller model
2. Close other apps
3. Increase idle timeout
4. Monitor with Activity Monitor

---

**Document Version**: 1.0
**Last Updated**: 2025-12-24
**Maintainer**: MLX Server V3 Team
