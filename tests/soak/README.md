# MLX Inference Server - Soak Testing Suite

Comprehensive test suite for validating Production-grade production features across the MLX Server fleet.

## Test Scripts

### `air_comprehensive_test.sh`
**Target:** Apple Silicon Mac M4 (32GB RAM)
**Duration:** ~6 minutes (includes 3-minute idle timeout wait)
**Model:** `mlx-community/Qwen2.5-7B-Instruct-4bit` (7B, 4-bit quantized)

## Features Tested

### 1. **On-Demand Loading**
- Verifies model loads only when first request arrives (not at startup)
- Validates cold-start model loading works correctly

### 2. **Request Handling**
- Single completion request
- Multiple sequential requests (3x)
- Validates model stays loaded between requests

### 3. **Concurrent Requests (Thread Safety)**
- 3 parallel requests sent simultaneously
- Validates thread-safe inference lock prevents Metal GPU crashes
- No "command encoder already encoding" errors

### 4. **Manual Unload**
- Admin API `POST /admin/unload` command
- Verifies immediate unload on demand

### 5. **Memory Recovery (97.5% Target)**
- Measures baseline memory (no model)
- Measures memory after model load
- Measures memory after unload
- Validates ≥95% memory recovery (target: 97.5%)

### 6. **Reload After Unload**
- Load → Unload → Load cycle
- Ensures server can reload models after manual unload
- Tests full lifecycle management

### 7. **Idle Timeout Unload**
- Waits for Air's 180s (3-minute) idle timeout
- Verifies automatic unload triggers
- Checks every 30 seconds for up to 210s

### 8. **Memory Leak Detection**
- Compares final memory to initial baseline
- Validates delta <0.15 GB after full test cycle (accounts for Python/MLX/Metal overhead)
- Ensures no memory leaks across load/unload cycles
- Note: 0.10-0.15 GB residual is normal framework overhead, not a leak

### 9. **Status API**
- Verifies all required fields present:
  - `status`
  - `memory` (active_memory_gb, model_loaded, etc.)
  - `requests` (total, active, uptime, last_activity)
  - `idle_timeout_seconds`

### 10. **Health API**
- Validates `/admin/health` returns `healthy`
- Basic server responsiveness check

### 11. **Request Tracking**
- Monitors active_requests counter
- Ensures proper request lifecycle tracking

## Expected Results

### Pass Criteria
- ✅ All 11 tests pass
- ✅ Memory recovery ≥95%
- ✅ Idle timeout triggers ~180s (±30s acceptable)
- ✅ No concurrent request failures
- ✅ Memory delta <0.1 GB after full cycle

### Performance Benchmarks (Air M4, 32GB)
- **Model Load Time:** 10-15 seconds (7B model, 4-bit)
- **First Token Latency:** <500ms
- **Tokens/sec:** 40-60 tok/s (depends on prompt complexity)
- **Memory Usage:** ~5-7 GB for 7B 4-bit model
- **Memory Recovery:** 97.5% (typical)

## Usage

### On Apple Silicon Mac

```bash
# Navigate to project
cd /path/to/mlx-inference-server

# Ensure server is running
curl http://localhost:11437/admin/health

# Run comprehensive test
./tests/soak/air_comprehensive_test.sh
```

### Expected Output

```
========================================================================
MLX Inference Server - Comprehensive Soak Test
Machine: Apple Silicon Mac (medium-memory)
Test Model: mlx-community/Qwen2.5-7B-Instruct-4bit
Started: Mon Dec 23 14:30:00 PST 2024
========================================================================

[TEST] 1. Health Check - Verify server is running
[PASS] Server is healthy

[TEST] 2. Baseline Memory - Verify no model loaded
[INFO] Baseline memory: 0.0000 GB
[PASS] No model loaded at baseline (expected)

[TEST] 3. On-Demand Loading - First request triggers model load
[INFO] Sending first completion request...
[PASS] Model loaded on-demand and generated completion
[INFO] Response: MLX Test 1 OK
[INFO] Memory after load: 5.47 GB (delta: 5.47 GB)

[TEST] 4. Subsequent Queries - Model stays loaded for follow-up requests
[INFO] Query 1/3...
[INFO]   Response: 1
[INFO] Query 2/3...
[INFO]   Response: 2
[INFO] Query 3/3...
[INFO]   Response: 3
[PASS] Multiple queries completed successfully

[TEST] 5. Concurrent Requests - Thread-safe inference lock
[INFO] Sending 3 concurrent requests...
[INFO]   Concurrent request 1: OK
[INFO]   Concurrent request 2: OK
[INFO]   Concurrent request 3: OK
[PASS] All concurrent requests handled successfully (thread-safe)

[TEST] 6. Manual Unload - Admin API unload command
[INFO] Sending manual unload command...
[PASS] Manual unload successful
[INFO] Memory freed: 5.47 GB

[TEST] 7. Memory Recovery - Verify memory returned to baseline
[INFO] Baseline: 0.0000 GB
[INFO] After unload: 0.1370 GB
[INFO] Delta: 0.1370 GB
[INFO] Recovery: 97.50%
[PASS] Memory recovery ≥95% (target: 97.5%)

[TEST] 8. Reload After Unload - Load → Unload → Load cycle
[INFO] Reloading model with new request...
[PASS] Model reloaded successfully after manual unload
[INFO] Response: RELOADED

[TEST] 9. Idle Timeout Unload - Auto-unload after 180s idle
[INFO] Waiting for idle timeout (180s = 3 minutes)...
[INFO] Current time: 14:31:45
[INFO]   30s elapsed - model still loaded (checking again...)
[INFO]   60s elapsed - model still loaded (checking again...)
[INFO]   90s elapsed - model still loaded (checking again...)
[INFO]   120s elapsed - model still loaded (checking again...)
[INFO]   150s elapsed - model still loaded (checking again...)
[INFO]   180s elapsed - model still loaded (checking again...)
[PASS] Idle timeout unload triggered after 180s (expected ~180s)

[TEST] 10. Memory Leak Detection - Final baseline check
[INFO] Initial baseline: 0.0000 GB
[INFO] Final memory: 0.0000 GB
[INFO] Total delta: 0.0000 GB
[PASS] No memory leaks detected (delta <0.1 GB)

[TEST] 11. Status API - Verify all fields present
[INFO]   Field 'status': present
[INFO]   Field 'memory': present
[INFO]   Field 'requests': present
[INFO]   Field 'idle_timeout_seconds': present
[PASS] All required status fields present

========================================================================
Test Summary
========================================================================
Tests Passed: 11
Tests Failed: 0
Completed: Mon Dec 23 14:35:00 PST 2024

✓ ALL TESTS PASSED - PRODUCTION VERIFIED
```

## Troubleshooting

### Test Failures

**Memory Recovery <95%:**
- Check for other processes holding model references
- Verify mlx cleanup is working (check logs)
- May indicate memory leak - investigate model provider unload logic

**Idle Timeout Not Triggering:**
- Verify idle_timeout_seconds in status API
- Check IdleMonitor thread is running (logs)
- Ensure no background requests keeping server active

**Concurrent Request Failures:**
- Metal GPU error - check for missing inference lock
- Review mlx_server_extended.py:110 (inference lock)
- Check logs for "command encoder" errors

**Model Load Failures:**
- Verify HF_HOME points to valid cache directory
- Check model exists: `ls ~/.local/mlx-models/hub/models--mlx-community/`
- Review logs/mlx_server.log for errors

## Next Steps After Passing

1. Run test 3x to verify consistency
2. Deploy to M4 Mini (16GB) - adjust model size
3. Deploy to M1 Mini - adjust expectations
4. Create long-term soak test (24-hour run)

## Test Variations

### Small Model (for 16GB machines)
Edit line 6:
```bash
MODEL="mlx-community/Qwen2.5-0.5B-Instruct-4bit"
```

### Large Model (for 128GB Studio)
Edit line 6:
```bash
MODEL="mlx-community/Qwen2.5-32B-Instruct-4bit"
```

### Extended Soak (longer timeout wait)
Edit line 164 (change 7 to 13 for 390s = 6.5min wait):
```bash
for i in {1..13}; do
```

## Log Collection

After test completion:
```bash
# Server logs
cat logs/mlx_server.log | tail -100

# Daemon logs
cat logs/daemon.log | tail -100

# System memory
vm_stat
```

## Author
MLX Inference Server Development Team
Tested on: Apple Silicon Mac (medium-memory), Apple Silicon Mac (high-memory)
Last Updated: 2024-12-23
