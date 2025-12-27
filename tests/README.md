# MLX Server Test Suite

Automated tests for MLX server functionality, with focus on concurrency handling.

## Test Files

### `test_concurrency.sh`
Comprehensive test suite for validating the concurrency fix.

**Tests included:**
1. Server health check
2. Single request baseline
3. Concurrent requests stress test (3 parallel requests)
4. Server stability verification
5. Error log analysis

## Running Tests

### Prerequisites
- MLX server must be running (`python3 mlx_server_extended.py`)
- Model `mlx-community/SmolLM2-1.7B-Instruct` must be available
- `curl` and `jq` installed

### Execute Test Suite

```bash
cd /path/to/mlx-inference-server
./tests/test_concurrency.sh
```

### Expected Output

```
=================================
MLX Server Concurrency Test Suite
=================================

[Test 1/5] Server Health Check
✓ Server is running and responsive

[Test 2/5] Single Request Baseline
✓ Single request completed successfully
  Response: Hello! How can I assist you today?...

[Test 3/5] Concurrent Requests Stress Test
  Sending 3 concurrent requests...
✓ All 3 concurrent requests completed successfully
  Duration: 180s

[Test 4/5] Server Stability Check
✓ Server remained stable (same PID: 34311)

[Test 5/5] Error Log Analysis
✓ No Metal GPU errors or crashes detected

=================================
Test Summary
=================================
Total Tests:  5
Passed:       5
Failed:       0

✓ ALL TESTS PASSED
Concurrency fix is working correctly!
```

## Test Coverage

### What's Tested
- ✅ Server responsiveness
- ✅ Single request inference
- ✅ Concurrent request handling (3 parallel)
- ✅ Server stability under load
- ✅ Metal GPU error detection
- ✅ Request queueing and serialization

### What's NOT Tested (Future Work)
- ⬜ Streaming responses with concurrent requests
- ⬜ Very high concurrency (10+ requests)
- ⬜ Model loading during inference
- ⬜ Mixed streaming/non-streaming concurrent requests
- ⬜ Large prompt concurrent requests
- ⬜ Timeout behavior under extreme load

## Troubleshooting

### Test Failures

**"Server is not running or not responsive"**
- Start server: `python3 mlx_server_extended.py`
- Verify port 11436 is accessible: `curl http://localhost:11436/v1/models`

**"Concurrent requests timed out"**
- Increase timeout in script: `TIMEOUT=600` (10 minutes)
- Check server logs: `tail -f logs/mlx_server.log`
- Verify model is loaded: `curl http://localhost:11437/admin/status`

**"Model not found"**
- Update MODEL variable in script to use available model
- List available models: `curl http://localhost:11436/v1/models`

## Test Results Archive

Test results are documented in `CONCURRENCY_TEST_RESULTS_2025-10-31.md`.

## CI/CD Integration (Future)

This test suite can be integrated into CI/CD pipeline:

```yaml
# Example GitHub Actions workflow
test:
  runs-on: macos-latest
  steps:
    - name: Start MLX Server
      run: python3 mlx_server_extended.py &
    - name: Run Concurrency Tests
      run: ./tests/test_concurrency.sh
```

## Manual Testing

For manual testing scenarios (Open WebUI rapid-fire):

1. Start server
2. Open WebUI in browser
3. Submit first query
4. Immediately submit second query (before first completes)
5. Verify both complete without server crash

## Performance Benchmarks

Expected performance with concurrency fix:

| Scenario | Time | Status |
|----------|------|--------|
| Single request | ~60s | Normal |
| 2 concurrent | ~120s | Serialized |
| 3 concurrent | ~180s | Serialized |

Performance impact is acceptable trade-off for stability.
