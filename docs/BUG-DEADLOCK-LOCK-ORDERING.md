# Bug Report: ABBA Lock Ordering Deadlock

**Severity:** Critical (Production Deadlock)
**Status:** Identified, Fix Ready
**Date:** 2025-12-31
**Affects:** MLX Inference Server v3.0.0-alpha

---

## Summary

Server deadlocks during concurrent model loading and idle unload operations due to ABBA lock ordering violation between `activity_lock` and `self.lock` in `WorkerManager`.

---

## Symptom

- Server becomes unresponsive (0% CPU)
- API requests timeout
- No errors in logs
- Requires manual restart to recover

**Trigger:** Race condition when idle timeout fires during new model loading.

---

## Root Cause

**ABBA Lock Ordering Violation** in `worker_manager.py`:

### Thread A (Idle Monitor)
```python
def unload_model_if_idle():
    with self.activity_lock:        # Acquire A
        # Check if idle...
        with self.lock:              # Acquire B
            # Unload worker
```

**Lock order:** A → B

### Thread B (Request Handler)
```python
def generate(request):
    worker = self._get_worker_for_request()  # Acquires B (self.lock)

    self._increment_active_requests()        # Acquires A (activity_lock)
```

**Lock order:** B → A

### The Deadlock
```
Time  Thread A (Idle)           Thread B (Request)
────  ─────────────────         ──────────────────
T0    acquire activity_lock ✓
T1                               acquire self.lock ✓
T2    acquire self.lock [WAIT]
T3                               acquire activity_lock [WAIT]
T4    ← DEADLOCK (both waiting forever)
```

---

## The Fix

**Enforce consistent lock ordering:** Always acquire `activity_lock` → `self.lock`

### Changes Required

**File:** `src/orchestrator/worker_manager.py`

**Method:** `generate()` (and `generate_stream()`)

**Before (WRONG - causes deadlock):**
```python
def generate(self, request: CompletionRequest) -> Dict[str, Any]:
    # Gets worker (acquires self.lock)
    worker = self._get_worker_for_request(request.model)

    # Increments counter (acquires activity_lock) ← WRONG ORDER
    self._increment_active_requests()
```

**After (CORRECT - prevents deadlock):**
```python
def generate(self, request: CompletionRequest) -> Dict[str, Any]:
    # Acquire activity_lock FIRST (same order as unload_model_if_idle)
    with self.activity_lock:
        self.active_requests += 1
        self.last_activity_time = time.time()

    # Then get worker (acquires self.lock internally)
    worker = self._get_worker_for_request(request.model)
```

**Key change:** Move activity tracking BEFORE worker acquisition to match idle monitor's lock order.

---

## Implementation Details

### Files to Modify

1. **`src/orchestrator/worker_manager.py`**
   - Method: `generate()` (~line 350)
   - Method: `generate_stream()` (~line 400)
   - Change: Move `_increment_active_requests()` before `_get_worker_for_request()`

### Lock Ordering Rule

**Everywhere in WorkerManager:**
```
activity_lock → self.lock → (no locks during IPC)
```

Never reverse this order.

---

## Testing

### Test Case 1: Idle Unload During Load
```python
# Trigger deadlock scenario
1. Load model A
2. Wait until nearly at idle timeout (e.g., 119 seconds)
3. Request model B (triggers spawn)
4. Verify: No deadlock, both complete successfully
```

### Test Case 2: Concurrent Requests
```python
# Multiple simultaneous requests
1. Send 10 concurrent requests
2. Verify: All complete, no deadlock
3. Check: Lock acquisition logs show consistent ordering
```

### Test Case 3: Stress Test
```python
# Rapid model switching under load
1. Continuously switch between models A and B
2. Run for 1 hour
3. Verify: No deadlocks, all requests succeed
```

---

## Verification

After fix is applied:

**Check logs for consistent lock ordering:**
```bash
grep "acquire.*lock" logs/mlx-inference-server.log
```

**Monitor for deadlocks:**
```bash
# Run for 24 hours, check CPU usage
ps aux | grep mlx-inference-server
# If CPU stays >0%, no deadlock
```

**Health check:**
```bash
while true; do
  curl -s --max-time 2 http://localhost:11440/health || echo "DEADLOCK!"
  sleep 60
done
```

---

## Workaround (Until Fixed)

**Option 1: Disable Idle Timeout**
```yaml
# config/config.yaml
idle_timeout: 0  # Prevents idle unload, no deadlock risk
```

**Trade-off:** Models stay loaded (uses more RAM)

**Option 2: Increase Timeout**
```yaml
idle_timeout: 3600  # 1 hour (less frequent unload = less risk)
```

**Option 3: Manual Restart**
```bash
# If server deadlocks
./bin/mlx-inference-server-daemon.sh restart
```

---

## Related Issues

- Performance: phi-4 model significantly slower than Qwen (separate investigation)
- IPC: Semaphore cleanup warnings in logs (non-critical, cosmetic)

---

## References

- Opus 4.5 Review: `docs/sessions/2025-12-31-opus-deadlock-review.md` (internal)
- Worker Manager: `src/orchestrator/worker_manager.py`
- Idle Monitor: `src/orchestrator/idle_monitor.py`

---

**Status:** Fix ready for implementation
**Priority:** Critical (blocks production use)
**Branch:** `fix/deadlock-lock-ordering` (to be created)
**Estimated Fix Time:** 30 minutes (code + tests)

---

**Document Version:** 1.0
**Last Updated:** 2025-12-31
**Reviewed By:** Opus 4.5
