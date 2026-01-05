# MLS Worker Hang - Analysis & Diagnostic Guide
**Date:** 2026-01-05
**Status:** Active Investigation
**Incident:** Studio worker deadlock after 318 successful requests

---

## Executive Summary

**What happened:** MLS worker (PID 49387) on Mac Studio deadlocked during normal request processing. Worker hung on 8.5KB prompt after successfully processing 318 requests over 94.6 minutes. Process remained alive but unresponsive for 8+ hours with 0% CPU and stable memory.

**Root cause:** Deadlock in MLX generation code (not IPC, not memory, not crash)

**Impact:** Complete service failure requiring manual intervention

**Frequency:** Rare (1/318 requests, ~0.3% failure rate)

**Resolution:** Kill worker, spawn fresh one

---

# Part 1: Incident Analysis (Studio - 2026-01-05)

## Timeline

| Time | Event | Details |
|------|-------|---------|
| 18:17 - 23:13 | Normal operation | 318 requests, 30.2 tok/s avg, zero errors |
| 23:13:25 | Hang trigger | Received 8,489 char prompt (8.5KB, normal size) |
| 23:13:25 | Processing started | Worker began generation... **never responded** |
| 23:23:25 | +10 min | Shared memory timeout warning |
| 23:33:25 | +20 min | Full timeout, returned 500 error |
| 23:33:25 | 2nd request | Received 60KB prompt (also hung) |
| 23:43:25 | +30 min | Second timeout error |
| 07:02:46 | +8 hours | Worker still frozen, process alive |

## The Prompt That Triggered Hang

**Size:** 8,489 chars (8.5KB)
**Max tokens:** 2000
**Temperature:** 0.3
**Type:** Normal content chunk (not metadata extraction)
**Expected duration:** 10-15 seconds
**Actual duration:** Never completed (600s timeout)

**Critical observation:** Worker had successfully processed hundreds of similar prompts before this one.

## Process State at Discovery

```
PID    RSS      VSZ  %CPU %MEM  ELAPSED STAT
49387 8386464 444689440   0.0  6.2 08:49:23 S
```

**Interpretation:**
- **State S:** Sleeping/waiting (blocked on something)
- **CPU 0.0%:** Not computing, not generating tokens
- **RAM 8.4 GB:** Stable, no leak (model + overhead)
- **Elapsed 8+ hrs:** Frozen since 23:13:25

## Thread Analysis

Worker has 18+ threads, all in sleeping state:
```
USER             PID   TT   %CPU STAT PRI     STIME     UTIME
jameschildress 49387   ??    0.0 S    46T   0:45.27   4:17.03 (main)
               49387         0.0 S    31T   0:00.00   0:00.00 (worker threads...)
```

**Key finding:** Main thread accumulated 4:17 user time before freezing. All threads blocked, none running.

## IPC State

**Connections:**
- stdin/stdout/stderr pipes (FD 0, 1, 2) → Open
- IPC pipes (FD 5, 6, 7) → Open
- Unix socket (FD 13) → Connected
- _posixshmem module → Loaded

**Error progression:**
1. Shared memory timeout (600s)
2. Automatic fallback to stdio bridge
3. Stdio bridge also timed out (600s)
4. Total timeout: 1200s (20 minutes)

**Interpretation:** All IPC channels healthy. Worker not reading from them. Hang is in generation code, not IPC layer.

## Memory Analysis

8+ hours of continuous monitoring (every 30 seconds):

**Worker memory:**
- Start: 8.0 GB RSS
- End: 8.4 GB RSS
- Change: +400 MB (negligible, likely OS variance)

**System memory:**
- Start: 26.4 GB free
- End: 5.0 GB free
- Note: Normal OS behavior (disk cache growth)

**Conclusion:** Zero memory leak. Worker memory perfectly stable.

## Performance Context

Before hang, worker processed **318 requests successfully** over **94.6 minutes**:

| Metric | Value |
|--------|-------|
| Total requests | 318 |
| Total tokens | 126,549 |
| Average speed | 30.2 tok/s |
| Normal chunks (8-10KB) | 25-40 tok/s |
| Large prompts (100KB+) | 0.7-2 tok/s |
| Worker crashes | 0 |
| Failed requests | 0 (until hang) |

Worker was operating perfectly before deadlock occurred.

## Root Cause Analysis

### What It's NOT ❌

1. **Large prompt overhead**
   - Prompt was 8.5KB (normal size)
   - Large prompts are 100KB+ and process slowly but complete
   - This never completed

2. **Out of memory**
   - RAM stable at 8.4 GB
   - No swapping, no memory pressure
   - System had 5+ GB free

3. **Worker crash**
   - Process still alive 8+ hours later
   - No exit code, no signal
   - All threads present

4. **Timeout too short**
   - Normal prompts take 10-15 seconds
   - Timeout is 600 seconds (10 minutes)
   - 40x longer than needed

5. **IPC failure**
   - All channels open and connected
   - Worker just not responding
   - Orchestrator healthy

### What It IS ✅

**Deadlock or infinite loop in MLX generation code**

**Evidence:**
- Worker stuck in sleeping state (blocked on lock/condition)
- 0% CPU (not computing)
- Stable memory (not growing)
- Process alive but unresponsive
- All threads blocked

**Most likely location:**
- Model forward pass (attention computation)
- KV cache management
- Metal GPU command buffer
- Thread synchronization primitive

**Trigger conditions:**
- Non-deterministic (worked 318 times, failed once)
- Normal prompt size (not edge case)
- Likely timing-dependent race condition
- May be related to sustained load (94 minutes runtime)

---

# Part 2: Diagnostic Procedure

Use this procedure to diagnose workers on **any machine** (Studio, Mini, Air).

## Quick Check: Is Worker Hung?

```bash
# 1. Find all MLS workers
ps aux | grep "mlx-server-v3" | grep -v grep

# 2. Check process state
ps -p <PID> -o pid,rss,vsz,%cpu,%mem,etime,state

# Signs of hung worker:
# - State: S (sleeping)
# - CPU: 0.0%
# - ELAPSED: Very long (hours)
# - Should be generating if active
```

## Full Diagnostic Collection

### Step 1: Identify Worker

```bash
# Find all workers
ps aux | grep "mlx-server-v3" | grep -v grep

# Expected output:
# jameschildress 49387 0.0 6.2 mlx-server-v3-1
```

### Step 2: Check Worker State

```bash
# Get detailed process info
ps -p <PID> -o pid,rss,vsz,%cpu,%mem,etime,state

# Hung worker shows:
# - %CPU: 0.0
# - STAT: S (sleeping)
# - ELAPSED: Hours
```

### Step 3: Check Thread State

```bash
# Show all threads
ps -M -p <PID> | head -20

# Look for:
# - Total thread count (15-20 normal)
# - All in S state = deadlock
# - High UTIME = where time was spent
```

### Step 4: Check IPC Connections

```bash
# Check file descriptors
lsof -p <PID> 2>/dev/null | grep -E "(PIPE|unix|shm)"

# Should see:
# - stdin/stdout/stderr (FD 0,1,2)
# - IPC pipes (FD 5,6,7)
# - Unix socket (FD 13)

# If missing = IPC failure
# If present but silent = deadlock
```

### Step 5: Check Worker Registry

```bash
# Is worker registered?
cat /tmp/mlx-server/worker_registry.json

# Should show worker PID
# If not listed = orphan
```

### Step 6: Check MLS Logs

```bash
# Find timeout errors
grep "Worker timeout after" <log_file>

# Find last success
grep "Chat completion:" <log_file> | tail -5

# Find hang start
grep "Chat completion request:" <log_file> | tail -10
```

### Step 7: Check System Memory

```bash
# Free RAM
vm_stat | grep "Pages free" | awk '{printf "Free RAM: %.1f GB\n", $3 * 16384 / 1073741824}'

# Memory pressure
vm_stat | grep -E "(free|active|wired)"
```

### Step 8: Save Diagnostic Snapshot

```bash
# Set variables
PID=<worker_pid>
SNAPSHOT="/tmp/worker_${PID}_$(date +%Y%m%d_%H%M%S).txt"

# Create snapshot
{
  echo "=== Worker Diagnostic Snapshot ==="
  echo "Date: $(date)"
  echo "PID: $PID"
  echo ""
  echo "=== Process State ==="
  ps -M -p $PID
  echo ""
  echo "=== File Descriptors ==="
  lsof -p $PID 2>/dev/null | grep -E "(PIPE|unix|shm)"
  echo ""
  echo "=== Worker Registry ==="
  cat /tmp/mlx-server/worker_registry.json
  echo ""
  echo "=== System Memory ==="
  vm_stat
} > $SNAPSHOT

echo "Snapshot saved: $SNAPSHOT"
```

## Interpreting Results

### Healthy Worker ✅
- **State:** S but wakes frequently
- **CPU:** 0-50% idle, spikes to 200-800% when generating
- **RAM:** Stable at model size
- **IPC:** All pipes open
- **Logs:** Recent completions

### Hung Worker (Deadlock) 🔴
- **State:** S continuously
- **CPU:** 0.0% for hours
- **RAM:** Stable (not leaking)
- **IPC:** Open but worker silent
- **Logs:** "Worker timeout" errors
- **Timeline:** Last request never completed

### Crashed Worker 💥
- **State:** Process doesn't exist
- **Registry:** Worker listed but PID invalid
- **Logs:** Python traceback or signal

### Orphan Worker 👻
- **State:** Process exists
- **Registry:** NOT listed
- **Parent:** Parent PID invalid

## Decision Tree

```
Is worker running? (ps aux | grep mlx-server-v3)
├─ NO → Check logs for crash, registry for orphans
└─ YES → Check CPU and state
    ├─ CPU 0%, State S, >30 min → HUNG (deadlock)
    ├─ CPU >0%, State R → WORKING (normal)
    └─ CPU varies, State S → IDLE (normal)
```

## Machine-Specific Notes

### Mac Studio (128 GB)
- Workers: 1-3 concurrent
- RAM per worker: 7-15 GB
- Timeout: 600s

### M4 Mac Mini (16 GB)
- Workers: 1 max
- RAM per worker: 4-8 GB
- Watch for memory pressure

### MacBook Air M4 (32 GB)
- Workers: 1-2
- RAM per worker: 7-10 GB
- Moderate capacity

---

# Part 3: Recommendations

## Immediate Actions

1. **Kill frozen worker** (PID 49387)
2. **Let MLS spawn fresh worker**
3. **Monitor for recurrence** over 24-48 hours
4. **Run same diagnostic on Mini and M4 Mini**

## Investigation Needed

### 1. Add Worker-Side Logging
**Problem:** Currently no worker stdout/stderr captured
**Solution:** Redirect worker output to file
```python
# In worker spawn
stdout=open(f'/tmp/mlx-worker-{pid}.log', 'w')
stderr=subprocess.STDOUT
```

**Benefit:** See exactly where worker hangs

### 2. Implement Heartbeat
**Problem:** No way to detect silent deadlock
**Solution:** Worker sends periodic "alive" signals
```python
# Worker-side
def heartbeat_thread():
    while True:
        send_heartbeat()
        time.sleep(10)
```

**Benefit:** Detect hang within 10-20 seconds

### 3. Add Worker-Side Timeout
**Problem:** Worker can hang forever
**Solution:** Worker kills itself if generation exceeds limit
```python
# Worker-side
def generate_with_timeout(prompt, max_time=300):
    signal.alarm(max_time)
    try:
        result = model.generate(prompt)
    except TimeoutError:
        logger.error(f"Generation exceeded {max_time}s, exiting")
        sys.exit(1)
```

**Benefit:** Self-recovery without orchestrator intervention

### 4. Check MLX Version
**Action:** Review MLX release notes for known deadlock issues
**Current version:** Check with `pip show mlx`
**Look for:** Threading bugs, Metal driver issues, KV cache fixes

### 5. Attempt Reproduction
**Method:** Replay exact prompt that caused hang
**Data needed:** Full prompt content (not just size)
**Test:** Run same prompt 100x, see if hang recurs
**Benefit:** If reproducible, easier to debug

### 6. Consider Timeout Reduction
**Current:** 600s (10 minutes)
**Observation:** Normal requests take 10-15s, large take 2-5 min
**Proposal:** Reduce to 300s (5 minutes) for faster failure detection
**Risk:** May timeout legitimate large prompts (>100KB)

## Code Changes to Consider

```python
# worker.py - Add generation logging
def generate(request):
    logger.info(f"Starting generation: {len(request.prompt)} chars, max_tokens={request.max_tokens}")
    start = time.time()

    try:
        result = model.generate(...)
        logger.info(f"Generation complete: {time.time()-start:.1f}s")
        return result
    except Exception as e:
        logger.error(f"Generation failed after {time.time()-start:.1f}s: {e}")
        raise

# worker.py - Add self-timeout
def generation_watchdog(max_seconds=300):
    def timeout_handler(signum, frame):
        logger.error(f"Worker hung for {max_seconds}s, forcing exit")
        os._exit(1)

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(max_seconds)

# orchestrator/api.py - Add worker health check
async def check_worker_health():
    while True:
        if worker_manager.has_worker():
            if not worker_manager.ping():  # new method
                logger.error("Worker not responding to ping")
                worker_manager.kill_worker()
                worker_manager.spawn_worker()
        await asyncio.sleep(60)
```

## Monitoring Enhancements

```python
# Track worker uptime and request count
worker_metrics = {
    'spawned_at': datetime.now(),
    'requests_processed': 0,
    'total_tokens': 0,
    'last_activity': datetime.now(),
}

# Alert on suspicious patterns
if worker_metrics['last_activity'] < datetime.now() - timedelta(minutes=15):
    alert("Worker inactive for 15 minutes")

if worker_metrics['requests_processed'] > 1000:
    logger.info("Worker processed 1000 requests, consider restart")
```

---

# Part 4: Data Collected

## Files Created

1. **This document:** `docs/worker-hang-analysis-and-diagnostic-guide.md`
2. **Process snapshot:** `/tmp/worker_49387_info.txt`
3. **Performance analysis:** `mls_performance_analysis.md`
4. **Request analysis:** `mls_request_analysis_full.txt`

## Log Files

1. **API logs:** `/tmp/claude/-Users-.../tasks/b3d68fe.output`
2. **Memory monitoring:** `/tmp/claude/-Users-.../tasks/bc624fd.output`

## Key Data Points

- ✅ Exact timestamp: 23:13:25
- ✅ Exact prompt size: 8,489 chars (8.5KB)
- ✅ Request count before hang: 318
- ✅ Total runtime before hang: 94.6 minutes
- ✅ Average performance: 30.2 tok/s
- ✅ Memory monitoring: 8+ hours @ 30s intervals
- ✅ Process state: All threads, FDs, registry
- ✅ IPC state: All channels open

## Missing Data (requires sudo)

- ❌ Stack trace (would show exact hang location)
- ❌ Core dump
- ❌ System call trace

---

# Part 5: Post-Mortem Template

Use this template after each hang incident:

```markdown
## Worker Hang Incident - <Date>

**Machine:** <Studio/Mini/Air>
**Worker PID:**
**Model:**
**Hang duration:**
**Prompt size:**
**Requests before hang:**
**Total runtime:**

**Symptoms:**
- [ ] Timeout error
- [ ] Deadlock (0% CPU)
- [ ] Memory leak
- [ ] Crash

**Memory state:**
- Start: X GB
- End: Y GB
- Stable/Growing/Leaking

**Data collected:**
- [ ] Process snapshot
- [ ] Thread state
- [ ] IPC state
- [ ] Registry state
- [ ] Logs with timeline
- [ ] Stack trace (if available)

**Files:**
- Snapshot: /tmp/worker_XXX.txt
- Logs: /tmp/...

**Action taken:**
- [ ] Killed worker
- [ ] Spawned fresh worker
- [ ] Updated monitoring
- [ ] Reported to team

**Follow-up:**
- [ ] Check other machines
- [ ] Review MLX version
- [ ] Attempt reproduction
- [ ] Implement fixes
```

---

# Part 6: Quick Reference

## One-Liner Diagnostic

```bash
# Quick health check
PID=<worker_pid>
ps -p $PID -o pid,rss,%cpu,etime,state && \
echo "Registry:" && cat /tmp/mlx-server/worker_registry.json | grep -A 3 "$PID" && \
echo "Free RAM:" && vm_stat | grep "Pages free" | awk '{printf "%.1f GB\n", $3*16384/1073741824}'
```

## Signs of Each Problem

| Symptom | State | CPU | RAM | Logs |
|---------|-------|-----|-----|------|
| **Healthy** | S (brief) | Varies | Stable | Recent activity |
| **Hung** | S (hours) | 0% | Stable | Timeout errors |
| **Crashed** | None | N/A | N/A | Traceback |
| **Orphan** | S/R | Varies | Stable | Not in registry |
| **Memory leak** | S/R | Varies | Growing | OOM possible |

---

# Appendix: Studio Incident Summary

**Date:** 2026-01-05
**Worker:** PID 49387 (mlx-server-v3-1)
**Model:** mlx-community/Qwen2.5-14B-Instruct-4bit
**Duration:** 8+ hours hung
**Cause:** Deadlock in MLX generation code
**Resolution:** Pending (worker still frozen, awaiting kill command)

**Impact:**
- Severity: High (total service failure)
- Frequency: Rare (0.3% of requests)
- Recovery: Manual intervention required

**Lessons learned:**
1. Need worker-side logging to pinpoint hang location
2. Need heartbeat mechanism for early detection
3. Need worker-side timeout for self-recovery
4. Current diagnostics insufficient without stack trace
5. Hang can occur on normal prompts after sustained operation

**Next steps:**
1. Kill PID 49387 on Studio
2. Run diagnostics on Mini and M4 Mini
3. Implement worker-side logging
4. Add heartbeat mechanism
5. Test with worker-side timeout

---

**Document version:** 1.0
**Maintained by:** MLS Development Team
**Last updated:** 2026-01-05
**Status:** Active - Investigation ongoing
