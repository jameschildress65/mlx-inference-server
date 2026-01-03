# Deployment Lessons Learned: M4 Mini Deployment
**Date:** 2025-12-29
**System:** Mac mini M4 (16GB RAM)
**Deployment Location:** ~/projects/mlx-inference-server

---

## Executive Summary

Successfully deployed mlx-inference-server to M4 Mini via remote SSH. Deployment revealed **1 critical missing dependency** and **3 documentation gaps** that should be addressed in the repository.

**Result:** Server running successfully, all tests passed, tokens_per_sec feature working (19.21 tok/s on Qwen2.5-0.5B-Instruct-4bit).

---

## Issues Encountered

### 1. **CRITICAL: Missing `posix_ipc` Dependency**

**Problem:**
```python
ModuleNotFoundError: No module named 'posix_ipc'
```

**Impact:** Server fails to start without this package.

**Root Cause:** `posix_ipc` is required by `src/ipc/shared_memory_bridge.py` but is **not listed in requirements.txt**.

**Current State:**
- INSTALLATION-GUIDE.md mentions it as optional workaround (line 201-204)
- Should be mandatory, not optional

**Fix Applied:**
```bash
pip install posix_ipc
```

**Recommendation:** Add to requirements.txt as mandatory dependency.

---

### 2. SSH PATH Issues for Remote Deployment

**Problem:**
When executing commands via SSH, non-login shells don't load Homebrew or pyenv paths.

**Evidence:**
```bash
ssh-m4mini "python3 --version"
# Returns: Python 3.9.6 (system Python, wrong)

ssh-m4mini "zsh -l -c 'python3 --version'"
# Returns: Python 3.14.2 (Homebrew Python, correct)
```

**Impact:** Remote deployment scripts must use `zsh -l -c` wrapper or full paths.

**Workaround Used:**
```bash
ssh-m4mini "zsh -l -c 'cd ~/projects/mlx-inference-server && python3 -m venv venv'"
```

**Recommendation:** Document this in deployment guide for remote/automated installations.

---

### 3. Startup Method Confusion

**Problem:**
Initial attempts to run server directly via Python module failed:
```bash
python -m src.orchestrator.api  # FAILS - no __main__ entry point
```

**Correct Method:**
```bash
./bin/mlx-inference-server-daemon.sh start  # SUCCESS
```

**Root Cause:**
- README.md shows both methods (line 59-62) without clarifying which is preferred
- `python mlx_inference_server.py` works for development
- Daemon script is production method
- Direct module execution (`python -m src.orchestrator.api`) doesn't work

**Recommendation:** Clarify in documentation that daemon script is the recommended production method.

---

### 4. Missing `timeout` Command on macOS

**Problem:**
GNU `timeout` command not available on macOS by default.

**Evidence:**
```bash
timeout 30 python -m src.orchestrator.api
# zsh:1: command not found: timeout
```

**Impact:** Minor - only affects testing/debugging commands.

**Workaround:** Use `nohup` or background jobs with manual kill.

**Recommendation:** Don't rely on `timeout` in scripts; use background jobs instead.

---

## What Worked Well

### ✅ Auto-Configuration
- Server correctly detected M4 Mini as "low-memory" (16GB)
- Applied appropriate idle timeout (180s)
- No manual configuration needed

### ✅ Model Detection
- Found existing cached models in `~/.cache/huggingface/hub/`
- Loaded Qwen2.5-0.5B-Instruct-4bit successfully (0.26 GB)
- No re-download required

### ✅ Daemon Script
- `bin/mlx-inference-server-daemon.sh` worked perfectly
- Clean startup, proper logging
- Correct PID management

### ✅ API Functionality
- Admin API (11441) working
- Main API (11440) working
- Chat completions successful
- **tokens_per_sec feature present in response** ✓

---

## Performance Validation

**Model:** mlx-community/Qwen2.5-0.5B-Instruct-4bit
**Memory:** 0.26 GB
**Load Time:** 5.4 seconds
**Inference Speed:** 19.21 tokens/sec

**Expected for M4 Mini 16GB:** ✓ Within normal range for small models

**Comparison:**
- M4 Mini (16GB): 19 tok/s (0.5B model)
- M4 Max Studio (128GB): 78 tok/s (0.5B model)

Performance appropriate for hardware tier.

---

## Required Changes to Repository

### 1. Update `requirements.txt`

**Current:**
```txt
# MLX and model serving
mlx-lm>=0.28.4
setproctitle>=1.3.3

# V3 API dependencies
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.0.0

# Testing
pytest>=7.4.0
pytest-asyncio>=0.21.0
httpx>=0.24.0
psutil>=5.9.0
```

**Add:**
```txt
# IPC dependencies
posix_ipc>=1.0.0
```

**Justification:** Required by shared_memory_bridge.py, server fails without it.

---

### 2. Update README.md

**Section: Start Server (lines 52-67)**

**Current:**
```markdown
### Start Server

**Production (background):**
```bash
./bin/mlx-inference-server-daemon.sh
```

**Development (foreground):**
```bash
python3 mlx_inference_server.py
```
```

**Recommended Change:**
```markdown
### Start Server

**Production (recommended):**
```bash
./bin/mlx-inference-server-daemon.sh start
```

**Development (foreground):**
```bash
python3 mlx_inference_server.py
```

**Note:** The daemon script is the recommended production method. It provides:
- Background execution
- Proper logging to `logs/mlx-inference-server.log`
- Clean shutdown via `./bin/mlx-inference-server-daemon.sh stop`
- Status monitoring via `./bin/mlx-inference-server-daemon.sh status`
```

---

### 3. Update INSTALLATION-GUIDE.md

**Section: Step 4: Install Dependencies (lines 193-213)**

**Current (lines 201-204):**
```markdown
**If you see errors about `posix_ipc`, install it separately:**
```bash
pip install posix_ipc
```
```

**Recommended Change:**
```markdown
**Note:** All required dependencies including `posix_ipc` are in requirements.txt.

**Verify installation:**
```bash
python -c "import mlx.core as mx; print(f'MLX version: {mx.__version__}')"
python -c "import fastapi; print('FastAPI installed ✓')"
python -c "import posix_ipc; print('posix_ipc installed ✓')"
```

All three should succeed.
```

---

### 4. Add Remote Deployment Section to INSTALLATION-GUIDE.md

**New Section (add after Step 4):**

```markdown
---

## Remote Deployment (SSH)

**If deploying to a remote Mac via SSH, use login shells for commands:**

### Why?
Non-login SSH shells don't load `~/.zshrc`, so Homebrew and pyenv aren't in PATH.

### Solution:
Wrap commands in `zsh -l -c`:

```bash
# Wrong (uses system Python 3.9)
ssh remote-mac "python3 --version"

# Correct (uses Homebrew/pyenv Python)
ssh remote-mac "zsh -l -c 'python3 --version'"
```

### Example: Remote Installation

```bash
# Clone repo
ssh remote-mac "mkdir -p ~/projects && cd ~/projects && git clone https://github.com/jameschildress65/mlx-inference-server.git"

# Create venv and install
ssh remote-mac "zsh -l -c 'cd ~/projects/mlx-inference-server && python3 -m venv venv'"
ssh remote-mac "zsh -l -c 'cd ~/projects/mlx-inference-server && source venv/bin/activate && pip install -r requirements.txt'"

# Start server
ssh remote-mac "cd ~/projects/mlx-inference-server && ./bin/mlx-inference-server-daemon.sh start"
```

### Verify Deployment

```bash
# Check status
ssh remote-mac "curl -s http://localhost:11441/admin/status | python3 -m json.tool"

# Load model and test
ssh remote-mac "curl -s -X POST 'http://localhost:11441/admin/load?model_path=mlx-community/Qwen2.5-0.5B-Instruct-4bit'"
```

---
```

---

### 5. Create DEPLOYMENT-CHECKLIST.md

**New File:** `docs/DEPLOYMENT-CHECKLIST.md`

```markdown
# Deployment Checklist

Use this checklist for deploying to new machines.

## Pre-Deployment

- [ ] macOS 13.0+ (Ventura or newer)
- [ ] Apple Silicon (M1/M2/M3/M4)
- [ ] 16GB+ RAM
- [ ] Homebrew installed
- [ ] Python 3.12+ via pyenv or Homebrew

## Installation Steps

- [ ] Choose non-iCloud location (e.g., `~/projects/`)
- [ ] Clone repository: `git clone https://github.com/jameschildress65/mlx-inference-server.git`
- [ ] Create venv: `python3 -m venv venv`
- [ ] Activate venv: `source venv/bin/activate`
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Verify posix_ipc: `python -c "import posix_ipc; print('OK')"`

## Testing

- [ ] Start server: `./bin/mlx-inference-server-daemon.sh start`
- [ ] Check logs: `tail -f logs/mlx-inference-server.log`
- [ ] Test admin API: `curl http://localhost:11441/admin/status`
- [ ] Load test model: `curl -X POST 'http://localhost:11441/admin/load?model_path=mlx-community/Qwen2.5-0.5B-Instruct-4bit'`
- [ ] Test completion: `curl -X POST http://localhost:11440/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"mlx-community/Qwen2.5-0.5B-Instruct-4bit","messages":[{"role":"user","content":"Hello"}],"max_tokens":20}'`
- [ ] Verify tokens_per_sec in response

## Production Readiness

- [ ] Server starts cleanly via daemon script
- [ ] Logs writing to `logs/` directory
- [ ] Admin API accessible
- [ ] Main API accessible
- [ ] Model loading works
- [ ] Inference working with expected performance

## Performance Baselines

Record these for future reference:

- Model used: ___________________
- Memory usage: ___________________
- Load time: ___________________
- Tokens/sec: ___________________
- System RAM: ___________________
- Chip type: ___________________

---

**Document these for troubleshooting:**

```bash
# System info
system_profiler SPHardwareDataType | grep -E "Model Name|Chip|Memory"

# Python version
python3 --version

# MLX version
python3 -c "import mlx.core as mx; print(mx.__version__)"

# Server status
curl http://localhost:11441/admin/status | python3 -m json.tool
```
```

---

## Deployment Timeline

| Step | Time Taken |
|------|------------|
| Environment setup (brew upgrade, env vars) | 5 min |
| Clone repository | 30 sec |
| Create venv | 10 sec |
| Install dependencies | 3 min |
| Install posix_ipc (missing) | 30 sec |
| Start server (first attempt) | 2 min |
| Test and validate | 2 min |
| **Total** | **~12 minutes** |

**Note:** This was for a system that already had Python 3.14.2 and Homebrew configured. Fresh system would add 20-30 minutes for pyenv/Python installation.

---

## Recommendations Summary

### Immediate (Required)

1. **Add `posix_ipc>=1.0.0` to requirements.txt**
   - Critical dependency
   - Server fails without it
   - Currently requires manual installation

### Important (Should Do)

2. **Clarify startup methods in README.md**
   - Emphasize daemon script for production
   - Explain when to use direct Python execution

3. **Update INSTALLATION-GUIDE.md**
   - Remove "optional" language around posix_ipc
   - Add remote deployment section with SSH examples

### Nice to Have (Consider)

4. **Create DEPLOYMENT-CHECKLIST.md**
   - Helps users validate deployments
   - Provides troubleshooting baseline

5. **Add installation verification script**
   - `bin/verify-installation.sh`
   - Checks all dependencies
   - Tests basic functionality

---

## Validation

✅ **Deployment successful on M4 Mini**
✅ **All core functionality working**
✅ **tokens_per_sec feature confirmed**
✅ **Performance within expected range**

**Ready for production use on 16GB Apple Silicon systems.**

---

## Files to Update

1. `requirements.txt` - Add posix_ipc
2. `README.md` - Clarify startup methods
3. `docs/INSTALLATION-GUIDE.md` - Remove "optional" posix_ipc, add remote deployment
4. `docs/DEPLOYMENT-CHECKLIST.md` - Create new file
5. `docs/DEPLOYMENT-LESSONS-M4-MINI.md` - This document (archive of findings)

---

**Next Steps:**
- Review and approve these recommendations
- Update files as specified
- Commit changes to repository
- Update version/changelog if appropriate

---

**Author:** Deployment analysis from M4 Mini testing
**Date:** 2025-12-29
**Deployment Type:** Remote SSH deployment
**Success Rate:** 100% (after posix_ipc fix)
