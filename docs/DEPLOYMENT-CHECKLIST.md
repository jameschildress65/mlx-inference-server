# Deployment Checklist

Use this checklist for deploying to new machines.

---

## Pre-Deployment Verification

- [ ] **macOS 13.0+** (Ventura or newer)
  ```bash
  sw_vers
  ```

- [ ] **Apple Silicon** (M1/M2/M3/M4)
  ```bash
  system_profiler SPHardwareDataType | grep Chip
  ```

- [ ] **16GB+ RAM** (32GB+ recommended for 7B+ models)
  ```bash
  system_profiler SPHardwareDataType | grep Memory
  ```

- [ ] **Homebrew installed**
  ```bash
  which brew
  ```

- [ ] **Python 3.12+** (via pyenv or Homebrew)
  ```bash
  python3 --version
  ```

---

## Installation Steps

- [ ] **Choose installation location** (non-iCloud)
  ```bash
  mkdir -p ~/projects
  cd ~/projects
  ```

- [ ] **Clone repository**
  ```bash
  git clone https://github.com/jameschildress65/mlx-inference-server.git
  cd mlx-inference-server
  ```

- [ ] **Create virtual environment**
  ```bash
  python3 -m venv venv
  ```

- [ ] **Activate virtual environment**
  ```bash
  source venv/bin/activate
  ```
  Prompt should show `(venv)` prefix

- [ ] **Upgrade pip**
  ```bash
  pip install --upgrade pip
  ```

- [ ] **Install dependencies**
  ```bash
  pip install -r requirements.txt
  ```
  Wait 5-10 minutes for installation

- [ ] **Verify critical dependencies**
  ```bash
  python -c "import mlx.core as mx; print(f'MLX: {mx.__version__}')"
  python -c "import fastapi; print('FastAPI: OK')"
  python -c "import posix_ipc; print('posix_ipc: OK')"
  python -c "from PIL import Image; print('Pillow: OK')"
  python -c "import fitz; print('PyMuPDF: OK')"
  ```
  All should succeed

---

## Vision Support Setup (Optional)

**Skip this section if you only need text inference.**

- [ ] **Create vision virtual environment**
  ```bash
  python3 -m venv venv-vision
  ```

- [ ] **Activate vision environment**
  ```bash
  source venv-vision/bin/activate
  ```
  Prompt should show `(venv-vision)` prefix

- [ ] **Install vision dependencies**
  ```bash
  pip install --upgrade pip
  pip install -r requirements-vision.txt
  ```
  Wait 5-10 minutes (PyTorch is large)

- [ ] **Verify vision dependencies**
  ```bash
  python -c "import mlx_vlm; print('mlx-vlm: OK')"
  python -c "from PIL import Image; print('Pillow: OK')"
  python -c "import torch; print('PyTorch: OK')"
  python -c "import torchvision; print('Torchvision: OK')"
  ```

- [ ] **Deactivate vision environment**
  ```bash
  deactivate
  ```

---

## Server Startup

- [ ] **Start server using daemon script**
  ```bash
  ./bin/mlx-inference-server-daemon.sh start
  ```

- [ ] **Verify server started**
  ```bash
  ./bin/mlx-inference-server-daemon.sh status
  ```
  Should show server running with PID

- [ ] **Check logs**
  ```bash
  tail -30 logs/mlx-inference-server.log
  ```
  Look for "Server started successfully!"

---

## Functional Testing

- [ ] **Test health endpoint**
  ```bash
  curl http://localhost:11440/health
  ```
  Expected: `{"status":"healthy",...}`

- [ ] **Test admin status**
  ```bash
  curl -s http://localhost:11441/admin/status | python3 -m json.tool
  ```
  Should show server running, no model loaded

- [ ] **Load test model**
  ```bash
  curl -s -X POST 'http://localhost:11441/admin/load?model_path=mlx-community/Qwen2.5-0.5B-Instruct-4bit' | python3 -m json.tool
  ```
  Wait for model to load (5-10 sec), should return success

- [ ] **Test chat completion**
  ```bash
  curl -s -X POST http://localhost:11440/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
      "messages": [{"role": "user", "content": "Say hello in 5 words"}],
      "max_tokens": 20
    }' | python3 -m json.tool
  ```
  Should return JSON with assistant response

- [ ] **Verify tokens_per_sec in response**
  Check that `usage` object contains `tokens_per_sec` field

---

## Performance Baseline

Record these values for future reference:

**System Information:**
```bash
system_profiler SPHardwareDataType | grep -E "Model Name|Chip|Memory"
```
- Model Name: ___________________
- Chip: ___________________
- Memory: ___________________

**Model Performance:**
- Model: mlx-community/Qwen2.5-0.5B-Instruct-4bit
- Load Time: ___________ seconds
- Memory Usage: ___________ GB
- Tokens/sec: ___________ tok/s

**Expected Performance by Hardware:**
| System | 0.5B-4bit Speed |
|--------|-----------------|
| M4 Mini 16GB | 15-25 tok/s |
| M4 Air 32GB | 15-20 tok/s |
| M4 Max 128GB | 70-90 tok/s |

---

## Production Readiness

- [ ] Server starts cleanly via daemon script
- [ ] Logs writing to `logs/mlx-inference-server.log`
- [ ] Admin API accessible on port 11441
- [ ] Main API accessible on port 11440
- [ ] Model loading successful
- [ ] Inference working
- [ ] Performance within expected range

---

## Common Issues

### Issue: "ModuleNotFoundError: No module named 'posix_ipc'"

**Fix:**
```bash
pip install posix_ipc
```

### Issue: Port already in use

**Fix:**
```bash
lsof -ti:11440,11441 | xargs kill -9
./bin/mlx-inference-server-daemon.sh start
```

### Issue: Python not found (SSH deployment)

**Fix:** Use login shell
```bash
ssh remote "zsh -l -c 'python3 --version'"
```

---

## Remote Deployment (SSH)

If deploying via SSH, use login shells for commands:

```bash
# Clone
ssh remote "mkdir -p ~/projects && cd ~/projects && git clone https://github.com/jameschildress65/mlx-inference-server.git"

# Setup
ssh remote "zsh -l -c 'cd ~/projects/mlx-inference-server && python3 -m venv venv'"
ssh remote "zsh -l -c 'cd ~/projects/mlx-inference-server && source venv/bin/activate && pip install -r requirements.txt'"

# Start
ssh remote "cd ~/projects/mlx-inference-server && ./bin/mlx-inference-server-daemon.sh start"

# Verify
ssh remote "curl -s http://localhost:11441/admin/status | python3 -m json.tool"
```

---

## Cleanup (if needed)

**Stop server:**
```bash
./bin/mlx-inference-server-daemon.sh stop
```

**Remove installation:**
```bash
cd ~/projects
rm -rf mlx-inference-server
```

**Remove models (optional, saves disk space):**
```bash
rm -rf ~/.cache/huggingface/hub/
```

---

## Success Criteria

✅ **All checkboxes above completed**
✅ **Server responding on both ports**
✅ **Model loads successfully**
✅ **Chat completions working**
✅ **Performance within expected range**

---

**Document Version:** 1.0
**Last Updated:** 2025-12-29
**Validated On:** M4 Mini (16GB), M4 Max Studio (128GB), M4 Air (32GB)
