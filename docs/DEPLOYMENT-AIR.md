# Deployment to MacBook Air M4 - Step-by-Step Guide
**Target:** MacBook Air M4 (32GB RAM)
**Date:** 2025-12-30
**Status:** Opus certified production-ready

---

## Pre-Deployment Verification

**This deployment is certified production-ready by:**
- Opus 4.5 comprehensive review (Phase 4)
- All security hardening validated (H1-H5, C1-C2, M2, M5-M7)
- Vision integration tested and working
- Performance validated (54-95 tok/s vision, 71-85 tok/s text)

**See:** `docs/OPUS-REVIEW-VISION-INTEGRATION.md`

---

## Step 1: Stop Existing Service (If Running)

```bash
# Check if server is running
cd ~/Documents/projects/utilities/mlx-inference-server
./bin/mlx-inference-server-daemon.sh status

# If running, stop it
./bin/mlx-inference-server-daemon.sh stop

# Verify it stopped
./bin/mlx-inference-server-daemon.sh status
```

**Expected output:** "Server is not running"

---

## Step 2: Pull Latest Code

```bash
# Navigate to project
cd ~/Documents/projects/utilities/mlx-inference-server

# Pull latest from GitHub
git pull origin main

# Verify you have latest commits
git log --oneline -5
```

**Expected commits (most recent first):**
```
29911b9 docs: add Opus 4.5 comprehensive system review (Phase 4)
867f11d docs: add vision API spec and document unauthorized RAGMT changes
c2adbe6 fix: vision API multimodal content handling
2898ecb fix: semaphore name generation must be deterministic
a020231 docs: add comprehensive vision model setup guide
```

---

## Step 3: Set Up Text Environment (venv)

```bash
# Remove old venv if it exists (optional, but recommended for clean install)
rm -rf venv

# Create new venv
python3 -m venv venv

# Activate
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
python -c "import mlx_lm; print('mlx_lm installed')"

# Deactivate
deactivate
```

**Expected:** No errors, mlx_lm imports successfully

---

## Step 4: Set Up Vision Environment (venv-vision)

```bash
# Remove old venv-vision if it exists (optional)
rm -rf venv-vision

# Create vision venv
python3 -m venv venv-vision

# Activate
source venv-vision/bin/activate

# Install vision dependencies
pip install --upgrade pip
pip install mlx-vlm pillow transformers>=4.44.0,<5.0

# Install server dependencies (needed in both envs)
pip install setproctitle pyyaml posix-ipc

# Verify installation
python -c "import mlx_vlm; print('mlx-vlm installed')"
python -c "from PIL import Image; print('PIL installed')"

# Deactivate
deactivate
```

**Expected:** No errors, all imports successful

**Note:** transformers must be 4.x for vision (not 5.x)

---

## Step 5: Verify Configuration

```bash
# Check config file exists
cat config/config.yaml
```

**Expected:** Configuration with text and vision model settings

**Default vision model:** `mlx-community/Qwen2.5-VL-7B-Instruct-4bit`

---

## Step 6: Download Vision Model (If Not Cached)

```bash
# Check if model is already downloaded
ls -lh ~/.cache/huggingface/hub/ | grep Qwen2.5-VL-7B-Instruct-4bit

# If not present, download now (optional - will auto-download on first request)
# This saves time on first vision request
source venv-vision/bin/activate
python -c "from huggingface_hub import snapshot_download; snapshot_download('mlx-community/Qwen2.5-VL-7B-Instruct-4bit')"
deactivate
```

**Expected:** Model downloads (~5GB) or already exists

---

## Step 7: Start Server

```bash
# Start in daemon mode
./bin/mlx-inference-server-daemon.sh start

# Wait 5 seconds for startup
sleep 5

# Check status
./bin/mlx-inference-server-daemon.sh status
```

**Expected output:**
```
[INFO] Server is running (PID: XXXXX)
[INFO] Main API: ✓ Healthy
[INFO] Admin API: ✓ Healthy
```

---

## Step 8: Verify Health

```bash
# Test health endpoint
curl http://localhost:11440/health
```

**Expected:** `{"status":"healthy"}`

---

## Step 9: Test Text Inference

```bash
# Simple text completion
curl -X POST http://localhost:11440/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Say hello in 5 words"}],
    "max_tokens": 20
  }'
```

**Expected:** JSON response with completion, ~70-85 tok/s

---

## Step 10: Test Vision Inference

Create test script:

```bash
cat > /tmp/test_vision.py << 'EOF'
#!/usr/bin/env python3
import requests
import base64
from PIL import Image
import io

# Create simple test image (red square)
img = Image.new('RGB', (100, 100), color='red')
buf = io.BytesIO()
img.save(buf, format='PNG')
img_bytes = buf.getvalue()

# Encode to base64
img_b64 = base64.b64encode(img_bytes).decode()
data_uri = f"data:image/png;base64,{img_b64}"

# Send to MLX server
response = requests.post(
    "http://localhost:11440/v1/chat/completions",
    json={
        "model": "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What color is this image?"},
                {"type": "image_url", "image_url": {"url": data_uri}}
            ]
        }],
        "max_tokens": 50
    },
    timeout=60
)

if response.status_code == 200:
    data = response.json()
    print(f"✅ Vision OK: {data['choices'][0]['message']['content']}")
    print(f"   Speed: {data['usage'].get('tokens_per_sec', 'N/A')} tok/s")
else:
    print(f"❌ Error: {response.status_code} - {response.text}")
EOF

# Run test
python3 /tmp/test_vision.py
```

**Expected:**
- First request: 5-10 second delay (model loading)
- Response mentions "red" color
- Speed: ~54-95 tok/s

---

## Step 11: Monitor Logs

```bash
# Watch logs in real-time
tail -f logs/mlx-inference-server.log
```

**Look for:**
- No errors
- Worker startup messages
- Request/response logging
- Performance metrics

**Press Ctrl+C to stop watching**

---

## Deployment Checklist

| Step | Task | Status |
|------|------|--------|
| 1 | Stop existing service | ☐ |
| 2 | Pull latest code | ☐ |
| 3 | Set up venv (text) | ☐ |
| 4 | Set up venv-vision | ☐ |
| 5 | Verify configuration | ☐ |
| 6 | Download vision model (optional) | ☐ |
| 7 | Start server | ☐ |
| 8 | Verify health | ☐ |
| 9 | Test text inference | ☐ |
| 10 | Test vision inference | ☐ |
| 11 | Monitor logs | ☐ |

---

## Troubleshooting

### Server Won't Start

**Check logs:**
```bash
tail -50 logs/mlx-inference-server.log
```

**Common issues:**
- Port 11440 already in use: `lsof -i :11440`
- Missing dependencies: Re-run pip install
- Python version: Requires Python 3.12+

### Vision Requests Fail

**Check vision environment:**
```bash
source venv-vision/bin/activate
python -c "import mlx_vlm; print(mlx_vlm.__version__)"
python -c "import transformers; print(transformers.__version__)"
deactivate
```

**Required:**
- mlx-vlm installed
- transformers 4.x (not 5.x)
- PIL/Pillow installed

### Performance Issues

**Check system resources:**
```bash
# Memory usage
ps aux | grep mlx-inference-server

# CPU usage
top -pid $(pgrep -f mlx-inference-server)
```

**MacBook Air M4 (32GB RAM):**
- Text models: Should use ~400MB-2GB
- Vision models: Should use ~5-8GB
- Total system RAM: Monitor doesn't exceed 25GB usage

---

## Post-Deployment Validation

### Success Criteria

- ✅ Server starts without errors
- ✅ Health endpoint responds
- ✅ Text inference works (~70-85 tok/s)
- ✅ Vision inference works (~54-95 tok/s)
- ✅ No errors in logs
- ✅ Memory usage acceptable (<25GB total)

### If All Tests Pass

**Server is deployed and ready for production use.**

You can now:
- Integrate with RAGMT
- Process PDFs with vision
- Run text completions
- Monitor via admin API

### If Any Test Fails

1. Check logs: `tail -50 logs/mlx-inference-server.log`
2. Verify environments: Both venv and venv-vision
3. Check dependencies: pip list in both envs
4. Review error messages
5. Restart server: `./bin/mlx-inference-server-daemon.sh restart`

---

## Maintenance Commands

**Stop server:**
```bash
./bin/mlx-inference-server-daemon.sh stop
```

**Restart server:**
```bash
./bin/mlx-inference-server-daemon.sh restart
```

**Check status:**
```bash
./bin/mlx-inference-server-daemon.sh status
```

**View logs:**
```bash
tail -f logs/mlx-inference-server.log
```

**Check server stats:**
```bash
curl http://localhost:11440/admin/status
```

---

## Deployment Complete

Once all steps pass, the MLX Inference Server is deployed and production-ready on MacBook Air M4.

**Certified by:** Opus 4.5 (Phase 4 review)
**Date:** 2025-12-30
**Version:** Current (vision-enabled)

---

**Questions or Issues?**
- Check logs first
- Review Opus review: `docs/OPUS-REVIEW-VISION-INTEGRATION.md`
- Review vision setup: `docs/VISION-SETUP.md`
- Review API spec: `docs/VISION-API-SPEC.md`
