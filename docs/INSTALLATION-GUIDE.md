# MLX Inference Server - Installation Guide
**Complete setup instructions from zero to running server**

---

## Hardware Requirements

**Supported:**
- Apple Silicon Mac (M1/M2/M3/M4 - any variant)
- 16GB+ RAM (32GB+ recommended for 7B+ models)
- macOS 13.0+ (Ventura or newer)

**Performance by chip (7B-4bit model):**

| Chip | Expected Speed | Max Recommended Model |
|------|----------------|----------------------|
| M1 | 20-30 tok/s | 14B |
| M1 Pro/Max | 30-45 tok/s | 32B |
| M2 | 25-35 tok/s | 14B |
| M2 Pro/Max | 35-50 tok/s | 32B |
| M3 | 30-40 tok/s | 14B |
| M3 Pro/Max | 40-60 tok/s | 32B |
| M4 | 35-45 tok/s | 14B |
| M4 Pro/Max | 50-95 tok/s | 72B |

**Note:** Performance depends on RAM, thermal conditions, and background load.

---

## Prerequisites

**⚠️ IMPORTANT: Complete ALL prerequisites (Steps 1-4) BEFORE moving to Installation Steps.**

Skipping prerequisites (especially Step 4: Python setup) will cause conflicts and errors later.

---

### 1. Check macOS Version

```bash
sw_vers
```

**Required:** macOS 13.0+ (Ventura or newer)

If you're on older macOS, update first:
- System Settings → General → Software Update

---

### 2. Install Homebrew (if not already installed)

Check if you have it:
```bash
which brew
```

If nothing shows up, install Homebrew:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

**After installation, follow the terminal instructions to add Homebrew to your PATH.**

Then verify:
```bash
brew --version
```

---

### 3. Install Git (if not already installed)

```bash
brew install git
```

Verify:
```bash
git --version
```

---

### 4. Install Python 3.12 via Pyenv

**Why pyenv?** It manages Python versions cleanly without breaking system Python.

#### Install pyenv:
```bash
brew install pyenv
```

#### Add to shell config:
```bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
```

#### Restart terminal or reload:
```bash
source ~/.zshrc
```

#### Install Python 3.12:
```bash
pyenv install 3.12.7
```

**This will take 5-10 minutes to compile.**

#### Set as global default:
```bash
pyenv global 3.12.7
```

#### Verify:
```bash
python --version  # Should show Python 3.12.7
which python      # Should show ~/.pyenv/versions/3.12.7/bin/python
```

---

## ✅ Prerequisites Checkpoint

**Before continuing, verify all prerequisites are complete:**

```bash
# Should show Python 3.12.7
python --version

# Should show ~/.pyenv/versions/3.12.7/bin/python
which python

# Should show git version
git --version

# Should show brew version
brew --version
```

**If any of these fail, go back to Prerequisites section above.**

---

## Installation Steps

### Step 1: Choose Installation Location

**Recommendation:** Install in `~/projects/` (NOT in iCloud synced folders)

```bash
mkdir -p ~/projects
cd ~/projects
```

**Why not iCloud?**
- Models are large (4-8GB)
- Server creates temporary files
- iCloud sync will slow things down

---

### Step 2: Clone the Repository

```bash
git clone https://github.com/jameschildress65/mlx-inference-server.git
cd mlx-inference-server
```

Verify:
```bash
pwd  # Should show: /Users/[your-username]/projects/mlx-inference-server
ls   # Should show: src, docs, tests, etc.
```

---

### Step 3: Create Virtual Environment

```bash
python -m venv venv
```

**Activate it:**
```bash
source venv/bin/activate
```

Your prompt should now show `(venv)` at the beginning.

**To deactivate later (don't do this now):**
```bash
deactivate
```

---

### Step 4: Install Dependencies

**This will take 5-10 minutes:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**If you see errors about `posix_ipc`, install it separately:**
```bash
pip install posix_ipc
```

**Verify installation:**
```bash
python -c "import mlx.core as mx; print(f'MLX version: {mx.__version__}')"
python -c "import fastapi; print('FastAPI installed ✓')"
python -c "import posix_ipc; print('posix_ipc installed ✓')"
```

All three should succeed.

---

## Model Setup

### Step 5: Understanding Models

**Model size guide (32GB RAM example):**

| Model Size | Memory Usage | Speed Estimate | Recommendation |
|------------|--------------|----------------|----------------|
| 0.5B-4bit  | ~260 MB      | 150-200 tok/s  | ✅ Excellent   |
| 3B-4bit    | ~1.9 GB      | 60-80 tok/s    | ✅ Great       |
| 7B-4bit    | ~4.1 GB      | 30-40 tok/s    | ✅ Good        |
| 14B-4bit   | ~8.2 GB      | 15-20 tok/s    | ⚠️ Usable     |
| 32B-4bit   | ~18 GB       | 6-10 tok/s     | ⚠️ Slow       |

**Recommendation for testing: Start with 7B-4bit (good balance)**

---

### Step 6: Download a Model

MLX models are hosted on HuggingFace. The server will auto-download on first use, but you can pre-download:

#### Option A: Auto-download (Easiest)
Skip to Step 7 - the server will download automatically when you first load a model.

#### Option B: Pre-download (Recommended for large models)

**Install HuggingFace CLI:**
```bash
pip install huggingface-hub
```

**Download a model:**
```bash
# For 7B model (recommended for testing)
huggingface-cli download mlx-community/Qwen2.5-7B-Instruct-4bit

# For smaller/faster model (if you want quick testing)
huggingface-cli download mlx-community/Qwen2.5-3B-Instruct-4bit

# For best quality (slower)
huggingface-cli download mlx-community/Qwen2.5-14B-Instruct-4bit
```

**This downloads to:** `~/.cache/huggingface/hub/`

**Download time:**
- 7B model: ~5-10 minutes (4.1 GB)
- 3B model: ~2-5 minutes (1.9 GB)
- 14B model: ~10-20 minutes (8.2 GB)

---

## Running the Server

### Step 7: Start the Server

**Make sure you're in the project directory with venv activated:**
```bash
cd ~/projects/mlx-inference-server
source venv/bin/activate  # If not already activated
```

**Start the server:**
```bash
python mlx_inference_server.py
```

**You should see:**
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:11440 (Press CTRL+C to quit)
INFO:     Admin API running on http://0.0.0.0:11441
```

**Keep this terminal window open!** The server is now running.

---

### Step 8: Test in a New Terminal Window

**Open a new terminal window** (don't close the server window).

#### Test 1: Health Check
```bash
curl http://localhost:11440/health
```

**Expected output:**
```json
{"status":"healthy","version":"3.0.0-alpha"}
```

#### Test 2: Load a Model
```bash
curl -X POST "http://localhost:11441/admin/load?model_path=mlx-community/Qwen2.5-7B-Instruct-4bit"
```

**Expected output:**
```json
{
  "status": "success",
  "model_name": "mlx-community/Qwen2.5-7B-Instruct-4bit",
  "memory_gb": 4.1,
  "load_time": 5.2
}
```

**Note:** First load will download the model if you didn't pre-download. This takes 5-10 minutes. Subsequent loads are instant.

#### Test 3: Generate Text
```bash
curl -X POST http://localhost:11440/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "messages": [
      {"role": "user", "content": "Write a haiku about coding"}
    ],
    "max_tokens": 50
  }'
```

**Expected output:**
```json
{
  "id": "chatcmpl-mlx-v3",
  "object": "chat.completion",
  "created": 1735410123,
  "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Code flows like water\nThrough circuits of thought and light\nBugs scatter at dawn"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 18,
    "total_tokens": 33
  }
}
```

**If you see JSON output with a haiku, success! 🎉**

---

## Performance Testing

### Test 4: Measure Performance

**Create a test script:**
```bash
cat > /tmp/test_performance.sh << 'EOF'
#!/bin/bash

echo "Performance Test: 100-token generation"
echo ""

for i in {1..5}; do
    echo "Run $i:"
    START=$(date +%s.%N)

    RESULT=$(curl -s -X POST http://localhost:11440/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "messages": [{"role": "user", "content": "Explain quantum computing in simple terms"}],
        "max_tokens": 100
      }')

    END=$(date +%s.%N)
    ELAPSED=$(echo "$END - $START" | bc)
    TOKENS=$(echo "$RESULT" | python3 -c "import sys, json; print(json.load(sys.stdin)['usage']['completion_tokens'])" 2>/dev/null || echo "0")

    if [ "$TOKENS" != "0" ]; then
        TPS=$(echo "scale=2; $TOKENS / $ELAPSED" | bc)
        echo "  Time: ${ELAPSED}s, Tokens: $TOKENS, Speed: ${TPS} tok/s"
    else
        echo "  Error: No tokens generated"
    fi
done

echo ""
echo "Test complete!"
EOF

chmod +x /tmp/test_performance.sh
```

**Run the test:**
```bash
/tmp/test_performance.sh
```

**Example output (results vary by hardware):**
```
Performance Test: 100-token generation

Run 1:
  Time: 3.2s, Tokens: 98, Speed: 30.6 tok/s
Run 2:
  Time: 3.1s, Tokens: 100, Speed: 32.2 tok/s
...
```

**Typical ranges for 7B-4bit:**
- M1/M2 base: 20-30 tok/s
- M1/M2/M3 Pro/Max: 30-50 tok/s
- M4 Pro/Max: 50-95 tok/s

If you're seeing speeds in these ranges for your chip, **everything is working!**

---

## Common Issues & Troubleshooting

### Issue 1: "Command not found: python"

**Fix:**
```bash
source ~/.zshrc
pyenv global 3.12.7
```

### Issue 2: "ModuleNotFoundError: No module named 'mlx'"

**Fix:**
```bash
cd ~/projects/mlx-inference-server
source venv/bin/activate  # Make sure venv is activated
pip install -r requirements.txt
```

### Issue 3: Server won't start - port already in use

**Check what's using the port:**
```bash
lsof -ti:11440
```

**Kill the process:**
```bash
lsof -ti:11440 | xargs kill -9
```

### Issue 4: Model download fails or is very slow

**Use a different HuggingFace mirror:**
```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download mlx-community/Qwen2.5-7B-Instruct-4bit
```

### Issue 5: "Memory error" or system becomes slow

**You're trying too large a model.** Switch to smaller:
```bash
curl -X POST http://localhost:11441/admin/unload  # Unload current
curl -X POST "http://localhost:11441/admin/load?model_path=mlx-community/Qwen2.5-3B-Instruct-4bit"
```

### Issue 6: Slow performance (<10 tok/s)

**Possible causes:**
1. **Thermal throttling** - Close other apps, let laptop cool
2. **Power mode** - Plug in to power adapter
3. **Background processes** - Check Activity Monitor
4. **Model too large** - Try smaller model

**Check system load:**
```bash
top -l 1 | grep "CPU usage"
```

### Issue 7: Server crashes or hangs

**Check logs in the server terminal window.**

**Restart cleanly:**
```bash
# In server terminal: Ctrl+C to stop
# Clean up any hung processes:
lsof -ti:11440,11441 | xargs kill -9
# Restart:
python mlx_inference_server.py
```

---

## What to Report Back

### Success Metrics

Please report:

1. **Installation time** - How long did Steps 1-6 take?
2. **First model load** - How long to download + load 7B model?
3. **Performance** - What tok/s do you get? (Run the performance test)
4. **System impact** - How much RAM used? (Check Activity Monitor)
5. **Any errors?** - Copy any error messages

### Useful Commands

**Check memory usage:**
```bash
# While server is running with model loaded
ps aux | grep python | grep mlx
# Look at the RSS column (memory in KB)
```

**Check GPU usage:**
```bash
sudo powermetrics --samplers gpu_power -i 1000 -n 1
```

**View server logs:**
Just look at the terminal window where server is running.

---

## Next Steps After Testing

### If Everything Works ✅

You can:
1. Try different models (see Step 6)
2. Test streaming: Add `"stream": true` to requests
3. Integrate with applications (OpenAI-compatible API)
4. Test with webui (if interested)

### If You Find Issues ❌

Please report:
1. Exact error message
2. Which step failed
3. Output of:
   ```bash
   python --version
   pip list | grep -E "mlx|fastapi|pydantic"
   sw_vers
   ```

---

## Quick Reference

### Start Server
```bash
cd ~/projects/mlx-inference-server
source venv/bin/activate
python mlx_inference_server.py
```

### Stop Server
Press `Ctrl+C` in the server terminal window

### Load Model
```bash
curl -X POST "http://localhost:11441/admin/load?model_path=mlx-community/Qwen2.5-7B-Instruct-4bit"
```

### Unload Model
```bash
curl -X POST http://localhost:11441/admin/unload
```

### Check Status
```bash
curl http://localhost:11441/admin/status
```

### Generate Text
```bash
curl -X POST http://localhost:11440/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Your prompt here"}],
    "max_tokens": 100
  }'
```

---

## Expected Performance

| Model | Memory | Speed Range | Best For |
|-------|--------|-------------|----------|
| 0.5B-4bit | 260 MB | 100-200 tok/s | Quick tasks, testing |
| 3B-4bit | 1.9 GB | 50-100 tok/s | Good balance |
| 7B-4bit | 4.1 GB | 20-50 tok/s | **Recommended starting point** |
| 14B-4bit | 8.2 GB | 10-30 tok/s | Better quality |
| 32B-4bit | 18 GB | 5-20 tok/s | High quality (needs 32GB+ RAM) |

**Recommendation:** Start with 7B-4bit for a good balance of speed and quality.

---

## Support

**If you get stuck:**
1. Check this guide's Troubleshooting section
2. Review error messages carefully
3. Report issues with full error output

**Note:** This is an alpha version, so some rough edges are expected. Your testing helps make it better!

---

## Uninstallation (If Needed)

```bash
# Stop server (Ctrl+C)

# Remove virtual environment
cd ~/projects/mlx-inference-server
rm -rf venv/

# Remove models (optional - saves ~10-20 GB)
rm -rf ~/.cache/huggingface/hub/

# Remove project
cd ~/projects
rm -rf mlx-inference-server/
```

---

**Good luck! Report back with your results! 🚀**
