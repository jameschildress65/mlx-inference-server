# Quick Start Guide

**⚡ Get MLX Inference Server running in 10 minutes**

**Note:** This guide installs text-only models. For vision/multimodal support, use `bash install.sh` instead (see [Vision Setup Guide](docs/VISION-SETUP.md)).

---

## System Requirements

- **macOS 13.0+** (Ventura or newer)
- **Apple Silicon** (M1/M2/M3/M4)
- **16GB+ RAM** (32GB+ recommended)

---

## Installation

**Recommended location:** `~/projects/` (avoid iCloud-synced folders - models are large)

```bash
# 1. Install dependencies (skip if already installed)
brew install pyenv git

# 2. Configure pyenv for your shell (one-time setup)
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
source ~/.zshrc

# 3. Install Python 3.12
pyenv install 3.12.7
pyenv global 3.12.7

# 4. Clone and enter directory
mkdir -p ~/projects && cd ~/projects
git clone https://github.com/jameschildress65/mlx-inference-server.git
cd mlx-inference-server

# 5. Create virtual environment
python -m venv venv
source venv/bin/activate

# 6. Install requirements
pip install -r requirements.txt
```

**Verify Python is correct before step 5:**
```bash
which python  # Should show: ~/.pyenv/versions/3.12.7/bin/python
```

---

## Start Server (1 command)

```bash
python mlx_inference_server.py
```

**Keep this terminal window open!**

---

## Test (New Terminal Window)

**⚠️ TEXT MODELS ONLY:** This setup supports text-only models. Vision models will fail with `Vision environment not found`. For vision support, use `bash install.sh` instead (see [Vision Setup](docs/VISION-SETUP.md)).

### 1. Health Check
```bash
curl http://localhost:11440/health
```

### 2. Load Model (Text Only)
```bash
curl -X POST "http://localhost:11441/admin/load?model_path=mlx-community/Qwen2.5-7B-Instruct-4bit"
```

**Wait for download (first time: ~5 min)**

**Note:** Vision models (e.g., Qwen2.5-VL-7B-Instruct-4bit) require `venv-vision`. Run `bash install.sh` for vision support.

### 3. Generate
```bash
curl -X POST http://localhost:11440/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

**If you see JSON with a response, it works! 🎉**

---

## Performance Expectations

| Hardware | 7B Model Speed |
|----------|----------------|
| M1 Pro 32GB | 28-38 tok/s |
| M4 Air 32GB | 15-20 tok/s |
| M4 Max 128GB | 82-94 tok/s |

---

## Troubleshooting

**Port already in use?**
```bash
lsof -ti:11440,11441 | xargs kill -9
```

**Need detailed help?**
See `docs/INSTALLATION-GUIDE.md`

---

## Stop Server

Press `Ctrl+C` in server terminal window

---

**Full docs:** `docs/INSTALLATION-GUIDE.md`
