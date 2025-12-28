# Quick Start Guide

**⚡ Get MLX Inference Server running in 10 minutes**

---

## System Requirements

- **macOS 13.0+** (Ventura or newer)
- **Apple Silicon** (M1/M2/M3/M4)
- **16GB+ RAM** (32GB+ recommended)

---

## Installation (5 commands)

```bash
# 1. Install dependencies
brew install pyenv git

# 2. Install Python 3.12
pyenv install 3.12.7
pyenv global 3.12.7

# 3. Clone and enter directory
git clone https://github.com/jameschildress65/mlx-inference-server.git
cd mlx-inference-server

# 4. Create virtual environment
python -m venv venv
source venv/bin/activate

# 5. Install requirements
pip install -r requirements.txt
```

---

## Start Server (1 command)

```bash
python mlx_inference_server.py
```

**Keep this terminal window open!**

---

## Test (New Terminal Window)

### 1. Health Check
```bash
curl http://localhost:11440/health
```

### 2. Load Model
```bash
curl -X POST http://localhost:11441/admin/load \
  -H "Content-Type: application/json" \
  -d '{"model_name":"mlx-community/Qwen2.5-7B-Instruct-4bit"}'
```

**Wait for download (first time: ~5 min)**

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
