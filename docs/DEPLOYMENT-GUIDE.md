# MLX Server V3 - Deployment Guide

**Version**: 3.0.0-alpha
**Date**: 2025-12-24
**Target**: Production deployment on Apple Silicon

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Starting the Server](#starting-the-server)
5. [Verification](#verification)
6. [Monitoring](#monitoring)
7. [Troubleshooting](#troubleshooting)
8. [Production Recommendations](#production-recommendations)

---

## Prerequisites

### Hardware Requirements

**Minimum**:
- Apple Silicon Mac (M1, M2, M3, M4)
- 16 GB RAM
- 20 GB free disk space

**Recommended**:
- Apple Silicon Mac (M4 Max/Ultra preferred)
- 32 GB+ RAM
- 50 GB+ free disk space (for models)

**Tested Configurations**:
- ✅ Mac Studio M4 Max (128GB RAM) - Excellent
- ✅ MacBook Air M4 (32GB RAM) - Good
- ✅ Mac Mini M4 (16GB RAM) - Adequate
- ✅ Mac Mini M1 (16GB RAM) - Adequate

### Software Requirements

- **macOS**: Sonoma 14.0+ or Sequoia 15.0+
- **Python**: 3.11 or 3.12 (via pyenv)
- **Git**: For cloning repository
- **Homebrew**: For installing dependencies (optional)

---

## Installation

### 1. Install Python via pyenv

```bash
# Install pyenv if not already installed
brew install pyenv

# Install Python 3.12
pyenv install 3.12.0

# Set global Python version
pyenv global 3.12.0

# Verify
python --version  # Should show 3.12.0
```

### 2. Clone Repository

```bash
cd ~/Documents/projects/utilities
git clone https://github.com/jameschildress65/utilities.git
cd mlx-server

# Checkout V3 branch
git checkout v3-process-isolation
```

### 3. Create Virtual Environment

```bash
# Create venv
python -m venv venv

# Activate
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### 4. Install Dependencies

```bash
# Install all requirements
pip install -r requirements.txt

# Verify MLX installation
python -c "import mlx.core as mx; print(f'MLX version: {mx.__version__}')"
```

### 5. Configure Model Cache (Optional)

```bash
# Set HuggingFace cache to custom location (if desired)
export HF_HOME="/path/to/your/model/cache"

# Or use default (~/.cache/huggingface)
# No action needed - will auto-create
```

---

## Configuration

### Auto-Detection (Recommended)

V3 auto-detects your hardware and configures itself:

```python
# config/server_config.py will auto-detect:
# - Total RAM
# - Chip model (M1/M2/M3/M4)
# - Machine type (Studio/Air/Mini/etc)
```

No manual configuration needed for most users!

### Manual Configuration (Advanced)

If you need to override defaults:

```python
# Edit config/server_config.py

# Example: Increase idle timeout
config.idle_timeout_seconds = 900  # 15 minutes (default: 600)

# Example: Change ports
config.main_port = 11440  # Default
config.admin_port = 11441  # Default
```

### Environment Variables

```bash
# Optional environment variables
export MLX_LOG_LEVEL=INFO          # DEBUG|INFO|WARNING|ERROR
export HF_HOME=/path/to/models     # Model cache location
export MLX_IDLE_TIMEOUT=600        # Idle timeout in seconds
```

---

## Starting the Server

### Option 1: Daemon Mode (Production)

```bash
# Start server as background daemon
./bin/mlx-inference-server-daemon.sh start

# Check status
./bin/mlx-inference-server-daemon.sh status

# View logs
./bin/mlx-inference-server-daemon.sh logs

# Stop server
./bin/mlx-inference-server-daemon.sh stop

# Restart server
./bin/mlx-inference-server-daemon.sh restart
```

**Daemon Features**:
- ✅ PID file tracking (`/tmp/mlx-inference-server.pid`)
- ✅ Log rotation (`logs/mlx-inference-server.log`)
- ✅ Graceful shutdown (SIGTERM → SIGKILL)
- ✅ Health check integration
- ✅ Auto-restart on crash (via systemd/launchd if configured)

### Option 2: Foreground Mode (Development)

```bash
# Activate venv
source venv/bin/activate

# Run server in foreground
python mlx_inference_server.py

# Stop with Ctrl+C
```

**Foreground Features**:
- ✅ Real-time log output
- ✅ Easy debugging
- ✅ Immediate shutdown

---

## Verification

### 1. Check Server Health

```bash
# Main API health check
curl http://localhost:11440/health

# Expected response:
# {"status":"healthy","version":"3.0.0-alpha"}

# Admin API health check
curl http://localhost:11441/admin/health

# Expected response:
# {"status":"healthy","worker_status":"no_worker","version":"3.0.0-alpha"}
```

### 2. Check Admin Status

```bash
curl http://localhost:11441/admin/status

# Expected response:
# {
#   "status": "running",
#   "version": "3.0.0-alpha",
#   "ports": {"main": 11440, "admin": 11441},
#   "model": {"loaded": false, "name": null, "memory_gb": 0.0},
#   "worker": {"healthy": false, "status": "no_worker"},
#   "config": {
#     "machine_type": "high-memory",
#     "total_ram_gb": 128.0,
#     "idle_timeout_seconds": 600
#   }
# }
```

### 3. Load a Model

```bash
curl -X POST "http://localhost:11441/admin/load?model_path=mlx-community/Qwen2.5-0.5B-Instruct-4bit"

# Expected response:
# {
#   "status": "success",
#   "model_name": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
#   "memory_gb": 0.26,
#   "load_time": 2.3
# }
```

### 4. Generate Completion

```bash
curl -X POST http://localhost:11440/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    "prompt": "What is MLX?",
    "max_tokens": 50,
    "temperature": 0.7
  }'

# Expected: OpenAI-compatible JSON response with generated text
```

### 5. Test Streaming

```bash
curl -X POST http://localhost:11440/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    "prompt": "Count to 5:",
    "max_tokens": 30,
    "stream": true
  }'

# Expected: Server-Sent Events (SSE) stream
# data: {"choices":[{"text":"1","index":0,"finish_reason":null}]}
# data: {"choices":[{"text":",","index":0,"finish_reason":null}]}
# ...
```

---

## Monitoring

### Log Files

```bash
# Daemon logs
tail -f logs/mlx-inference-server.log

# System logs (macOS)
log stream --predicate 'process == "python"' --level debug
```

### Process Monitoring

```bash
# Check main process
ps aux | grep mlx_inference_server

# Check worker subprocess
ps aux | grep "python -m src.worker"

# Memory usage
ps -o pid,ppid,%cpu,%mem,rss,command -p $(cat /tmp/mlx-inference-server.pid)
```

### Health Monitoring

```bash
# Continuous health check (every 10 seconds)
while true; do
  curl -s http://localhost:11440/health | jq .
  sleep 10
done
```

### Metrics Export (Future)

V3 will support Prometheus metrics export in Phase 6:
- Request count/rate
- Generation latency
- Model memory usage
- Worker health status

---

## Troubleshooting

### Server Won't Start

**Symptom**: `./bin/mlx-inference-server-daemon.sh start` fails

**Solutions**:
```bash
# 1. Check if port is in use
lsof -i :11440
lsof -i :11441

# 2. Kill conflicting processes
kill <PID>

# 3. Check logs
cat logs/mlx-inference-server.log

# 4. Verify venv
source venv/bin/activate
python -c "import mlx.core"
```

### Model Load Fails

**Symptom**: Admin load returns error

**Solutions**:
```bash
# 1. Check disk space
df -h

# 2. Check HuggingFace cache
ls -lh $HF_HOME/hub/

# 3. Manually download model
python -c "from huggingface_hub import snapshot_download; snapshot_download('mlx-community/Qwen2.5-0.5B-Instruct-4bit')"

# 4. Check MLX version
pip show mlx-lm
```

### Worker Crashes

**Symptom**: "Worker crashed" errors in logs

**Solutions**:
```bash
# 1. Check worker stderr
# (logged to mlx-inference-server.log)

# 2. Test worker manually
python -m src.worker mlx-community/Qwen2.5-0.5B-Instruct-4bit

# 3. Check system resources
top -o MEM

# 4. Reduce model size
# Use 4-bit quantized models for 16GB RAM machines
```

### Memory Not Freed

**Symptom**: Memory usage keeps growing

**Solutions**:
```bash
# 1. Verify process isolation
ps aux | grep "python -m src.worker"
# Should show separate worker process

# 2. Check unload actually kills worker
curl -X POST http://localhost:11441/admin/unload
ps aux | grep "python -m src.worker"
# Should show NO worker

# 3. Check for Python memory leaks (rare)
# Run memory cleanup tests
pytest tests/integration/test_memory_cleanup.py -v
```

### Idle Timeout Not Working

**Symptom**: Model not auto-unloaded

**Solutions**:
```bash
# 1. Check IdleMonitor is running
curl http://localhost:11441/admin/status | jq .

# 2. Check idle timeout config
# Default: 600 seconds (10 minutes)

# 3. Verify no active requests
curl http://localhost:11441/admin/status | jq '.worker.healthy'

# 4. Check logs for auto-unload messages
grep "Auto-unload" logs/mlx-inference-server.log
```

---

## Production Recommendations

### 1. Use Daemon Mode

```bash
# Always use daemon script for production
./bin/mlx-inference-server-daemon.sh start

# Add to startup (macOS launchd)
# See: docs/LAUNCHD-SETUP.md (TBD)
```

### 2. Configure Firewall

```bash
# Allow only local connections (default)
# Ports 11440 and 11441 bound to 0.0.0.0

# For remote access, use SSH tunnel or VPN
ssh -L 11440:localhost:11440 user@machine
```

### 3. Set Resource Limits

```bash
# Limit max file descriptors (if needed)
ulimit -n 4096

# Monitor with Activity Monitor
# V3 should use 0-5% CPU when idle
```

### 4. Model Selection

**For 16GB RAM**:
- ✅ Qwen2.5-0.5B-Instruct-4bit (~260 MB)
- ✅ Qwen2.5-1.5B-Instruct-4bit (~900 MB)
- ⚠️ Qwen2.5-3B-Instruct-4bit (~1.8 GB)

**For 32GB RAM**:
- ✅ Qwen2.5-7B-Instruct-4bit (~4 GB)
- ⚠️ Qwen2.5-14B-Instruct-4bit (~8 GB)

**For 64GB+ RAM**:
- ✅ Qwen2.5-32B-Instruct-4bit (~18 GB)
- ✅ Qwen2.5-72B-Instruct-4bit (~40 GB)

### 5. Backup and Recovery

```bash
# Backup configuration
cp -r config/ config.backup/

# Backup logs
tar -czf logs-$(date +%Y%m%d).tar.gz logs/

# Recovery: just restart
./bin/mlx-inference-server-daemon.sh restart
```

### 6. Security Best Practices

- ✅ Bind to localhost only (default)
- ✅ Use SSH tunnels for remote access
- ✅ Never expose to public internet
- ✅ Use firewall rules
- ❌ Don't run as root
- ❌ Don't disable health checks

---

## API Reference

See: `docs/API.md` for full API documentation

**Quick Reference**:
- `POST /v1/completions` - Text completion (OpenAI compatible)
- `POST /v1/chat/completions` - Chat completion
- `GET /v1/models` - List loaded models
- `POST /admin/load` - Load model
- `POST /admin/unload` - Unload model
- `GET /admin/status` - Full system status

---

## Next Steps

After deployment:
1. ✅ Verify health checks
2. ✅ Run integration tests
3. ✅ Monitor logs for errors
4. ✅ Test idle timeout (wait 10+ minutes)
5. ✅ Test memory cleanup (load/unload/check memory)
6. 📋 Setup monitoring dashboard (Phase 6)
7. 📋 Configure metrics export (Phase 6)

---

**Document Version**: 1.0
**Last Updated**: 2025-12-24
**Maintainer**: MLX Server V3 Team
