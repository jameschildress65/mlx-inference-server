# MLX Inference Server

Production-grade LLM inference server for Apple Silicon with process isolation and OpenAI-compatible API.

## Why This Exists

Built to address limitations in existing MLX servers: lack of process isolation, unreliable crash recovery, and missing production-grade features. Developed iteratively with Claude Code and refined through multiple code reviews with Claude Opus to ensure correctness and reliability.

**Key principle**: Personal use case driving development, sharing in case helpful to others.

## Features

### Core Capabilities
- **Process Isolation**: Worker processes separated from orchestrator for crash resilience
- **robust ProcessRegistry**: Crash-safe worker lifecycle management with automatic orphan cleanup
- **Vision/Multimodal**: Support for vision-language models with dual-venv architecture
- **POSIX Semaphores**: Cross-process synchronization with proper memory ordering
- **Atomic Operations**: Crash-safe file operations using temp-file + rename pattern
- **Async-Signal-Safe**: Proper signal handling without deadlock risks
- **OpenAI Compatible**: Drop-in replacement for OpenAI API endpoints

### Operational
- **Auto-Configuration**: Detects RAM and applies appropriate settings
- **On-Demand Loading**: Models load automatically on first request
- **Idle Timeout**: Automatic unload to free memory
- **Admin API**: Status monitoring, manual controls, health checks

## Quick Start

**⚡ New to this? See [QUICKSTART.md](QUICKSTART.md) for 10-minute setup guide**

**📖 Detailed instructions: [Installation Guide](docs/INSTALLATION-GUIDE.md)**

**🖼️ Vision/Multimodal support: [Vision Setup Guide](docs/VISION-SETUP.md)**

### Installation

**⚠️ IMPORTANT: Use the automated installer - DO NOT create new install scripts!**

**Requirements:**
- Apple Silicon Mac (M1/M2/M3/M4)
- Python 3.12+ (via pyenv recommended)
- 16GB+ RAM (32GB+ recommended for 7B+ models)

**Automated Installation (Recommended):**
```bash
# Clone repository
git clone https://github.com/jameschildress65/mlx-inference-server.git
cd mlx-inference-server

# Run automated installer (handles everything)
bash install.sh
```

The installer automatically:
- Checks prerequisites and system requirements
- Sets up dual virtual environments (text + vision)
- Installs all dependencies
- Starts the server
- Tests text and vision inference
- **Tests robust ProcessRegistry** (crash-safe worker management)

**Manual Installation (if needed):**
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Start Server

**Production (background):**
```bash
./bin/mlx-inference-server-daemon.sh
```

**Development (foreground):**
```bash
python3 mlx_inference_server.py
```

**Stop:**
```bash
pkill -f mlx_inference_server
```

### Basic Usage

**Chat completion:**
```bash
curl -X POST http://localhost:11440/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

**Check status:**
```bash
curl http://localhost:11441/admin/status
```

## Configuration

Server auto-detects hardware and configures itself based on available RAM:

| RAM Tier | Timeout | Memory Threshold | Use Case |
|----------|---------|------------------|----------|
| High (≥64GB) | 10 min | RAM - 20GB | Server/production |
| Medium (24-63GB) | 5 min | RAM - 8GB | Developer/power user |
| Low (<24GB) | 3 min | RAM - 4GB | Conservative |

**Override with environment variables:**
```bash
export MLX_IDLE_TIMEOUT=600          # Idle timeout (seconds)
export MLX_REQUEST_TIMEOUT=300       # Request timeout (seconds)
export MLX_ADMIN_PORT=11441          # Admin API port
export HF_HOME=~/.cache/huggingface  # Model cache directory
```

## Architecture

### Process Model

```
┌─────────────────────────────────────┐
│ Orchestrator (main process)         │
│ - HTTP server (FastAPI)             │
│ - Request routing                   │
│ - Health monitoring                 │
└──────────────┬──────────────────────┘
               │
               │ IPC (Shared Memory + POSIX Semaphores)
               │
┌──────────────▼──────────────────────┐
│ Worker (isolated process)           │
│ - Model loading                     │
│ - Inference execution               │
│ - Automatic crash recovery          │
└─────────────────────────────────────┘
```

**Why process isolation?**
- Worker crashes don't kill the server
- Clean memory separation
- Resource limits per worker
- Simpler debugging

### IPC Design

**Shared memory ring buffers** for zero-copy data transfer:
- Request queue (orchestrator → worker)
- Response queue (worker → orchestrator)

**POSIX semaphores** for synchronization:
- Prevents memory ordering issues on ARM64
- Cross-process visibility guarantees
- No busy-waiting

## Development Story

### Iteration Process

1. **Initial Implementation**: Built basic functionality
2. **Opus Review #1**: Identified 3 critical correctness issues
3. **Fix Iteration**: Implemented POSIX semaphores, atomic writes, signal-safe handlers
4. **Opus Review #2**: Verified fixes, certified as production-ready
5. **Performance Testing**: Benchmarked against alternatives

### Tools Used

- **Claude Code**: Primary development environment
- **Claude Opus**: Code review and architectural guidance
- **Apple MLX**: Underlying inference framework

### Design Decisions

**Why POSIX semaphores over locks?**
- Kernel-level memory barriers (critical on ARM64)
- Cross-process support
- Signal-safe (no deadlock in signal handlers)

**Why atomic file operations?**
- Eliminates corruption window on crash
- POSIX guarantees for `rename()` atomicity
- Safe even during power loss

**Why pipe-based signals?**
- `os.write()` is async-signal-safe per POSIX
- No locks in signal handlers
- Clean background thread cleanup

## Performance

Tested on Apple Silicon with various model sizes. Example benchmark (compared to Ollama):

| Model | Cold Start | Hot Start (tok/s) | Notes |
|-------|------------|------------------|-------|
| 7-14B | Competitive | ~50 tok/s | Good for interactive |
| 32B | 24% faster | ~23 tok/s | Production sweet spot |
| 72B+ | 42% faster | ~10 tok/s | Scales well |

**Note:** Actual performance depends on your hardware. These are reference points, not guarantees.

## API Reference

See [docs/API-REFERENCE.md](docs/API-REFERENCE.md) for complete API documentation.

**Key endpoints:**
- `POST /v1/chat/completions` - Chat completion (OpenAI compatible)
- `GET /v1/models` - List available models
- `GET /admin/status` - Server status
- `POST /admin/unload` - Manual model unload

## Documentation

- [API Reference](docs/API-REFERENCE.md) - Complete API documentation
- [Vision API Specification](docs/VISION-API-SPEC.md) - Vision/multimodal API for integrators
- [Deployment Guide](docs/DEPLOYMENT-GUIDE.md) - Production deployment
- [Security Best Practices](docs/SECURITY-BEST-PRACTICES.md) - Security hardening
- [Performance Tuning](docs/PERFORMANCE-TUNING.md) - Optimization guide

## Testing

```bash
# Run full test suite
pytest tests/

# Run integration tests only
pytest tests/integration/

# Run with coverage
pytest --cov=src tests/
```

## Known Limitations

- **macOS/Linux only**: POSIX semaphores required
- **Apple Silicon optimized**: Uses MLX framework (Metal)
- **Single worker**: Current implementation (multi-worker planned)

## Credits

- **Apple MLX Team**: For the excellent MLX framework
- **Anthropic**: Claude Code and Claude Opus for development assistance
- **Open Source Community**: For `mlx-lm` and related tools

## License

MIT License - see LICENSE file for details.

## Contributing

This is a personal project shared for community benefit. Issues and PRs welcome, but response time may vary.

---

**Built with Claude Code**
Developed through iterative refinement and code review with Claude Opus
