#!/bin/bash
# MLX Inference Server - Background Daemon Starter
# This script runs the server in the background without keeping a terminal open

set -e

# Navigate to project directory
cd "$(dirname "$0")/.."

# Verify mlx-server virtualenv exists
VENV_PATH="$HOME/.pyenv/versions/mlx-server"
if [ ! -d "$VENV_PATH" ]; then
    echo "Error: pyenv virtualenv 'mlx-server' not found at $VENV_PATH"
    echo "Create it with: pyenv virtualenv 3.12.0 mlx-server && cd $(pwd) && pip install -r requirements.txt"
    exit 1
fi

# Apply mlx-lm CPU fix patch if needed
MLX_SERVER_PY="$VENV_PATH/lib/python3.12/site-packages/mlx_lm/server.py"
if [ -f "$MLX_SERVER_PY" ]; then
    if ! grep -q "timeout=0.05" "$MLX_SERVER_PY" 2>/dev/null; then
        echo "Applying mlx-lm CPU fix patch..."
        # Backup original
        cp "$MLX_SERVER_PY" "$MLX_SERVER_PY.backup-$(date +%Y%m%d-%H%M%S)"
        # Apply patch: change get_nowait() to get(timeout=0.05)
        sed -i.tmp 's/return self\.requests\.get_nowait()/return self.requests.get(timeout=0.05)  # PATCH: CPU fix/' "$MLX_SERVER_PY"
        rm -f "$MLX_SERVER_PY.tmp"
        # Clear bytecode cache
        find "$VENV_PATH" -type d -name "__pycache__" -path "*/mlx_lm/*" -exec rm -rf {} + 2>/dev/null || true
        echo "✓ MLX-LM patch applied (fixes 100% CPU idle bug)"
    else
        echo "✓ MLX-LM patch already applied"
    fi
else
    echo "Warning: mlx_lm/server.py not found - patch skipped"
fi

# Start server in background, redirect output to log file
# Auto-detection handled by Python (server_config.py)
# Use explicit virtualenv path (pyenv shims don't work in nohup/non-interactive shells)
nohup ~/.pyenv/versions/mlx-server/bin/python3 mlx_server_extended.py \
    --port 11436 \
    --host 0.0.0.0 \
    >> logs/daemon.log 2>&1 &

# Note: PID file (.mlx_server.pid) is created by Python itself
# This ensures proper cleanup even if script is killed
# Wait up to 5 seconds for Python to create the PID file
for i in {1..10}; do
    if [ -f .mlx_server.pid ]; then
        PYTHON_PID=$(cat .mlx_server.pid)
        echo "✓ MLX Inference Server started in background (PID: $PYTHON_PID)"
        echo "  Logs: logs/daemon.log"
        echo "  Stop: ./bin/mlx-server-stop.sh"
        exit 0
    fi
    sleep 0.5
done

# If we get here, PID file was not created - startup failed
echo "✗ Failed to start MLX Inference Server (no PID file created)"
echo "  Check logs/daemon.log for errors"
exit 1
