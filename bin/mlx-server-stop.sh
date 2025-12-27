#!/bin/bash
# MLX Server V2 - Stop Background Daemon with Graceful Shutdown

set -e

# Navigate to project directory
cd "$(dirname "$0")/.."

if [ -f .mlx_server.pid ]; then
    PID=$(cat .mlx_server.pid)
    echo "Stopping MLX Server V2 (PID: $PID)..."

    # Send SIGTERM for graceful shutdown
    if ! kill $PID 2>/dev/null; then
        echo "Process already stopped (PID not found)"
        rm .mlx_server.pid
        exit 0
    fi

    # Wait up to 10 seconds for graceful shutdown
    for i in {1..10}; do
        if ! ps -p $PID > /dev/null 2>&1; then
            echo "✓ Server stopped gracefully"
            rm .mlx_server.pid 2>/dev/null || true  # May already be removed by Python atexit
            exit 0
        fi
        sleep 1
    done

    # Force kill if still running after 10 seconds
    echo "⚠ Server did not stop gracefully, forcing shutdown..."
    kill -9 $PID 2>/dev/null || true
    rm .mlx_server.pid 2>/dev/null || true
    echo "✓ Server stopped (forced)"
else
    echo "No PID file found. Trying to find and stop any running mlx_server_extended.py..."
    pkill -f mlx_server_extended.py || echo "No running server found"
fi
