#!/bin/bash
#
# MLX Inference Server - Production Daemon Launcher
#
# Starts server in background with proper logging and PID tracking.
# Production-grade process management for reliable operation.
#

set -euo pipefail

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="$PROJECT_ROOT/venv"
PYTHON="$VENV_PATH/bin/python"
SERVER_SCRIPT="$PROJECT_ROOT/mlx_inference_server.py"
PID_FILE="/tmp/mlx-inference-server.pid"
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/mlx-inference-server.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if server is running
is_running() {
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            return 0
        else
            # Stale PID file
            rm -f "$PID_FILE"
            return 1
        fi
    fi
    return 1
}

# Start server
start_server() {
    log_info "MLX Inference Server - Production Daemon"
    log_info "=========================================="

    # Check if already running
    if is_running; then
        local pid=$(cat "$PID_FILE")
        log_warn "Server already running (PID: $pid)"
        exit 1
    fi

    # Verify venv exists
    if [ ! -f "$PYTHON" ]; then
        log_error "Virtual environment not found at: $VENV_PATH"
        log_error "Run: python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
        exit 1
    fi

    # Create log directory
    mkdir -p "$LOG_DIR"

    # Start server in background
    log_info "Starting inference server..."
    log_info "Python: $PYTHON"
    log_info "Script: $SERVER_SCRIPT"
    log_info "Log: $LOG_FILE"

    nohup "$PYTHON" "$SERVER_SCRIPT" >> "$LOG_FILE" 2>&1 &
    local pid=$!

    # Save PID
    echo "$pid" > "$PID_FILE"

    # Wait briefly and verify it started
    sleep 2

    if ps -p "$pid" > /dev/null 2>&1; then
        log_info "Server started successfully (PID: $pid)"
        log_info "Main API: http://0.0.0.0:11440"
        log_info "Admin API: http://0.0.0.0:11441"
        log_info ""
        log_info "Tail logs: tail -f $LOG_FILE"
        log_info "Stop server: $0 stop"
    else
        log_error "Server failed to start! Check logs: $LOG_FILE"
        rm -f "$PID_FILE"
        exit 1
    fi
}

# Stop server
stop_server() {
    log_info "Stopping V3 server..."

    if ! is_running; then
        log_warn "Server is not running"
        exit 1
    fi

    local pid=$(cat "$PID_FILE")
    log_info "Sending SIGTERM to PID: $pid"

    # Graceful shutdown
    kill -TERM "$pid" 2>/dev/null || true

    # Wait up to 10 seconds for graceful shutdown
    local count=0
    while ps -p "$pid" > /dev/null 2>&1 && [ $count -lt 10 ]; do
        sleep 1
        count=$((count + 1))
    done

    # Force kill if still running
    if ps -p "$pid" > /dev/null 2>&1; then
        log_warn "Graceful shutdown timeout, force killing..."
        kill -KILL "$pid" 2>/dev/null || true
        sleep 1
    fi

    # Cleanup
    rm -f "$PID_FILE"
    log_info "Server stopped"
}

# Show status
status_server() {
    if is_running; then
        local pid=$(cat "$PID_FILE")
        log_info "Server is running (PID: $pid)"

        # Show basic stats
        echo ""
        ps -p "$pid" -o pid,ppid,%cpu,%mem,etime,command | head -2

        # Check API health
        echo ""
        log_info "Checking API health..."
        if curl -s -f http://localhost:11440/health > /dev/null 2>&1; then
            log_info "Main API: ✓ Healthy"
        else
            log_warn "Main API: ✗ Not responding"
        fi

        if curl -s -f http://localhost:11441/admin/health > /dev/null 2>&1; then
            log_info "Admin API: ✓ Healthy"
        else
            log_warn "Admin API: ✗ Not responding"
        fi
    else
        log_warn "Server is not running"
        exit 1
    fi
}

# Restart server
restart_server() {
    log_info "Restarting V3 server..."
    stop_server || true
    sleep 2
    start_server
}

# Show logs
show_logs() {
    if [ -f "$LOG_FILE" ]; then
        tail -f "$LOG_FILE"
    else
        log_error "Log file not found: $LOG_FILE"
        exit 1
    fi
}

# Usage
usage() {
    echo "Usage: $0 {start|stop|restart|status|logs}"
    echo ""
    echo "Commands:"
    echo "  start   - Start V3 server in background"
    echo "  stop    - Stop V3 server gracefully"
    echo "  restart - Restart V3 server"
    echo "  status  - Show server status"
    echo "  logs    - Tail server logs (Ctrl+C to exit)"
    exit 1
}

# Main
case "${1:-}" in
    start)
        start_server
        ;;
    stop)
        stop_server
        ;;
    restart)
        restart_server
        ;;
    status)
        status_server
        ;;
    logs)
        show_logs
        ;;
    *)
        usage
        ;;
esac
