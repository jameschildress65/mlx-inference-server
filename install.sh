#!/bin/bash
#
# MLX Inference Server - Smart Installation Script
# Handles fresh installs, upgrades, and cleanup
#

set -e  # Exit on any error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Track what we've done for rollback
CREATED_VENV=false
CREATED_VENV_VISION=false
STARTED_SERVER=false

# Cleanup function for errors
cleanup_on_error() {
    echo ""
    echo -e "${RED}Installation failed! Rolling back changes...${NC}"

    if [ "$STARTED_SERVER" = true ]; then
        echo "  → Stopping server..."
        ./bin/mlx-inference-server-daemon.sh stop 2>/dev/null || true
    fi

    if [ "$CREATED_VENV" = true ]; then
        echo "  → Removing incomplete main venv..."
        rm -rf venv
    fi

    if [ "$CREATED_VENV_VISION" = true ]; then
        echo "  → Removing incomplete vision venv..."
        rm -rf venv-vision
    fi

    echo -e "${RED}Rollback complete. Please check the error above and try again.${NC}"
    exit 1
}

trap cleanup_on_error ERR

# Header
echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        MLX Inference Server - Smart Installer              ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check prerequisites
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Step 1: Checking Prerequisites${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Check macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo -e "${RED}✗ ERROR: This requires macOS (detected: $OSTYPE)${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Operating System: macOS${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ ERROR: python3 not found${NC}"
    echo "  Install with: brew install python@3.12"
    exit 1
fi
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓ Python: $PYTHON_VERSION${NC}"

# Check Apple Silicon
CHIP=$(system_profiler SPHardwareDataType | grep "Chip:" | awk '{print $2, $3}')
if [[ ! "$CHIP" =~ "Apple" ]]; then
    echo -e "${RED}✗ ERROR: Apple Silicon required (M1/M2/M3/M4)${NC}"
    echo "  Detected: $CHIP"
    exit 1
fi
echo -e "${GREEN}✓ Chip: $CHIP${NC}"

# Check RAM
RAM=$(system_profiler SPHardwareDataType | grep "Memory:" | awk '{print $2, $3}')
RAM_GB=$(echo "$RAM" | awk '{print $1}')
if [ "$RAM_GB" -lt 16 ]; then
    echo -e "${YELLOW}⚠ Warning: ${RAM} detected (16GB+ recommended)${NC}"
else
    echo -e "${GREEN}✓ RAM: $RAM${NC}"
fi

echo ""

# Configure model cache location (HF_HOME)
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Step 2: Configure Model Storage${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

echo ""
echo "Models can be large (4-20GB each). Where should they be stored?"
echo ""

# Check if HF_HOME is already set
if [ -n "$HF_HOME" ]; then
    echo -e "${GREEN}✓ HF_HOME already set: $HF_HOME${NC}"
    echo ""
    echo "  ${GREEN}1)${NC} Keep current setting"
    echo "  ${YELLOW}2)${NC} Change to different location"
    echo ""
    read -p "Enter choice [1-2]: " hf_choice

    if [ "$hf_choice" = "2" ]; then
        HF_HOME=""  # Reset to trigger new selection
    fi
fi

if [ -z "$HF_HOME" ]; then
    DEFAULT_CACHE="$HOME/.cache/huggingface"

    echo "  ${GREEN}1)${NC} Default: $DEFAULT_CACHE (Recommended)"
    echo "  ${YELLOW}2)${NC} Custom location (external drive, etc.)"
    echo ""
    read -p "Enter choice [1-2]: " hf_choice

    case $hf_choice in
        1)
            HF_HOME="$DEFAULT_CACHE"
            echo -e "${GREEN}→ Using default: $HF_HOME${NC}"
            ;;
        2)
            echo ""
            read -p "Enter full path for model storage: " custom_path
            # Expand ~ if used
            custom_path="${custom_path/#\~/$HOME}"

            # Validate path exists or can be created
            if [ -d "$custom_path" ]; then
                HF_HOME="$custom_path"
                echo -e "${GREEN}→ Using: $HF_HOME${NC}"
            else
                echo "  Directory doesn't exist. Create it? [y/n]: "
                read create_dir
                if [ "$create_dir" = "y" ] || [ "$create_dir" = "Y" ]; then
                    mkdir -p "$custom_path"
                    HF_HOME="$custom_path"
                    echo -e "${GREEN}→ Created and using: $HF_HOME${NC}"
                else
                    echo -e "${YELLOW}→ Using default instead: $DEFAULT_CACHE${NC}"
                    HF_HOME="$DEFAULT_CACHE"
                fi
            fi
            ;;
        *)
            HF_HOME="$DEFAULT_CACHE"
            echo -e "${GREEN}→ Using default: $HF_HOME${NC}"
            ;;
    esac
fi

# Ensure HF_HOME directory exists
mkdir -p "$HF_HOME"

# Export for this session
export HF_HOME="$HF_HOME"

# Check if already in shell config
SHELL_CONFIG="$HOME/.zshrc"
if grep -q "export HF_HOME=" "$SHELL_CONFIG" 2>/dev/null; then
    echo -e "${GREEN}✓ HF_HOME already in $SHELL_CONFIG${NC}"
else
    echo ""
    echo "Add HF_HOME to your shell config for future sessions? [y/n]: "
    read add_to_shell
    if [ "$add_to_shell" = "y" ] || [ "$add_to_shell" = "Y" ]; then
        echo "" >> "$SHELL_CONFIG"
        echo "# MLX Inference Server - Model cache location" >> "$SHELL_CONFIG"
        echo "export HF_HOME=\"$HF_HOME\"" >> "$SHELL_CONFIG"
        echo -e "${GREEN}✓ Added to $SHELL_CONFIG${NC}"
    fi
fi

echo ""

# Select starter model
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Step 3: Download Starter Model${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

echo ""
echo "Download a model now? (Can also download later on first use)"
echo ""
echo "  ${GREEN}1)${NC} Qwen2.5-3B-Instruct-4bit (~1.9GB, fast download, good for testing)"
echo "  ${YELLOW}2)${NC} Qwen2.5-7B-Instruct-4bit (~4.1GB, recommended for regular use)"
echo "  ${BLUE}3)${NC} Skip (download on first use)"
echo ""
read -p "Enter choice [1-3]: " model_choice

DOWNLOAD_MODEL=""
case $model_choice in
    1)
        DOWNLOAD_MODEL="mlx-community/Qwen2.5-3B-Instruct-4bit"
        echo -e "${GREEN}→ Will download 3B model${NC}"
        ;;
    2)
        DOWNLOAD_MODEL="mlx-community/Qwen2.5-7B-Instruct-4bit"
        echo -e "${GREEN}→ Will download 7B model${NC}"
        ;;
    3|*)
        echo -e "${BLUE}→ Skipping model download${NC}"
        ;;
esac

# Download model if selected (before venv setup so user sees progress)
if [ -n "$DOWNLOAD_MODEL" ]; then
    echo ""
    echo "  → Downloading $DOWNLOAD_MODEL..."
    echo "  → This may take a few minutes depending on your connection"
    echo ""

    # Use huggingface-cli if available, otherwise will download on first use
    if command -v huggingface-cli &> /dev/null; then
        huggingface-cli download "$DOWNLOAD_MODEL" --quiet
        echo -e "${GREEN}  ✓ Model downloaded to $HF_HOME${NC}"
    else
        # Will install huggingface-hub in venv later, download then
        echo -e "${YELLOW}  → huggingface-cli not found, will download after venv setup${NC}"
        DOWNLOAD_AFTER_VENV=true
    fi
fi

echo ""

# Check current installation status
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Step 4: Checking Current Installation${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

SERVER_RUNNING=false
VENV_EXISTS=false
VENV_VISION_EXISTS=false

# Check if server is running
if pgrep -f "mlx-inference-server" > /dev/null; then
    SERVER_PID=$(pgrep -f "mlx-inference-server")
    echo -e "${YELLOW}⚠ Server is currently running (PID: $SERVER_PID)${NC}"
    SERVER_RUNNING=true
else
    echo -e "${GREEN}✓ No server currently running${NC}"
fi

# Check if venvs exist
if [ -d "venv" ]; then
    VENV_SIZE=$(du -sh venv 2>/dev/null | awk '{print $1}')
    echo -e "${YELLOW}⚠ Main venv exists ($VENV_SIZE)${NC}"
    VENV_EXISTS=true
else
    echo -e "${GREEN}✓ No main venv found${NC}"
fi

if [ -d "venv-vision" ]; then
    VENV_VISION_SIZE=$(du -sh venv-vision 2>/dev/null | awk '{print $1}')
    echo -e "${YELLOW}⚠ Vision venv exists ($VENV_VISION_SIZE)${NC}"
    VENV_VISION_EXISTS=true
else
    echo -e "${GREEN}✓ No vision venv found${NC}"
fi

echo ""

# Determine installation mode
INSTALL_MODE=""

if [ "$SERVER_RUNNING" = true ] || [ "$VENV_EXISTS" = true ] || [ "$VENV_VISION_EXISTS" = true ]; then
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}Existing Installation Detected${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "Choose installation mode:"
    echo ""
    echo "  ${GREEN}1)${NC} Clean Install (remove everything and start fresh)"
    echo "  ${YELLOW}2)${NC} Upgrade (keep existing, just update dependencies)"
    echo "  ${RED}3)${NC} Cancel"
    echo ""
    read -p "Enter choice [1-3]: " choice

    case $choice in
        1)
            INSTALL_MODE="clean"
            echo -e "${GREEN}→ Clean install selected${NC}"
            ;;
        2)
            INSTALL_MODE="upgrade"
            echo -e "${YELLOW}→ Upgrade selected${NC}"
            ;;
        3)
            echo -e "${BLUE}Installation cancelled${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid choice. Exiting.${NC}"
            exit 1
            ;;
    esac
else
    INSTALL_MODE="clean"
    echo -e "${GREEN}→ Fresh installation (no existing components)${NC}"
fi

echo ""

# Stop server if running
if [ "$SERVER_RUNNING" = true ]; then
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}Step 5: Stopping Existing Server${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    ./bin/mlx-inference-server-daemon.sh stop
    sleep 2

    # Verify stopped
    if pgrep -f "mlx-inference-server" > /dev/null; then
        echo -e "${RED}✗ Server still running, force killing...${NC}"
        pkill -9 -f "mlx-inference-server"
        sleep 1
    fi

    echo -e "${GREEN}✓ Server stopped${NC}"
    echo ""
fi

# Clean install: remove venvs
if [ "$INSTALL_MODE" = "clean" ]; then
    if [ "$VENV_EXISTS" = true ] || [ "$VENV_VISION_EXISTS" = true ]; then
        echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${YELLOW}Step 6: Removing Old Virtual Environments${NC}"
        echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

        if [ "$VENV_EXISTS" = true ]; then
            echo "  → Removing venv/ ($VENV_SIZE)..."
            rm -rf venv
            echo -e "${GREEN}  ✓ Main venv removed${NC}"
        fi

        if [ "$VENV_VISION_EXISTS" = true ]; then
            echo "  → Removing venv-vision/ ($VENV_VISION_SIZE)..."
            rm -rf venv-vision
            echo -e "${GREEN}  ✓ Vision venv removed${NC}"
        fi

        echo ""
    fi
fi

# Install main venv
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Step 7: Setting Up Main Virtual Environment${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

if [ ! -d "venv" ] || [ "$INSTALL_MODE" = "clean" ]; then
    echo "  → Creating venv..."
    python3 -m venv venv
    CREATED_VENV=true
    echo -e "${GREEN}  ✓ Virtual environment created${NC}"
fi

source venv/bin/activate

echo "  → Upgrading pip..."
pip install --upgrade pip -q

if [ "$INSTALL_MODE" = "clean" ]; then
    echo "  → Installing dependencies (5-10 minutes)..."
    pip install -r requirements.txt -q
else
    echo "  → Updating dependencies..."
    pip install --upgrade -r requirements.txt -q
fi

echo "  → Verifying installation..."
python -c "import mlx.core as mx; print(f'    ✓ MLX: {mx.__version__}')"
python -c "import fastapi; print('    ✓ FastAPI')"
python -c "import posix_ipc; print('    ✓ posix_ipc')"
python -c "from PIL import Image; print('    ✓ Pillow')"
python -c "import fitz; print('    ✓ PyMuPDF')"

deactivate
echo -e "${GREEN}✓ Main venv ready${NC}"
echo ""

# Download model if deferred from earlier
if [ "$DOWNLOAD_AFTER_VENV" = true ] && [ -n "$DOWNLOAD_MODEL" ]; then
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}Downloading Selected Model${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "  → Downloading $DOWNLOAD_MODEL..."
    echo "  → This may take a few minutes..."
    echo ""

    source venv/bin/activate
    pip install huggingface-hub -q
    huggingface-cli download "$DOWNLOAD_MODEL"
    deactivate

    echo -e "${GREEN}  ✓ Model downloaded to $HF_HOME${NC}"
    echo ""
fi

# Install vision venv
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Step 8: Setting Up Vision Virtual Environment${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

if [ ! -d "venv-vision" ] || [ "$INSTALL_MODE" = "clean" ]; then
    echo "  → Creating venv-vision..."
    python3 -m venv venv-vision
    CREATED_VENV_VISION=true
    echo -e "${GREEN}  ✓ Virtual environment created${NC}"
fi

source venv-vision/bin/activate

echo "  → Upgrading pip..."
pip install --upgrade pip -q

if [ "$INSTALL_MODE" = "clean" ]; then
    echo "  → Installing vision dependencies (10-15 minutes)..."
    echo "  → (PyTorch is large, please be patient)"
    pip install -r requirements-vision.txt -q
else
    echo "  → Updating vision dependencies..."
    pip install --upgrade -r requirements-vision.txt -q
fi

echo "  → Verifying installation..."
python -c "import mlx_vlm; print('    ✓ mlx-vlm')"
python -c "from PIL import Image; print('    ✓ Pillow')"
python -c "import torch; print('    ✓ PyTorch')"
python -c "import torchvision; print('    ✓ Torchvision')"

deactivate
echo -e "${GREEN}✓ Vision venv ready${NC}"
echo ""

# Start server
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Step 9: Starting Server${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

./bin/mlx-inference-server-daemon.sh start
STARTED_SERVER=true

echo "  → Waiting for initialization..."
sleep 5

# Verify health
echo "  → Checking health endpoints..."
MAIN_HEALTH=$(curl -s http://localhost:11440/health 2>/dev/null || echo "failed")
ADMIN_HEALTH=$(curl -s http://localhost:11441/admin/health 2>/dev/null || echo "failed")

if [[ "$MAIN_HEALTH" == *"healthy"* ]]; then
    echo -e "${GREEN}  ✓ Main API: http://localhost:11440${NC}"
else
    echo -e "${RED}  ✗ Main API not responding${NC}"
    exit 1
fi

if [[ "$ADMIN_HEALTH" == *"healthy"* ]] || [[ "$ADMIN_HEALTH" == *"degraded"* ]]; then
    echo -e "${GREEN}  ✓ Admin API: http://localhost:11441${NC}"
else
    echo -e "${RED}  ✗ Admin API not responding${NC}"
    exit 1
fi

echo ""

# Test text inference
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Step 10: Testing Text Inference${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Use downloaded model if available, otherwise use small 0.5B for quick test
TEST_MODEL="${DOWNLOAD_MODEL:-mlx-community/Qwen2.5-0.5B-Instruct-4bit}"
echo "  → Testing with: $TEST_MODEL"
echo "  → Sending test request..."
RESPONSE=$(curl -s -X POST http://localhost:11440/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$TEST_MODEL\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Say hello in 3 words\"}],
    \"max_tokens\": 10
  }" 2>/dev/null)

if [[ "$RESPONSE" == *"assistant"* ]]; then
    TOKENS_PER_SEC=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['usage'].get('tokens_per_sec', 'N/A'))" 2>/dev/null || echo "N/A")
    CONTENT=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "")
    echo -e "${GREEN}  ✓ Text inference working${NC}"
    echo "    Speed: $TOKENS_PER_SEC tok/s"
    echo "    Response: \"$CONTENT\""
else
    echo -e "${RED}  ✗ Text inference failed${NC}"
    exit 1
fi

echo ""

# Test ProcessRegistry (robust)
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Step 11: Testing ProcessRegistry${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Test 1: Registry initialization
echo "  → Checking ProcessRegistry initialization..."
if grep -q "ProcessRegistry initialized" logs/mlx-inference-server.log; then
    echo -e "${GREEN}  ✓ ProcessRegistry initialized${NC}"
else
    echo -e "${RED}  ✗ ProcessRegistry not initialized${NC}"
    exit 1
fi

# Test 2: Worker registration
echo "  → Testing worker registration with: $TEST_MODEL"
curl -s -X POST "http://localhost:11441/admin/load?model_path=$TEST_MODEL" > /dev/null
sleep 5

if [ -f "/tmp/mlx-server/worker_registry.json" ]; then
    WORKER_COUNT=$(cat /tmp/mlx-server/worker_registry.json | python3 -c "import sys, json; print(len(json.load(sys.stdin).get('workers', {})))" 2>/dev/null || echo "0")
    if [ "$WORKER_COUNT" == "1" ]; then
        WORKER_PID=$(cat /tmp/mlx-server/worker_registry.json | python3 -c "import sys, json; w=json.load(sys.stdin).get('workers', {}); print(list(w.keys())[0] if w else '')" 2>/dev/null)
        echo -e "${GREEN}  ✓ Worker registered (PID: $WORKER_PID)${NC}"
    else
        echo -e "${RED}  ✗ Worker registration failed${NC}"
        exit 1
    fi
else
    echo -e "${RED}  ✗ Registry file not found${NC}"
    exit 1
fi

# Test 3: Orphan cleanup (critical test)
echo "  → Testing orphan cleanup (simulated crash)..."
SERVER_PID=$(cat /tmp/mlx-inference-server.pid 2>/dev/null)
WORKER_PID_BEFORE=$(cat /tmp/mlx-server/worker_registry.json | python3 -c "import sys, json; w=json.load(sys.stdin).get('workers', {}); print(list(w.keys())[0] if w else '')" 2>/dev/null)

# Simulate crash
kill -9 $SERVER_PID 2>/dev/null || true
sleep 2

# Restart
./bin/mlx-inference-server-daemon.sh start > /dev/null 2>&1
sleep 5

# Check if orphan was detected and cleaned
if grep -q "Found 1 orphaned workers" logs/mlx-inference-server.log 2>/dev/null; then
    if ! ps -p $WORKER_PID_BEFORE > /dev/null 2>&1; then
        echo -e "${GREEN}  ✓ Orphan cleanup working (killed worker $WORKER_PID_BEFORE)${NC}"
    else
        echo -e "${YELLOW}  ⚠ Orphan detected but not killed${NC}"
    fi
else
    if ! ps -p $WORKER_PID_BEFORE > /dev/null 2>&1; then
        echo -e "${GREEN}  ✓ Worker cleaned up (exited gracefully)${NC}"
    else
        echo -e "${RED}  ✗ Orphan cleanup failed - worker $WORKER_PID_BEFORE still running${NC}"
        exit 1
    fi
fi

# Verify registry is clean
FINAL_WORKER_COUNT=$(cat /tmp/mlx-server/worker_registry.json | python3 -c "import sys, json; print(len(json.load(sys.stdin).get('workers', {})))" 2>/dev/null || echo "0")
if [ "$FINAL_WORKER_COUNT" == "0" ]; then
    echo -e "${GREEN}  ✓ Registry clean after restart${NC}"
else
    echo -e "${RED}  ✗ Registry not clean (contains $FINAL_WORKER_COUNT workers)${NC}"
    exit 1
fi

echo ""

# Success summary
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║       Installation Successful! (robust) ✓             ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${CYAN}Installation Summary:${NC}"
echo -e "  ${GREEN}✓${NC} Main venv installed ($(du -sh venv 2>/dev/null | awk '{print $1}'))"
echo -e "  ${GREEN}✓${NC} Vision venv installed ($(du -sh venv-vision 2>/dev/null | awk '{print $1}'))"
echo -e "  ${GREEN}✓${NC} Model cache: $HF_HOME"
if [ -n "$DOWNLOAD_MODEL" ]; then
    echo -e "  ${GREEN}✓${NC} Model downloaded: $DOWNLOAD_MODEL"
fi
echo -e "  ${GREEN}✓${NC} Server running (PID: $(pgrep -f mlx-inference-server))"
echo -e "  ${GREEN}✓${NC} Text inference tested ($TOKENS_PER_SEC tok/s)"
echo ""
echo -e "${CYAN}Server Endpoints:${NC}"
echo "  • Main API:  http://localhost:11440"
echo "  • Admin API: http://localhost:11441"
echo ""
echo -e "${CYAN}Management Commands:${NC}"
echo "  • Status:  ./bin/mlx-inference-server-daemon.sh status"
echo "  • Logs:    tail -f logs/mlx-inference-server.log"
echo "  • Stop:    ./bin/mlx-inference-server-daemon.sh stop"
echo "  • Restart: ./bin/mlx-inference-server-daemon.sh restart"
echo ""
echo -e "${CYAN}Next Steps:${NC}"
echo "  1. Install Open WebUI:"
echo "     ${BLUE}pip install open-webui${NC}"
echo ""
echo "  2. Start Open WebUI:"
echo "     ${BLUE}open-webui serve${NC}"
echo ""
echo "  3. Configure connection:"
echo "     Point to: ${BLUE}http://localhost:11440/v1${NC}"
echo ""
echo -e "${GREEN}Ready to use! 🚀${NC}"
echo ""
