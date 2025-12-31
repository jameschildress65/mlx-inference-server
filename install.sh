#!/bin/bash
#
# MLX Inference Server - Automated Installation Script
# Handles both text and vision venvs, verifies everything, starts server
#

set -e  # Exit on any error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}MLX Inference Server - Automated Installation${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Check prerequisites
echo -e "${YELLOW}[1/8]${NC} Checking prerequisites..."

# Check macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo -e "${RED}ERROR: This script requires macOS${NC}"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "  ✓ Python: $PYTHON_VERSION"

# Check Apple Silicon
CHIP=$(system_profiler SPHardwareDataType | grep Chip | awk '{print $2, $3}')
if [[ ! "$CHIP" =~ "Apple" ]]; then
    echo -e "${RED}ERROR: Apple Silicon required (M1/M2/M3/M4)${NC}"
    exit 1
fi
echo "  ✓ Chip: $CHIP"

# Check RAM
RAM=$(system_profiler SPHardwareDataType | grep "Memory:" | awk '{print $2, $3}')
echo "  ✓ RAM: $RAM"

echo ""

# Main venv setup
echo -e "${YELLOW}[2/8]${NC} Setting up main virtual environment (text models)..."

if [ -d "venv" ]; then
    echo "  ⚠ venv exists, removing old one..."
    rm -rf venv
fi

python3 -m venv venv
source venv/bin/activate

echo "  → Installing dependencies (this may take 5-10 minutes)..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo "  → Verifying main venv..."
python -c "import mlx.core as mx; print(f'  ✓ MLX: {mx.__version__}')"
python -c "import fastapi; print('  ✓ FastAPI: OK')"
python -c "import posix_ipc; print('  ✓ posix_ipc: OK')"
python -c "from PIL import Image; print('  ✓ Pillow: OK')"
python -c "import fitz; print('  ✓ PyMuPDF: OK')"

deactivate
echo -e "${GREEN}  ✓ Main venv ready${NC}"
echo ""

# Vision venv setup
echo -e "${YELLOW}[3/8]${NC} Setting up vision virtual environment..."

if [ -d "venv-vision" ]; then
    echo "  ⚠ venv-vision exists, removing old one..."
    rm -rf venv-vision
fi

python3 -m venv venv-vision
source venv-vision/bin/activate

echo "  → Installing vision dependencies (this may take 10-15 minutes)..."
echo "  → (PyTorch is large, please be patient)"
pip install --upgrade pip -q
pip install -r requirements-vision.txt -q

echo "  → Verifying vision venv..."
python -c "import mlx_vlm; print('  ✓ mlx-vlm: OK')"
python -c "from PIL import Image; print('  ✓ Pillow: OK')"
python -c "import torch; print('  ✓ PyTorch: OK')"
python -c "import torchvision; print('  ✓ Torchvision: OK')"

deactivate
echo -e "${GREEN}  ✓ Vision venv ready${NC}"
echo ""

# Check for old server
echo -e "${YELLOW}[4/8]${NC} Checking for existing server..."
if pgrep -f mlx-inference-server > /dev/null; then
    echo "  ⚠ Server already running, stopping it..."
    pkill -f mlx-inference-server || true
    sleep 2
fi
echo "  ✓ No conflicts"
echo ""

# Start server
echo -e "${YELLOW}[5/8]${NC} Starting MLX Inference Server..."
./bin/mlx-inference-server-daemon.sh start

echo "  → Waiting for server to initialize..."
sleep 5

# Check health
echo -e "${YELLOW}[6/8]${NC} Verifying server health..."
HEALTH=$(curl -s http://localhost:11440/health)
if [[ "$HEALTH" == *"healthy"* ]]; then
    echo -e "  ${GREEN}✓ Main API: http://localhost:11440${NC}"
else
    echo -e "  ${RED}✗ Main API not responding${NC}"
    exit 1
fi

ADMIN=$(curl -s http://localhost:11441/admin/health)
if [[ "$ADMIN" == *"degraded"* ]] || [[ "$ADMIN" == *"healthy"* ]]; then
    echo -e "  ${GREEN}✓ Admin API: http://localhost:11441${NC}"
else
    echo -e "  ${RED}✗ Admin API not responding${NC}"
    exit 1
fi

echo ""

# Test text inference
echo -e "${YELLOW}[7/8]${NC} Testing text inference..."
RESPONSE=$(curl -s -X POST http://localhost:11440/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Say hello in 3 words"}],
    "max_tokens": 10
  }')

if [[ "$RESPONSE" == *"assistant"* ]]; then
    TOKENS_PER_SEC=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['usage'].get('tokens_per_sec', 'N/A'))")
    echo -e "  ${GREEN}✓ Text inference working (${TOKENS_PER_SEC} tok/s)${NC}"
else
    echo -e "  ${RED}✗ Text inference failed${NC}"
    echo "$RESPONSE"
    exit 1
fi

echo ""

# Summary
echo -e "${YELLOW}[8/8]${NC} Installation Summary"
echo ""
echo -e "${GREEN}✓ Main venv installed${NC} (venv/)"
echo -e "${GREEN}✓ Vision venv installed${NC} (venv-vision/)"
echo -e "${GREEN}✓ Server running${NC} (PID: $(pgrep -f mlx-inference-server))"
echo -e "${GREEN}✓ Text inference tested${NC}"
echo ""
echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}Installation Complete!${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""
echo "Server is running at:"
echo "  • Main API:  http://localhost:11440"
echo "  • Admin API: http://localhost:11441"
echo ""
echo "Useful commands:"
echo "  • Status:  ./bin/mlx-inference-server-daemon.sh status"
echo "  • Logs:    tail -f logs/mlx-inference-server.log"
echo "  • Stop:    ./bin/mlx-inference-server-daemon.sh stop"
echo "  • Restart: ./bin/mlx-inference-server-daemon.sh restart"
echo ""
echo "Next steps:"
echo "  1. Install Open WebUI: pip install open-webui"
echo "  2. Start it: open-webui serve"
echo "  3. Configure it to use: http://localhost:11440/v1"
echo ""
echo -e "${GREEN}Ready for use!${NC}"
echo ""
