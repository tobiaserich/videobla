#!/bin/bash
set -e

echo "🚀 LongCat-Video Setup Script"
echo "=============================="
echo ""

# Farben für Output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Funktion für farbigen Output
info() { echo -e "${GREEN}✓${NC} $1"; }
warn() { echo -e "${YELLOW}⚠${NC} $1"; }
error() { echo -e "${RED}✗${NC} $1"; }

# 1. System-Check
echo "1️⃣  Checking system requirements..."

# Python Check
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    info "Python found: $PYTHON_VERSION"
else
    error "Python 3 not found! Please install Python 3.9+"
    exit 1
fi

# Git Check
if command -v git &> /dev/null; then
    info "Git found: $(git --version)"
else
    error "Git not found! Please install git"
    exit 1
fi

# GPU Check
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n1)
    info "GPU detected: $GPU_NAME ($GPU_MEMORY)"
    HAS_GPU=true
else
    warn "No NVIDIA GPU detected - will run in CPU mode (very slow!)"
    HAS_GPU=false
fi

echo ""
echo "2️⃣  Setting up Python environment..."

# Virtual Environment erstellen
if [ ! -d "venv" ]; then
    info "Creating virtual environment..."
    python3 -m venv venv
else
    info "Virtual environment already exists"
fi

# Aktivieren
source venv/bin/activate
info "Virtual environment activated"

echo ""
echo "3️⃣  Installing dependencies..."

# Upgrade pip
pip install --upgrade pip -q

# PyTorch installieren (GPU oder CPU)
if [ "$HAS_GPU" = true ]; then
    info "Installing PyTorch 2.5.1 with CUDA support..."
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121 -q
else
    warn "Installing PyTorch 2.5.1 CPU-only version..."
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 -q
fi

# Basis-Dependencies
info "Installing base dependencies..."
cd serverless
pip install -r requirements.txt -q
cd ..

echo ""
echo "4️⃣  Setting up LongCat-Video..."

# LongCat-Video Repository
if [ ! -d "LongCat-Video" ]; then
    info "Cloning LongCat-Video repository..."
    git clone --depth 1 https://github.com/meituan-longcat/LongCat-Video.git
else
    info "LongCat-Video repository already exists"
fi

# LongCat Dependencies (ohne flash-attn wenn es Probleme gibt)
cd LongCat-Video
info "Installing LongCat-Video core dependencies..."
pip install -q loguru ftfy regex hf-transfer  # Wichtige Dependencies

if pip install -r requirements.txt 2>/dev/null; then
    info "LongCat-Video dependencies installed"
else
    warn "Some dependencies failed, installing without flash-attn..."
    grep -v "flash-attn" requirements.txt > requirements_safe.txt
    pip install -r requirements_safe.txt -q
fi
cd ..

echo ""
echo "5️⃣  Environment configuration..."

# .env erstellen falls nicht vorhanden
if [ ! -f ".env" ]; then
    info "Creating .env file from template..."
    cp .env.example .env
    warn "Please edit .env and add your RUNPOD_API_KEY!"
else
    info ".env file already exists"
fi

# Python Path für LongCat-Video
export PYTHONPATH="${PYTHONPATH}:$(pwd)/LongCat-Video"
info "PYTHONPATH configured"

echo ""
echo "6️⃣  Testing installation..."

# Quick test
python3 -c "
import torch
print(f'✓ PyTorch {torch.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ CUDA device: {torch.cuda.get_device_name(0)}')
" && info "PyTorch test passed"

echo ""
echo "=============================="
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Activate environment: source venv/bin/activate"
echo "  2. Configure .env file with your API keys"
echo "  3. Run client: python local/client.py --prompt 'test'"
echo "  4. Or run local server: python standalone_server.py"
echo ""

if [ "$HAS_GPU" = false ]; then
    warn "No GPU detected! Video generation will be VERY slow."
    echo "   Consider using Runpod serverless instead (already configured)"
fi

echo ""
