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

# Detect if we should use system PyTorch (for servers with pre-installed torch)
USE_SYSTEM_TORCH=false
if python3 -c "import torch; print(torch.__version__)" 2>/dev/null | grep -q "2.8"; then
    info "Detected system PyTorch 2.8+ - will use system-site-packages"
    USE_SYSTEM_TORCH=true
fi

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

# HuggingFace Cache-Verzeichnis konfigurieren (wichtig für /workspace!)
if [ -d "/workspace" ] && [ ! -d "/workspace/.cache" ]; then
    info "Setting up HuggingFace cache in /workspace..."
    mkdir -p /workspace/.cache/huggingface
    export HF_HOME=/workspace/.cache/huggingface
    info "HF_HOME set to /workspace/.cache/huggingface"
elif [ -d "/workspace/.cache/huggingface" ]; then
    export HF_HOME=/workspace/.cache/huggingface
    info "Using existing HF cache at /workspace/.cache/huggingface"
fi

# Virtual Environment erstellen
if [ ! -d "venv" ]; then
    info "Creating virtual environment..."
    if [ "$USE_SYSTEM_TORCH" = true ]; then
        python3 -m venv venv --system-site-packages
        info "Created venv with --system-site-packages (using system PyTorch)"
    else
        python3 -m venv venv
        info "Created isolated venv"
    fi
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

# PyTorch installieren (nur wenn nicht system-site-packages)
if [ "$USE_SYSTEM_TORCH" = true ]; then
    info "Using system PyTorch (already installed)"
    python3 -c "import torch; print(f'  PyTorch {torch.__version__}')"
elif [ "$HAS_GPU" = true ]; then
    info "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
else
    warn "Installing PyTorch CPU-only version..."
    pip install torch torchvision torchaudio -q
fi

# Basis-Dependencies
info "Installing base dependencies..."
cd serverless
pip install -r requirements.txt -q
cd ..

# Wichtige zusätzliche Packages (die oft fehlen)
info "Installing additional required packages..."
pip install -q runpod loguru ftfy regex opencv-python-headless hf-transfer transformers safetensors

echo ""
echo "4️⃣  Setting up LongCat-Video..."

# LongCat-Video Repository
if [ ! -d "LongCat-Video" ]; then
    info "Cloning LongCat-Video repository..."
    git clone --depth 1 https://github.com/meituan-longcat/LongCat-Video.git
else
    info "LongCat-Video repository already exists"
fi

# LongCat Dependencies
cd LongCat-Video
info "Installing LongCat-Video core dependencies..."

# Installiere Requirements ohne flash-attn (das kommt später)
if grep -q "flash-attn" requirements.txt; then
    grep -v "flash-attn" requirements.txt > requirements_safe.txt
    pip install -r requirements_safe.txt -q 2>&1 | grep -v "already satisfied" || true
    rm requirements_safe.txt
else
    pip install -r requirements.txt -q 2>&1 | grep -v "already satisfied" || true
fi

# Flash-attention separat installieren (braucht CUDA und kann lange dauern)
if [ "$HAS_GPU" = true ]; then
    info "Installing flash-attn (this may take 3-5 minutes to compile)..."
    if pip install flash-attn --no-build-isolation 2>&1 | tee /tmp/flash_install.log | tail -1 | grep -q "Successfully installed"; then
        info "flash-attn installed successfully"
    else
        if grep -q "Successfully installed flash-attn" /tmp/flash_install.log; then
            info "flash-attn installed successfully"
        else
            warn "flash-attn installation failed - will use slower attention fallback"
        fi
    fi
    rm -f /tmp/flash_install.log
else
    warn "Skipping flash-attn (no GPU detected)"
fi
cd ..

echo ""
echo "5️⃣  Environment configuration..."

# .env erstellen falls nicht vorhanden
if [ ! -f ".env" ]; then
    info "Creating .env file from template..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
    else
        # Fallback: Minimale .env erstellen
        cat > .env << EOF
# RunPod Configuration
RUNPOD_API_KEY=your_api_key_here

# Model Settings
HF_HUB_ENABLE_HF_TRANSFER=1
HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}

# Server Settings
PORT=8000
HOST=0.0.0.0
EOF
    fi
    warn "Please edit .env and add your RUNPOD_API_KEY!"
else
    info ".env file already exists"
fi

# Environment Variablen in .bashrc oder .profile setzen (für persistente Session)
if [ -d "/workspace" ]; then
    if ! grep -q "HF_HOME" ~/.bashrc 2>/dev/null; then
        echo "" >> ~/.bashrc
        echo "# HuggingFace Cache Configuration" >> ~/.bashrc
        echo "export HF_HOME=/workspace/.cache/huggingface" >> ~/.bashrc
        echo "export HF_HUB_ENABLE_HF_TRANSFER=1" >> ~/.bashrc
        info "Added HF environment variables to ~/.bashrc"
    fi
fi

# Python Path für LongCat-Video
export PYTHONPATH="${PYTHONPATH}:$(pwd)/LongCat-Video"
if ! grep -q "LongCat-Video" ~/.bashrc 2>/dev/null; then
    echo "export PYTHONPATH=\"\${PYTHONPATH}:$(pwd)/LongCat-Video\"" >> ~/.bashrc
    info "Added PYTHONPATH to ~/.bashrc"
fi
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
    print(f'✓ CUDA version: {torch.version.cuda}')
" && info "PyTorch test passed"

# Test kritische Imports
python3 -c "
import sys
sys.path.insert(0, 'LongCat-Video')
try:
    from longcat_video.pipeline_longcat_video import LongCatVideoPipeline
    print('✓ LongCat-Video imports working')
except Exception as e:
    print(f'⚠ LongCat-Video import warning: {e}')

try:
    import flash_attn
    print('✓ flash-attn available')
except:
    print('⚠ flash-attn not available (slower fallback will be used)')
" 2>&1

echo ""
echo "=============================="
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Activate environment: source venv/bin/activate"
if [ ! -f ".env" ] || grep -q "your_api_key_here" .env 2>/dev/null; then
    echo "  2. Configure .env file with your RUNPOD_API_KEY"
fi
echo "  3. Start server: ./start_server.sh"
echo "     OR"
echo "     python standalone_server.py"
echo ""
echo "Test endpoint after starting server:"
echo "  curl -X POST http://localhost:8000/generate \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"prompt\":\"a cat\",\"height\":384,\"width\":384,\"num_frames\":4,\"num_inference_steps\":10,\"guidance_scale\":3.0}'"
echo ""

if [ "$HAS_GPU" = false ]; then
    warn "No GPU detected! Video generation will be VERY slow."
    echo "   Consider using Runpod serverless instead"
fi

if [ -d "/workspace" ]; then
    info "Running in /workspace - HuggingFace cache configured for large storage"
fi

echo ""
echo "  4. Or run local server: python standalone_server.py"
echo ""

if [ "$HAS_GPU" = false ]; then
    warn "No GPU detected! Video generation will be VERY slow."
    echo "   Consider using Runpod serverless instead (already configured)"
fi

echo ""
