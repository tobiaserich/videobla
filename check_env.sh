#!/bin/bash
# Environment Diagnostic Tool for LongCat-Video

# Farben
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "🔍 LongCat-Video Environment Diagnostic"
echo "========================================"
echo ""

# Aktiviere venv falls vorhanden
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo -e "${GREEN}✓${NC} Virtual environment activated"
else
    echo -e "${YELLOW}⚠${NC} No venv found"
fi

echo ""
echo "1️⃣  Python Environment"
echo "--------------------"
python3 --version
which python3
echo ""

echo "2️⃣  GPU Check"
echo "--------------------"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
    python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'Current device: {torch.cuda.get_device_name(0)}')
"
else
    echo -e "${RED}✗${NC} No NVIDIA GPU detected"
fi

echo ""
echo "3️⃣  PyTorch Versions"
echo "--------------------"
python3 -c "
import torch
import torchvision
import torchaudio

print(f'torch:       {torch.__version__}')
print(f'torchvision: {torchvision.__version__}')
print(f'torchaudio:  {torchaudio.__version__}')

# Compatibility check
torch_ver = torch.__version__.split('+')[0]
tv_ver = torchvision.__version__.split('+')[0]
torch_minor = '.'.join(torch_ver.split('.')[:2])
tv_minor = '.'.join(tv_ver.split('.')[:2])

compat_map = {
    '2.8': '0.23',  # PyTorch 2.8 + CUDA 12.8
    '2.7': '0.22',
    '2.6': '0.21',
    '2.5': '0.20',
    '2.4': '0.19',
    '2.3': '0.18',
    '2.2': '0.17',
}

expected_tv = compat_map.get(torch_minor, tv_minor)
if tv_minor == expected_tv:
    print(f'\n✓ Versions compatible')
else:
    print(f'\n⚠ Version mismatch! torch {torch_minor} expects torchvision {expected_tv}, got {tv_minor}')
    print('  Run: pip install --upgrade torch torchvision torchaudio')
"

echo ""
echo "4️⃣  Critical Packages"
echo "--------------------"
python3 -c "
packages = [
    'transformers',
    'diffusers',
    'accelerate',
    'safetensors',
    'runpod',
    'fastapi',
    'uvicorn',
]

import importlib
import sys

for pkg in packages:
    try:
        mod = importlib.import_module(pkg)
        ver = getattr(mod, '__version__', 'unknown')
        print(f'✓ {pkg:15s} {ver}')
    except ImportError:
        print(f'✗ {pkg:15s} NOT INSTALLED')
"

echo ""
echo "5️⃣  LongCat-Video Import Test"
echo "--------------------"
export PYTHONPATH="${PYTHONPATH}:$(pwd)/LongCat-Video"
python3 -c "
import sys
import os

# Check if LongCat-Video exists
if os.path.exists('LongCat-Video'):
    sys.path.insert(0, 'LongCat-Video')
    print('✓ LongCat-Video directory found')
    
    try:
        from longcat_video.pipeline_longcat_video import LongCatVideoPipeline
        print('✓ LongCat-Video imports working')
    except Exception as e:
        print(f'✗ Import failed: {e}')
        import traceback
        traceback.print_exc()
else:
    print('✗ LongCat-Video directory not found')
    print('  Run: git clone https://github.com/meituan-longcat/LongCat-Video.git')
" 2>&1

echo ""
echo "6️⃣  Flash Attention"
echo "--------------------"
python3 -c "
try:
    import flash_attn
    print(f'✓ flash-attn installed (version: {flash_attn.__version__})')
except ImportError as e:
    print('⚠ flash-attn not available (slower fallback will be used)')
    print(f'  Error: {e}')
except Exception as e:
    print(f'✗ flash-attn error: {e}')
" 2>&1

echo ""
echo "7️⃣  Cache Configuration"
echo "--------------------"
echo "HF_HOME: ${HF_HOME:-not set}"
if [ -d "/workspace/.cache/huggingface" ]; then
    du -sh /workspace/.cache/huggingface 2>/dev/null || echo "Directory exists but cannot check size"
fi
if [ -d "$HOME/.cache/huggingface" ]; then
    du -sh $HOME/.cache/huggingface 2>/dev/null || echo "Directory exists but cannot check size"
fi

echo ""
echo "8️⃣  Disk Space"
echo "--------------------"
df -h / | tail -1
if [ -d "/workspace" ]; then
    df -h /workspace | tail -1
fi

echo ""
echo "========================================"
echo "Diagnostic complete!"
echo ""
echo "To fix issues:"
echo "  • Version mismatch: pip install --upgrade torch torchvision torchaudio"
echo "  • Missing packages: pip install -r serverless/requirements.txt"
echo "  • LongCat-Video: git clone https://github.com/meituan-longcat/LongCat-Video.git"
echo "  • Flash-attn: pip install flash-attn --no-build-isolation"
echo ""
