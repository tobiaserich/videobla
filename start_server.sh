#!/bin/bash
# Start LongCat-Video Server with optimized settings

set -e

# Farben
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}🚀 Starting LongCat-Video Server${NC}"
echo "================================"

# Check if running in /workspace or local
if [ -d "/workspace" ]; then
    export HF_HOME=/workspace/.cache/huggingface
    export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
    export HF_DATASETS_CACHE=/workspace/.cache/huggingface
    echo "📦 Cache: /workspace/.cache/huggingface"
else
    mkdir -p ~/.cache/huggingface
    export HF_HOME=~/.cache/huggingface
    echo "📦 Cache: ~/.cache/huggingface"
fi

# Enable fast downloads
export HF_HUB_ENABLE_HF_TRANSFER=1

# Add LongCat-Video to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/LongCat-Video"

# Aktiviere venv falls vorhanden
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✓ Virtual environment activated"
else
    echo -e "${YELLOW}⚠ No venv found, using system Python${NC}"
fi

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader | head -n1)
    echo "🎮 GPU: $GPU_NAME (Free: $GPU_MEM)"
else
    echo -e "${YELLOW}⚠ No GPU detected${NC}"
fi

echo "🌐 Starting server..."
echo ""

# Start mit nohup wenn im background gewünscht
if [ "$1" = "--background" ] || [ "$1" = "-b" ]; then
    nohup python standalone_server.py > /tmp/server.log 2>&1 &
    PID=$!
    echo "✓ Server started in background (PID: $PID)"
    echo "  Logs: tail -f /tmp/server.log"
    echo "  Stop: kill $PID"
else
    python standalone_server.py
fi
