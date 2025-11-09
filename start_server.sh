#!/bin/bash
# Start LongCat-Video Server with correct cache directory

# Aktiviere venv
source venv/bin/activate

# Setze Cache auf /workspace (hat mehr Platz als /root)
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/.cache/huggingface

# Deaktiviere hf_transfer
unset HF_HUB_ENABLE_HF_TRANSFER

echo "🚀 Starting LongCat-Video Server..."
echo "📦 Cache directory: $HF_HOME"
echo ""

python standalone_server.py
