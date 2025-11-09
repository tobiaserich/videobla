#!/bin/bash
# Quick update script - pull latest code and restart server

set -e

echo "🔄 Updating LongCat-Video Server..."

# Git pull
echo "📥 Pulling latest code..."
git pull origin main

# Activate venv
source venv/bin/activate

# Reinstall dependencies (nur wenn sich was geändert hat)
echo "📦 Checking dependencies..."
pip install -q --upgrade -r serverless/requirements.txt
pip install -q loguru ftfy regex

echo "✅ Update complete!"
echo ""
echo "To restart server:"
echo "  python standalone_server.py"
