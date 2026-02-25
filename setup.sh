#!/bin/bash

# MLX MedSAM3 Setup Script
# Automates installation and weight download

set -e  # Exit on error

echo "╔════════════════════════════════════════╗"
echo "║   MLX MedSAM3 Setup                    ║"
echo "╚════════════════════════════════════════╝"
echo ""

# Check prerequisites
echo "🔍 Checking prerequisites..."

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.13+"
    exit 1
fi
echo "✓ Python found: $(python3 --version)"

# Check uv
if ! command -v uv &> /dev/null; then
    echo "❌ uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi
echo "✓ uv found: $(uv --version)"

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "⚠️  Node.js not found. Frontend will not work."
    echo "   Install from: https://nodejs.org/"
else
    echo "✓ Node.js found: $(node --version)"
fi

echo ""
echo "📦 Installing Python dependencies..."
uv sync

echo ""
echo "⬇️  Downloading MedSAM3 weights..."
if [ -f "weights/medsam3_lora.safetensors" ]; then
    echo "✓ Weights already exist, skipping download"
else
    uv run python scripts/download_medsam3_weights.py
    echo ""
    echo "🔄 Converting weights to MLX format..."
    uv run python scripts/convert_medsam3_to_mlx.py
fi

echo ""
echo "✅ Verifying installation..."
uv run python scripts/verify_mlx_weights.py

echo ""
echo "📦 Installing frontend dependencies..."
if [ -d "app/frontend" ]; then
    cd app/frontend
    if [ -f "package.json" ]; then
        npm install
        cd ../..
        echo "✓ Frontend dependencies installed"
    else
        cd ../..
        echo "⚠️  No package.json found in app/frontend"
    fi
else
    echo "⚠️  Frontend directory not found"
fi

echo ""
echo "╔════════════════════════════════════════╗"
echo "║   Setup Complete! 🎉                   ║"
echo "╚════════════════════════════════════════╝"
echo ""
echo "To start the application:"
echo "  cd app"
echo "  ./run.sh"
echo ""
echo "Then open: http://localhost:3000"
echo ""
