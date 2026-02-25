#!/bin/bash
# Setup MedSAM3 integration - Complete automated setup

set -e

echo ""
echo "=========================================="
echo "MedSAM3 Setup Script"
echo "=========================================="
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Check Python environment
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 not found"
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"

# Check if in virtual environment (recommended)
if [[ -z "${VIRTUAL_ENV}" ]]; then
    echo "⚠️  Warning: Not in a virtual environment"
    echo "   Recommended: source .venv/bin/activate"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install -q huggingface_hub torch 2>/dev/null || {
    echo "⚠️  Some dependencies may already be installed"
}
echo "✓ Dependencies ready"

# Create directories
echo ""
echo "📁 Creating directories..."
mkdir -p weights/medsam3
mkdir -p scripts
echo "✓ Directories created"

# Download weights
echo ""
echo "⬇️  Downloading MedSAM3 weights..."
echo "   This may take a few minutes..."
python3 scripts/download_medsam3_weights.py || {
    echo "❌ Download failed!"
    echo "   Check your internet connection and HuggingFace authentication"
    echo "   Run: huggingface-cli login"
    exit 1
}

# Convert weights
echo ""
echo "🔄 Converting weights to MLX format..."
python3 scripts/convert_medsam3_to_mlx.py || {
    echo "❌ Conversion failed!"
    exit 1
}

# Verify installation
echo ""
echo "✅ Verifying installation..."
python3 scripts/verify_mlx_weights.py || {
    echo "❌ Verification failed!"
    exit 1
}

echo ""
echo "=========================================="
echo "✓ MedSAM3 setup complete!"
echo "=========================================="
echo ""
echo "📋 Next steps:"
echo "   1. Start the backend:"
echo "      cd app && ./run.sh"
echo ""
echo "   2. Check logs for MedSAM3 confirmation:"
echo "      Look for '✓ MedSAM3 weights loaded successfully!'"
echo ""
echo "   3. Test with medical images through the API"
echo ""
echo "📚 Documentation:"
echo "   - MEDSAM3_INTEGRATION_PLAN.md"
echo "   - MEDSAM3_INVESTIGATION_REPORT.md"
echo ""
