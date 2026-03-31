#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# setup.sh — TripoSR local environment setup (Mac / Linux)
# Run once:  chmod +x setup.sh && ./setup.sh
# Then run:  ./start.sh
# ─────────────────────────────────────────────────────────────────────────────
set -e

VENV=".venv"
TRIPOSR_DIR="TripoSR"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   DEPTH3D · TripoSR Setup                ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# Python check
if ! command -v python3 &>/dev/null; then
    echo "✗ python3 not found. Install Python 3.10+ first."
    exit 1
fi
PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✓ Python $PY_VER"

# Create venv
if [ ! -d "$VENV" ]; then
    echo "→ Creating virtual environment…"
    python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"
echo "✓ Virtual environment active"

# Upgrade pip + setuptools
pip install --upgrade pip setuptools wheel -q

# Detect CUDA
CUDA_VER=""
if command -v nvcc &>/dev/null; then
    CUDA_VER=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
    CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
    echo "✓ CUDA $CUDA_VER detected"
else
    echo "⚠ No CUDA detected — will install CPU-only PyTorch (slow but works)"
fi

# Install PyTorch
if python3 -c "import torch" &>/dev/null 2>&1; then
    echo "✓ PyTorch already installed"
else
    echo "→ Installing PyTorch…"
    if [ -n "$CUDA_MAJOR" ]; then
        if [ "$CUDA_MAJOR" -ge 12 ]; then
            pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q
        elif [ "$CUDA_MAJOR" -eq 11 ]; then
            pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 -q
        else
            echo "⚠ CUDA $CUDA_VER may not be supported. Installing CPU PyTorch."
            pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu -q
        fi
    else
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu -q
    fi
    echo "✓ PyTorch installed"
fi

# Clone TripoSR if needed
if [ ! -d "$TRIPOSR_DIR" ]; then
    echo "→ Cloning TripoSR repository…"
    git clone https://github.com/VAST-AI-Research/TripoSR.git "$TRIPOSR_DIR"
else
    echo "✓ TripoSR repo already cloned"
fi

# Install TripoSR deps
echo "→ Installing TripoSR dependencies…"
pip install -r "$TRIPOSR_DIR/requirements.txt" -q

# Add TripoSR to path so `from tsr.system import TSR` works
if ! grep -q "TripoSR" "$VENV/lib/python$PY_VER/site-packages/triposr.pth" 2>/dev/null; then
    SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
    echo "$(pwd)/$TRIPOSR_DIR" > "$SITE/triposr.pth"
fi
echo "✓ TripoSR on Python path"

# Install remaining deps
echo "→ Installing server deps…"
pip install flask pillow trimesh huggingface_hub -q

# Optional: rembg for clean background removal
echo "→ Installing rembg (background removal)… (optional, may take a moment)"
pip install rembg -q && echo "✓ rembg installed" || echo "⚠ rembg skipped (optional)"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   Setup complete!                        ║"
echo "║                                          ║"
echo "║   Start server:  ./start.sh              ║"
echo "║   Then open:     index.html              ║"
echo "║                                          ║"
echo "║   First run downloads ~2 GB model        ║"
echo "║   weights from HuggingFace.              ║"
echo "╚══════════════════════════════════════════╝"
echo ""
