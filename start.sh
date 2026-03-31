#!/usr/bin/env bash
# start.sh — activate venv and launch TripoSR server
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Add TripoSR to PYTHONPATH so `from tsr.system import TSR` resolves
export PYTHONPATH="$SCRIPT_DIR/TripoSR:$PYTHONPATH"

source ".venv/bin/activate" 2>/dev/null || {
    echo "✗ Virtual environment not found. Run ./setup.sh first."
    exit 1
}

echo ""
echo "  ▸ Starting TripoSR server on http://localhost:7860"
echo "  ▸ Open index.html in your browser"
echo "  ▸ First request downloads model weights (~2 GB) — subsequent runs are instant"
echo "  ▸ Press Ctrl+C to stop"
echo ""

python app.py
