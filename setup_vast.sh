#!/bin/bash
# =============================================================================
#  SUPER CONTRA AI — VAST.AI SETUP
# =============================================================================
#  Run this ON the Vast.ai instance, from the extracted project root
#  (e.g. ~/super-contra), after uploading the project tarball.
#
#  Recommended Vast image: a PyTorch/CUDA template (e.g. "pytorch/pytorch"
#  or a "NVIDIA CUDA" image). These already have the NVIDIA driver + CUDA.
#  Vast instances usually run as ROOT, so no 'sudo' is needed (kept optional).
# =============================================================================
set -e

# Vast containers usually run as root; use sudo only if present + not root.
SUDO=""
if [ "$(id -u)" != "0" ] && command -v sudo >/dev/null 2>&1; then SUDO="sudo"; fi

echo "=== SUPER CONTRA AI — VAST.AI SETUP ==="

# 1. Check the GPU is visible
echo "[1/7] Checking GPU..."
nvidia-smi || { echo "ERROR: nvidia-smi failed — pick a CUDA/PyTorch template."; exit 1; }

# 2. System packages (ffmpeg for recordings, zlib for build, venv, git)
echo "[2/7] Installing system packages..."
$SUDO apt-get update -qq
$SUDO apt-get install -y -qq ffmpeg zlib1g-dev python3-venv python3-pip git

# 3. Create the virtual environment
echo "[3/7] Creating virtual environment..."
python3 -m venv ~/contra-venv
source ~/contra-venv/bin/activate
pip install --upgrade pip -q

# 4. Install PyTorch (CUDA build) then the rest
#    NOTE: match the CUDA index to the instance's CUDA. cu128 works for
#    CUDA 12.x images. If the image is CUDA 12.1, use cu121 instead.
echo "[4/7] Installing PyTorch + ML packages..."
pip install torch --index-url https://download.pytorch.org/whl/cu128 -q
pip install -r requirements.txt -q

# 5. Import ROM and copy game config + curriculum states into stable-retro
echo "[5/7] Importing ROM and copying game config + states..."
mkdir -p ~/rom_temp
cp "roms/Super C (USA).nes" ~/rom_temp/
python -m retro.import ~/rom_temp/
GAME_DIR=$(python -c "import stable_retro, os; print(os.path.join(os.path.dirname(stable_retro.__file__), 'data', 'stable', 'SuperC-Nes-v0'))")
cp config/data.json "$GAME_DIR/data.json"
cp config/scenario.json "$GAME_DIR/scenario.json"
# Curriculum save-states (Level1_mid, Level1_boss) so training can start at
# the hard sections. Without these, train.py's AVAILABLE_STATES would fail.
if [ -d states ]; then
  cp states/*.state "$GAME_DIR/" && echo "  Copied curriculum states: $(ls states/)"
fi

# 6. Output dirs
echo "[6/7] Creating output directories..."
mkdir -p checkpoints logs recordings

# 7. Verify
echo "[7/7] Verifying installation..."
python checks/check_system.py

echo ""
echo "=========================================="
echo "  SETUP COMPLETE"
echo "=========================================="
echo ""
echo "  Activate : source ~/contra-venv/bin/activate"
echo "  Workers  : auto-detected from CPU count ($(nproc) here). No export needed."
echo "             To use fewer, run:  export NUM_CPU=<n>"
echo "  Train    : python src/train.py --phase 3"
echo "  Monitor  : nvidia-smi"
echo ""
echo "  Run inside tmux so training survives SSH disconnect:"
echo "    tmux new -s training"
echo "    python src/train.py --phase 3     # uses all $(nproc) CPUs automatically"
echo "    (detach: Ctrl+B then D  |  reattach: tmux attach -t training)"
echo ""
echo "  IMPORTANT: back up models/ to your machine or cloud storage —"
echo "  Vast instances are ephemeral; if the host is reclaimed you lose local data."
echo ""
