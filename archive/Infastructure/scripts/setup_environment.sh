#!/bin/bash
set -e

echo "============================================"
echo "  PlantCLEF 2026 - Environment Setup"
echo "============================================"

# ── 1. System packages ──────────────────────────────
echo "[1/5] Installing system packages..."
apt-get update -qq && apt-get install -y -qq \
    aria2 \
    pigz \
    pv \
    tmux \
    htop \
    unzip \
    p7zip-full \
    > /dev/null 2>&1
echo "  ✓ System packages installed"

# ── 2. Python packages ──────────────────────────────
echo "[2/5] Installing Python packages..."
pip install -q --upgrade pip
pip install -q \
    kaggle \
    timm \
    albumentations \
    scikit-learn \
    pandas \
    matplotlib \
    seaborn \
    tqdm \
    wandb \
    safetensors \
    pillow \
    opencv-python-headless
echo "  ✓ Python packages installed"

# ── 3. Kaggle API setup ─────────────────────────────
echo "[3/5] Setting up Kaggle API..."
KAGGLE_DIR="$HOME/.kaggle"
KAGGLE_JSON="$KAGGLE_DIR/kaggle.json"

if [ -f "$KAGGLE_JSON" ]; then
    echo "  ✓ Kaggle API already configured"
else
    echo ""
    echo "  Kaggle API credentials not found."
    echo "  To set up:"
    echo "    1. Go to https://www.kaggle.com/settings → API → Create New Token"
    echo "    2. Download kaggle.json"
    echo "    3. Run these commands:"
    echo ""
    echo "       mkdir -p ~/.kaggle"
    echo "       echo '{\"username\":\"YOUR_USERNAME\",\"key\":\"YOUR_API_KEY\"}' > ~/.kaggle/kaggle.json"
    echo "       chmod 600 ~/.kaggle/kaggle.json"
    echo ""
    echo "  ⚠ Kaggle API not configured — you'll need this for downloading test data and uploading datasets"
fi

# ── 4. Directory structure ───────────────────────────
echo "[4/5] Creating directory structure..."
BASE_DIR="/workspace/plantclef"
mkdir -p "$BASE_DIR"/{raw/{train,pseudo_quadrats,test,models},processed,submissions}
echo "  Directory structure:"
echo "    $BASE_DIR/"
echo "    ├── raw/"
echo "    │   ├── train/          ← Single-plant training images"
echo "    │   ├── pseudo_quadrats/ ← Unlabeled pseudo-quadrat images"
echo "    │   ├── test/           ← Quadrat test images from Kaggle"
echo "    │   └── models/         ← Pre-trained DINOv2 models"
echo "    ├── processed/          ← Processed/transformed data"
echo "    └── submissions/        ← Submission CSV files"
echo "  ✓ Directories created"

# ── 5. Verify ────────────────────────────────────────
echo "[5/5] Verifying setup..."
python3 -c "
import torch, timm, sklearn, PIL, cv2, albumentations
print(f'  PyTorch:        {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:            {torch.cuda.get_device_name(0)}')
    print(f'  VRAM:           {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
print(f'  timm:           {timm.__version__}')
print(f'  scikit-learn:   {sklearn.__version__}')
print(f'  Pillow:         {PIL.__version__}')
"

echo ""
echo "============================================"
echo "  Setup complete!"
echo "  Next step: bash scripts/download_datasets.sh"
echo "============================================"
