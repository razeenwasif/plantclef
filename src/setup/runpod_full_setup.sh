#!/bin/bash
set -e

echo "========================================"
echo " PlantCLEF 2026 - Full RunPod Setup"
echo "========================================"

# ── 1. Directories ──
echo "[1/6] Creating directory structure..."
mkdir -p /workspace/plantclef/raw/{train,pseudo_quadrats,test,models}
mkdir -p /workspace/plantclef/{processed,submissions}
echo "  Done."

# ── 2. System packages ──
echo "[2/6] Installing system packages..."
apt-get update -qq
apt-get install -y -qq aria2 pigz pv tmux htop unzip p7zip-full > /dev/null 2>&1
echo "  Done."

# ── 3. Python packages ──
echo "[3/6] Installing Python packages..."
pip install -q --upgrade pip
pip install -q kaggle timm albumentations scikit-learn pandas matplotlib seaborn tqdm wandb safetensors pillow opencv-python-headless
echo "  Done."

# ── 4. Verify environment ──
echo "[4/6] Verifying environment..."
python3 << 'PYEOF'
import torch, timm, sklearn, PIL
print("  PyTorch:", torch.__version__)
print("  CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("  GPU:", torch.cuda.get_device_name(0))
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print("  VRAM: %.1f GB" % vram)
print("  timm:", timm.__version__)
print("  sklearn:", sklearn.__version__)
PYEOF
echo "  Done."

# ── 5. Download pre-trained models from Zenodo ──
echo "[5/6] Downloading pre-trained models from Zenodo..."
cd /workspace/plantclef/raw/models

python3 << 'PYEOF'
import json, urllib.request, os

record = json.loads(urllib.request.urlopen("https://zenodo.org/api/records/10848263").read())
for f in record.get("files", []):
    name = f["key"]
    url = f["links"]["self"]
    size_mb = f["size"] / 1e6
    path = os.path.join("/workspace/plantclef/raw/models", name)
    if os.path.exists(path):
        print("  SKIP %s (already exists)" % name)
    else:
        print("  Downloading %s (%.0f MB)..." % (name, size_mb))
        urllib.request.urlretrieve(url, path)
        print("  Done: %s" % name)
PYEOF
echo "  Models done."

# ── 6. Download test set from Seafile (alternative link, not Kaggle) ──
echo "[6/6] Downloading test set from Seafile..."
cd /workspace/plantclef/raw/test

python3 << 'PYEOF'
import json, urllib.request, os, sys

base_url = "https://lab.plantnet.org/seafile"
token = "6a91e3de6b5b49e5a70c"
api_url = "%s/api/v2.1/share-links/%s/dirents/?path=/" % (base_url, token)

try:
    resp = urllib.request.urlopen(api_url, timeout=30)
    data = json.loads(resp.read())
    files = data.get("dirent_list", [])
    print("  Found %d items" % len(files))

    for f in files:
        if "file_name" in f:
            name = f["file_name"]
            size = f.get("file_size", 0)
            out = os.path.join("/workspace/plantclef/raw/test", name)
            if os.path.exists(out) and os.path.getsize(out) > 0:
                print("  SKIP %s" % name)
                continue
            dl_url = "%s/d/%s/files/?p=/%s&dl=1" % (base_url, token, urllib.request.quote(name))
            print("  Downloading %s (%.1f MB)..." % (name, size / 1e6))
            urllib.request.urlretrieve(dl_url, out)
            print("  Done: %s" % name)
        elif "folder_name" in f:
            print("  Found folder: %s (will need separate handling)" % f["folder_name"])
except Exception as e:
    print("  Seafile API error: %s" % str(e))
    print("  Falling back: you can download test data from Kaggle instead:")
    print("    kaggle competitions download -c plantclef-2026 -p /workspace/plantclef/raw/test/")
PYEOF
echo "  Test set done."

echo ""
echo "========================================"
echo " Setup complete!"
echo "========================================"
echo ""
echo " Directory structure:"
ls -la /workspace/plantclef/raw/
echo ""
echo " Model files:"
ls -lh /workspace/plantclef/raw/models/ 2>/dev/null || echo "  (none yet)"
echo ""
echo " Test files:"
ls -lh /workspace/plantclef/raw/test/ 2>/dev/null || echo "  (none yet)"
echo ""
echo " Disk usage:"
du -sh /workspace/plantclef/raw/*
echo ""
echo "========================================"
echo " NEXT STEPS - Training Data Download"
echo "========================================"
echo ""
echo " The training set (~160GB) must be downloaded from Seafile."
echo " Run inside tmux so it survives disconnects:"
echo ""
echo "   tmux new -s download"
echo "   python3 /workspace/plantclef/seafile_downloader.py \\"
echo "     --url https://lab.plantnet.org/seafile/d/303fec50b1a544c6a2ed \\"
echo "     --output /workspace/plantclef/raw/train \\"
echo "     --workers 4"
echo ""
echo " For pseudo-quadrats (~170GB):"
echo "   python3 /workspace/plantclef/seafile_downloader.py \\"
echo "     --url https://lab.plantnet.org/seafile/d/f3a63defc5f44220b194 \\"
echo "     --output /workspace/plantclef/raw/pseudo_quadrats \\"
echo "     --workers 4"
echo ""
