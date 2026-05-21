#!/bin/bash
set -e

# ══════════════════════════════════════════════════════
#  PlantCLEF 2026 - Dataset Download Script
#
#  Run inside tmux/screen to survive SSH disconnects:
#    tmux new -s download
#    bash scripts/download_datasets.sh
# ══════════════════════════════════════════════════════

BASE_DIR="/workspace/plantclef/raw"
COMPETITION="plantclef-2026"

# Change this if you want both resolutions of training data
TRAIN_RESOLUTION="max"  # "max" = max-side-800px (~160GB), "min" = min-side-800px (~281GB)

echo "============================================"
echo "  PlantCLEF 2026 - Dataset Downloader"
echo "============================================"
echo "  Base directory: $BASE_DIR"
echo "  Training resolution: $TRAIN_RESOLUTION"
echo ""

# ── Helper: check disk space ─────────────────────────
check_disk_space() {
    local available_gb
    available_gb=$(df --output=avail -BG "$BASE_DIR" | tail -1 | tr -d ' G')
    echo "  Available disk space: ${available_gb} GB"
    if [ "$available_gb" -lt 50 ]; then
        echo "  ⚠ WARNING: Less than 50GB available. Downloads may fail."
    fi
}

# ══════════════════════════════════════════════════════
# STEP 1: Kaggle competition data (test set + metadata)
# ══════════════════════════════════════════════════════
download_kaggle_data() {
    echo ""
    echo "── [1/4] Kaggle Competition Data (~8.3 GB) ──"
    check_disk_space

    if [ -d "$BASE_DIR/test/PlantCLEF2025test" ]; then
        echo "  ✓ Test data already exists, skipping."
        return
    fi

    if ! command -v kaggle &> /dev/null; then
        echo "  ✗ Kaggle CLI not found. Run: pip install kaggle"
        return 1
    fi

    if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
        echo "  ✗ Kaggle credentials not found. See setup_environment.sh"
        return 1
    fi

    echo "  Downloading competition data..."
    kaggle competitions download -c "$COMPETITION" -p "$BASE_DIR/test/"
    echo "  Extracting..."
    cd "$BASE_DIR/test/"
    unzip -q -o "*.zip" 2>/dev/null || true
    rm -f *.zip
    echo "  ✓ Kaggle data downloaded and extracted"
}

# ══════════════════════════════════════════════════════
# STEP 2: Training set (single-plant images from Seafile)
# ══════════════════════════════════════════════════════
download_training_data() {
    echo ""
    echo "── [2/4] Training Set (single-plant images) ──"
    check_disk_space

    local TRAIN_DIR="$BASE_DIR/train"

    if [ "$TRAIN_RESOLUTION" = "max" ]; then
        SEAFILE_URL="https://lab.plantnet.org/seafile/d/303fec50b1a544c6a2ed"
        EXPECTED_SIZE="~160 GB"
    else
        SEAFILE_URL="https://lab.plantnet.org/seafile/d/01ab657c21e44a0c8a3e"
        EXPECTED_SIZE="~281 GB"
    fi

    echo "  Resolution: ${TRAIN_RESOLUTION}-side-800px ($EXPECTED_SIZE)"
    echo "  Source: $SEAFILE_URL"
    echo ""
    echo "  ╔══════════════════════════════════════════════════╗"
    echo "  ║  SEAFILE MANUAL DOWNLOAD REQUIRED                ║"
    echo "  ║                                                   ║"
    echo "  ║  Seafile shared links require browser interaction ║"
    echo "  ║  or the Seafile API for bulk download.            ║"
    echo "  ║                                                   ║"
    echo "  ║  OPTION A: Use the Seafile download token API     ║"
    echo "  ║  OPTION B: Download from browser & upload via SCP ║"
    echo "  ║  OPTION C: Use the provided helper script below   ║"
    echo "  ╚══════════════════════════════════════════════════╝"
    echo ""

    # Attempt to get download links via Seafile API
    echo "  Attempting Seafile API download..."

    local SHARE_TOKEN
    if [ "$TRAIN_RESOLUTION" = "max" ]; then
        SHARE_TOKEN="303fec50b1a544c6a2ed"
    else
        SHARE_TOKEN="01ab657c21e44a0c8a3e"
    fi

    echo "  Fetching file list from Seafile..."
    local FILE_LIST
    FILE_LIST=$(curl -s "https://lab.plantnet.org/seafile/api/v2.1/share-links/${SHARE_TOKEN}/dirents/?path=/" 2>/dev/null || echo "FAILED")

    if echo "$FILE_LIST" | python3 -c "import sys, json; json.load(sys.stdin)" 2>/dev/null; then
        echo "  ✓ Seafile API accessible. Listing files..."
        echo "$FILE_LIST" | python3 -c "
import sys, json
data = json.load(sys.stdin)
files = data.get('dirent_list', [])
for f in files:
    name = f.get('file_name', f.get('folder_name', 'unknown'))
    size = f.get('file_size', 0)
    print(f'    {name}  ({size / 1e9:.1f} GB)' if size > 0 else f'    {name}/')
"
        echo ""
        echo "  Downloading files..."
        echo "$FILE_LIST" | python3 -c "
import sys, json
data = json.load(sys.stdin)
files = data.get('dirent_list', [])
for f in files:
    if 'file_name' in f:
        name = f['file_name']
        url = f'https://lab.plantnet.org/seafile/d/${SHARE_TOKEN}/files/?p=/{name}&dl=1'
        print(url)
" | while read -r url; do
            filename=$(echo "$url" | grep -oP 'p=/\K[^&]+')
            echo "  Downloading: $filename"
            if [ -f "$TRAIN_DIR/$filename" ]; then
                echo "    ✓ Already exists, skipping"
            else
                aria2c -x 16 -s 16 -d "$TRAIN_DIR" -o "$filename" "$url" || \
                    wget -c -q --show-progress -O "$TRAIN_DIR/$filename" "$url" || \
                    curl -L -o "$TRAIN_DIR/$filename" "$url"
            fi
        done
        echo "  ✓ Training data downloaded"
    else
        echo "  ✗ Seafile API not directly accessible."
        echo "    Please download manually from: $SEAFILE_URL"
        echo "    Then place files in: $TRAIN_DIR/"
        echo ""
        echo "    Alternative: Use the Python downloader:"
        echo "    python3 scripts/seafile_downloader.py --url $SEAFILE_URL --output $TRAIN_DIR"
    fi

    # Extract tar/zip files if present
    echo "  Checking for archives to extract..."
    for archive in "$TRAIN_DIR"/*.tar "$TRAIN_DIR"/*.tar.gz "$TRAIN_DIR"/*.zip; do
        if [ -f "$archive" ]; then
            echo "  Extracting: $(basename "$archive")"
            case "$archive" in
                *.tar.gz) pv "$archive" | pigz -d | tar xf - -C "$TRAIN_DIR" ;;
                *.tar)    pv "$archive" | tar xf - -C "$TRAIN_DIR" ;;
                *.zip)    unzip -q -o "$archive" -d "$TRAIN_DIR" ;;
            esac
            echo "    ✓ Extracted"
        fi
    done
}

# ══════════════════════════════════════════════════════
# STEP 3: Pseudo-quadrat images (unlabeled)
# ══════════════════════════════════════════════════════
download_pseudo_quadrats() {
    echo ""
    echo "── [3/4] Pseudo-Quadrat Images (~170 GB) ──"
    check_disk_space

    local PQ_DIR="$BASE_DIR/pseudo_quadrats"
    local SHARE_TOKEN="f3a63defc5f44220b194"

    echo "  Source: https://lab.plantnet.org/seafile/d/$SHARE_TOKEN"

    echo "  Fetching file list from Seafile..."
    local FILE_LIST
    FILE_LIST=$(curl -s "https://lab.plantnet.org/seafile/api/v2.1/share-links/${SHARE_TOKEN}/dirents/?path=/" 2>/dev/null || echo "FAILED")

    if echo "$FILE_LIST" | python3 -c "import sys, json; json.load(sys.stdin)" 2>/dev/null; then
        echo "  ✓ Seafile API accessible."
        echo "$FILE_LIST" | python3 -c "
import sys, json
data = json.load(sys.stdin)
files = data.get('dirent_list', [])
for f in files:
    if 'file_name' in f:
        name = f['file_name']
        url = f'https://lab.plantnet.org/seafile/d/${SHARE_TOKEN}/files/?p=/{name}&dl=1'
        print(url)
" | while read -r url; do
            filename=$(echo "$url" | grep -oP 'p=/\K[^&]+')
            echo "  Downloading: $filename"
            if [ -f "$PQ_DIR/$filename" ]; then
                echo "    ✓ Already exists, skipping"
            else
                aria2c -x 16 -s 16 -d "$PQ_DIR" -o "$filename" "$url" || \
                    wget -c -q --show-progress -O "$PQ_DIR/$filename" "$url" || \
                    curl -L -o "$PQ_DIR/$filename" "$url"
            fi
        done

        # Extract
        for archive in "$PQ_DIR"/*.tar "$PQ_DIR"/*.tar.gz; do
            if [ -f "$archive" ]; then
                echo "  Extracting: $(basename "$archive")"
                pv "$archive" | pigz -d 2>/dev/null | tar xf - -C "$PQ_DIR" || \
                    pv "$archive" | tar xf - -C "$PQ_DIR"
                echo "    ✓ Extracted"
            fi
        done
        echo "  ✓ Pseudo-quadrat data downloaded"
    else
        echo "  ✗ Seafile API not directly accessible."
        echo "    Download manually from: https://lab.plantnet.org/seafile/d/$SHARE_TOKEN"
        echo "    Place files in: $PQ_DIR/"
    fi
}

# ══════════════════════════════════════════════════════
# STEP 4: Pre-trained models from Zenodo
# ══════════════════════════════════════════════════════
download_pretrained_models() {
    echo ""
    echo "── [4/4] Pre-trained Models (Zenodo) ──"
    check_disk_space

    local MODEL_DIR="$BASE_DIR/models"

    # Zenodo record 10848263
    local ZENODO_API="https://zenodo.org/api/records/10848263"
    echo "  Fetching model URLs from Zenodo..."

    local RECORD
    RECORD=$(curl -s "$ZENODO_API")

    if echo "$RECORD" | python3 -c "import sys, json; json.load(sys.stdin)" 2>/dev/null; then
        echo "$RECORD" | python3 -c "
import sys, json
data = json.load(sys.stdin)
files = data.get('files', [])
for f in files:
    name = f['key']
    size_mb = f['size'] / 1e6
    url = f['links']['self']
    print(f'  {name} ({size_mb:.0f} MB)')
    print(f'    URL: {url}')
" 
        echo ""
        echo "  Downloading models..."
        echo "$RECORD" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for f in data.get('files', []):
    print(f['key'] + '|' + f['links']['self'])
" | while IFS='|' read -r filename url; do
            if [ -f "$MODEL_DIR/$filename" ]; then
                echo "    ✓ $filename already exists, skipping"
            else
                echo "    Downloading: $filename"
                aria2c -x 16 -s 16 -d "$MODEL_DIR" -o "$filename" "$url" || \
                    wget -c -q --show-progress -O "$MODEL_DIR/$filename" "$url"
            fi
        done
        echo "  ✓ Pre-trained models downloaded"
    else
        echo "  ✗ Zenodo API error. Download manually from:"
        echo "    https://zenodo.org/records/10848263"
        echo "    Place files in: $MODEL_DIR/"
    fi
}

# ══════════════════════════════════════════════════════
# Main: run all downloads
# ══════════════════════════════════════════════════════
echo ""
echo "Starting downloads at $(date)"
echo ""

download_kaggle_data
download_training_data
download_pseudo_quadrats
download_pretrained_models

echo ""
echo "============================================"
echo "  All downloads complete! $(date)"
echo "============================================"
echo ""
echo "  Directory sizes:"
du -sh "$BASE_DIR"/* 2>/dev/null || true
echo ""
echo "  Next: Upload to Kaggle with:"
echo "    python3 scripts/upload_to_kaggle.py"
