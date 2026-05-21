# QuickStart: RunPod + Data Pipeline

## Step-by-Step

### 1. Set Up RunPod

Follow the guide in `setup_runpod.md`. TL;DR:

```bash
# Create a Network Volume: 600GB, name it "plantclef-data"
# Deploy a pod: PyTorch template, attach the volume at /workspace/data
```

### 2. SSH Into Your Pod

```bash
# Copy SSH command from RunPod dashboard, e.g.:
ssh root@<pod-ip> -p <port> -i ~/.ssh/id_ed25519
```

### 3. Clone This Project

```bash
cd /workspace/data
# If using git:
git clone <your-repo-url> plantclef && cd plantclef
# Otherwise, upload the scripts folder via SCP:
scp -P <port> -r scripts/ root@<pod-ip>:/workspace/plantclef/scripts/
```

### 4. Run Environment Setup

```bash
bash scripts/setup_environment.sh
```

### 5. Configure Kaggle API

```bash
mkdir -p ~/.kaggle
# Paste your credentials (get from https://www.kaggle.com/settings → API)
echo '{"username":"YOUR_USERNAME","key":"YOUR_API_KEY"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Verify
kaggle competitions list -s plantclef
```

### 6. Download Datasets

Use tmux so downloads survive SSH disconnects:

```bash
tmux new -s download
bash scripts/download_datasets.sh
# Ctrl+B then D to detach, `tmux attach -t download` to reattach
```

Or download individually:

```bash
# Kaggle test data only (~8.3 GB)
kaggle competitions download -c plantclef-2026 -p /workspace/plantclef/raw/test/
unzip /workspace/plantclef/raw/test/*.zip -d /workspace/plantclef/raw/test/

# Training data via Seafile helper (~160 GB, takes several hours)
python3 scripts/seafile_downloader.py \
    --url https://lab.plantnet.org/seafile/d/303fec50b1a544c6a2ed \
    --output /workspace/plantclef/raw/train \
    --workers 4

# Pseudo-quadrat data (~170 GB)
python3 scripts/seafile_downloader.py \
    --url https://lab.plantnet.org/seafile/d/f3a63defc5f44220b194 \
    --output /workspace/plantclef/raw/pseudo_quadrats \
    --workers 4

# Pre-trained models (small, fast)
# These download automatically in download_datasets.sh via Zenodo API
```

### 7. Upload to Kaggle

```bash
# Upload everything (auto-splits large datasets)
python3 scripts/upload_to_kaggle.py --what all

# Or upload individually
python3 scripts/upload_to_kaggle.py --what train
python3 scripts/upload_to_kaggle.py --what pseudo
python3 scripts/upload_to_kaggle.py --what models
```

### 8. Verify on Kaggle

Your datasets will appear at: `https://www.kaggle.com/<username>/datasets`

You can then attach them to Kaggle notebooks for the competition.

---

## Storage Estimates

| Dataset | Size | Download Time (1 Gbps) |
|---------|------|----------------------|
| Test set + metadata (Kaggle) | ~8.3 GB | ~1 min |
| Training (max-side 800px) | ~160 GB | ~20 min |
| Training (min-side 800px) | ~281 GB | ~35 min |
| Pseudo-quadrat images | ~170 GB | ~25 min |
| Pre-trained models | ~1 GB | <1 min |
| **Total (max-side)** | **~340 GB** | **~45 min** |

RunPod typically provides 1-5 Gbps bandwidth, so actual times may vary.

## Troubleshooting

**Seafile download fails:**
- Try the Python downloader: `python3 scripts/seafile_downloader.py --url <URL> --output <DIR>`
- If the API is blocked, download via browser and `scp` to the pod
- Some Seafile links may require visiting the URL in a browser first to "activate" the share

**Kaggle upload fails with "too large":**
- The upload script auto-splits datasets >95GB
- If it still fails, reduce `max_size_gb` in `upload_to_kaggle.py`

**Pod ran out of disk:**
- Check usage: `df -h /workspace/data`
- Delete archives after extraction: `rm /workspace/plantclef/raw/**/*.tar`
- Consider downloading only `max-side-800px` (160GB vs 281GB)

**SSH disconnected during download:**
- Always use `tmux` or `screen` for long-running tasks
- Downloads with `aria2c` support resume automatically
