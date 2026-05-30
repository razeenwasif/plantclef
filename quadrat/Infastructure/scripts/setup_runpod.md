# RunPod Setup Guide for PlantCLEF 2026

## 1. Create a Network Volume (Persistent Storage)

Network volumes persist across pod restarts and are essential for storing ~500GB+ of data.

1. Go to [RunPod Console](https://www.runpod.io/console/pods) → **Storage** → **Network Volumes**
2. Click **+ Network Volume**
3. Configure:
   - **Name:** `plantclef-data`
   - **Region:** Pick the region closest to you (or cheapest)
   - **Size:** `600 GB` minimum (training ~160GB + pseudo-quadrat ~170GB + test ~8GB + models + workspace)
   - If you only want the max-side-800px training set: **400 GB** is sufficient
   - If you want both training set versions: **900 GB**
4. Click **Create**

> **Cost note:** Network volume storage costs ~$0.07/GB/month. 600GB ≈ $42/month.

## 2. Create a GPU Pod

1. Go to **Pods** → **+ Deploy**
2. Select a template: **RunPod PyTorch 2.x** (comes with CUDA, PyTorch, etc.)
3. GPU selection:
   - **For downloading/uploading only:** Any cheap GPU (e.g., RTX 3090, RTX 4090) or even a CPU pod
   - **For training later:** A100 80GB or H100 recommended for ViT-base models with large batch sizes
   - **Budget option:** RTX 4090 24GB (works fine for inference and fine-tuning with smaller batches)
4. Configure:
   - **Container Disk:** `50 GB` (for OS, packages, temp files)
   - **Network Volume:** Attach `plantclef-data` → mount at `/workspace/data`
5. Click **Deploy**

## 3. Accessing Your Pod

Once deployed:
- Click **Connect** → **Start Web Terminal** (browser-based terminal)
- Or use **SSH**: Click **Connect** → copy the SSH command

## 4. Verify Setup

```bash
# Check GPU
nvidia-smi

# Check storage
df -h /workspace/data

# Check Python/PyTorch
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## 5. First-Time Setup

Run the environment setup script:

```bash
cd /workspace/data
git clone <your-repo-url> plantclef  # or just create the directory
cd plantclef
bash scripts/setup_environment.sh
```

Then download the datasets:

```bash
bash scripts/download_datasets.sh
```

## Tips

- **Stop your pod** when not using it to save on GPU costs (network volume persists)
- **Use `tmux` or `screen`** for long downloads so they survive SSH disconnects
- RunPod provides high bandwidth (~1-5 Gbps) which helps with large downloads
- The web terminal auto-disconnects after idle; prefer SSH + tmux for long tasks
