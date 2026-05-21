import os
import sys
from pathlib import Path

# --- ORACLE: ATOMIC ENVIRONMENT HARDENING ---
# 1. Path Sync
# Using .resolve() ensures absolute paths even if script is called via relative path
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

# 2. Hardware Satiation (Blackwell Max-Efficiency)
os.environ["TORCH_CUDA_MATMUL_TF32"] = "1"
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
# os.environ["HF_HUB_OFFLINE"] = "1" # ORACLE: Commented out to allow dynamic downloads if needed

# 3. CPU Satiation (Optimized for Ryzen 9 / EPYC)
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

# 4. Custom CUDA Extension Linking
try:
    import torch
    torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
    if torch_lib not in os.environ.get("LD_LIBRARY_PATH", ""):
        os.environ["LD_LIBRARY_PATH"] = f"{os.environ.get('LD_LIBRARY_PATH', '')}:{torch_lib}"
except ImportError:
    pass

import time
import argparse
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm

# These imports MUST come after sys.path is updated
from src.data.dataloader import get_dali_loaders, PlantDALIPipeline, DALIGenericIterator
from src.training.lightweight import get_lightweight_model
from src import config

def build_extra_loader(csv_path, batch_size, resolution, device_id):
    """
    Builds a DALI loader for a custom CSV manifest (e.g. iNat data).
    Expects columns: 'image_path' and 'species_id'.
    """
    print(f"[Loader] Building extra loader from {csv_path}...")
    df = pd.read_csv(csv_path)

    # iNat downloader saves files with `.jpg` even when the source URL was a GIF, which
    # crashes nvjpeg mid-pipeline ("GIF images are not supported") and invalidates the
    # whole DALI graph. The `url` column preserves the original extension, so filter on
    # that — costs nothing and is exact.
    if 'url' in df.columns:
        before = len(df)
        ext_pat = df['url'].astype(str).str.rsplit('.', n=1).str[-1].str.lower()
        df = df[ext_pat.isin(['jpg', 'jpeg', 'png'])].reset_index(drop=True)
        dropped = before - len(df)
        if dropped:
            print(f"[Loader] Dropped {dropped} non-JPEG/PNG entries (gif/webp/etc.) by URL ext.")

    # Map species_id to 0-indexed classes using the project's mapping
    mapping_path = "/workspace/plantclef/processed/species_ids_mapping.csv"
    if os.path.exists(mapping_path):
        df_map = pd.read_csv(mapping_path, header=None)
        id_to_idx = {int(row[0]): i for i, row in df_map.iterrows()}
        labels = [id_to_idx.get(int(sid), 0) for sid in df['species_id']]
    else:
        print("[Warning] Species mapping not found. Using raw species_id (this may crash if not 0-indexed).")
        labels = df['species_id'].tolist()

    file_paths = df['image_path'].tolist()
    
    pipe = PlantDALIPipeline(
        batch_size=batch_size,
        num_threads=16,
        device_id=device_id,
        file_paths=file_paths,
        labels=labels,
        training=True,
        resolution=resolution
    )
    pipe.build()
    return DALIGenericIterator([pipe], ['data', 'label'], reader_name="Reader", auto_reset=False)

def _build_species_lookup(device, mapping_path: str = "/workspace/plantclef/processed/species_ids_mapping.csv"):
    """Returns a [max_raw_id+1] int64 tensor mapping raw species IDs (e.g. 1361087) to
    zero-indexed class IDs (0..7805). Unmapped slots are -1.

    The DALI WebDataset path (dataloader.py: fn.reinterpret) returns the .cls bytes
    as raw species IDs, NOT class indices. Without this lookup, the previous clamp
    silently funnelled every label to 7807, training was meaningless, and any
    pre-clamp batch would trigger the nll_loss `t < n_classes` device assert.
    """
    import csv
    pairs = []
    with open(mapping_path) as f:
        for idx, row in enumerate(csv.reader(f)):
            if row:
                pairs.append((int(row[0].strip()), idx))
    max_id = max(rid for rid, _ in pairs)
    lut = torch.full((max_id + 1,), -1, dtype=torch.long, device=device)
    for rid, cls in pairs:
        lut[rid] = cls
    print(f"[SpeciesLUT] Built {len(pairs)}-entry raw_id → class_idx map "
          f"(table size: {(max_id + 1) * 8 / 1e6:.1f} MB on {device}).")
    return lut


def train_lightweight():
    parser = argparse.ArgumentParser(description="ORACLE: Lightweight Model Training")
    parser.add_argument("--model", type=str, default="bioclip", choices=["bioclip", "dinov3"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--extra_csv", type=str, default="/workspace/plantclef/processed/inat_research_grade_manifest.csv",
                        help="Path to extra training data manifest.")
    parser.add_argument("--only_extra", action="store_true", help="Only train on the extra data.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 7808 
    
    # ORACLE: Explicitly initialize CUDA context for DALI
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        torch.cuda.set_device(device_id)
        # Dummy operation to force context creation
        torch.cuda.empty_cache()
        _ = torch.tensor([0.0]).cuda()
        print(f"[Init] CUDA Context initialized on device {device_id}")
    else:
        device_id = 0

    # 1. Build Model
    model = get_lightweight_model(args.model, num_classes=num_classes, input_res=args.resolution).to(device)

    # Lookup for the WebDataset path which returns raw species IDs from .cls bytes.
    species_lut = _build_species_lookup(device)

    # 2. Setup Data Loaders. Each entry is (loader, label_kind):
    #   - "raw_id":    DALI WebDataset shards → raw species IDs (e.g. 1361087); needs LUT.
    #   - "class_idx": build_extra_loader already maps via id_to_idx → class indices.
    loaders = []

    # Official Shards
    if not args.only_extra:
        train_loader, val_loader, _, _ = get_dali_loaders(
            batch_size=args.batch_size,
            resolution=args.resolution,
            device_id=device_id,
            training=True
        )
        loaders.append((train_loader, "raw_id"))
    else:
        val_loader = None

    # Extra Data (iNat)
    if args.extra_csv and os.path.exists(args.extra_csv):
        extra_loader = build_extra_loader(args.extra_csv, args.batch_size, args.resolution, device_id)
        loaders.append((extra_loader, "class_idx"))

    # 3. Optimizer: Lion
    try:
        from lion_pytorch import Lion
        optimizer = Lion(model.parameters(), lr=args.lr, weight_decay=0.01)
        print("[Training] Using Lion optimizer.")
    except ImportError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        print("[Warning] Lion not found, using AdamW.")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # 5. Loss: Label Smoothing Cross Entropy (Good for large-scale long-tail)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    print(f"\n--- Starting Lightweight {args.model.upper()} Training ---")
    print(f"Target: {args.epochs} epochs | Batch Size: {args.batch_size} | Res: {args.resolution}px")

    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0

        for loader_idx, (loader, label_kind) in enumerate(loaders):
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs} [Loader {loader_idx}]")
            for data in pbar:
                inputs = data[0]["data"].to(device)
                targets = data[0]["label"].squeeze().long().to(device)

                # Map labels into the model's class space [0, num_classes).
                if label_kind == "raw_id":
                    # DALI WebDataset returns raw species IDs (e.g. 1361087) — look up.
                    in_range = (targets >= 0) & (targets < species_lut.shape[0])
                    safe = torch.where(in_range, targets, torch.zeros_like(targets))
                    targets = species_lut[safe]
                    valid = in_range & (targets >= 0)
                else:
                    # build_extra_loader already mapped to class indices.
                    valid = (targets >= 0) & (targets < num_classes)

                if not valid.all():
                    inputs  = inputs[valid]
                    targets = targets[valid]
                    if targets.numel() == 0:
                        continue  # whole batch unmapped — skip

                optimizer.zero_grad()

                # AMP Autocast (BFloat16 for Blackwell speed)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = outputs.max(1)
                epoch_total += targets.size(0)
                epoch_correct += predicted.eq(targets).sum().item()
                
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100.*epoch_correct/epoch_total:.2f}%"})
            
            loader.reset()
        
        scheduler.step()
        
        # Validation Phase
        if val_loader:
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                # ORACLE: Autocast required for TransformerEngine layers in eval mode
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    for data in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                        inputs = data[0]["data"].to(device, non_blocking=True)
                        targets = data[0]["label"].squeeze().long().to(device, non_blocking=True)
                        
                        # Apply same safety clamping as training
                        targets = torch.clamp(targets, 0, num_classes - 1)
                        
                        outputs = model(inputs)
                        _, predicted = outputs.max(1)
                        val_total += targets.size(0)
                        val_correct += predicted.eq(targets).sum().item()
            
            print(f"Epoch {epoch+1} Val Acc: {100.*val_correct/val_total:.2f}%")
            val_loader.reset()

        # Save Checkpoint
        save_path = f"models/lightweight_{args.model}_latest.pth"
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), save_path)
    
    total_time = time.time() - start_time
    print(f"\n[Done] Training completed in {total_time/3600:.2f} hours.")
    print(f"[Output] Final weights saved to {save_path}")

if __name__ == "__main__":
    train_lightweight()
