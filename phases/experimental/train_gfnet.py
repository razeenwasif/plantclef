import os
import time
import math
import torch
import torch.multiprocessing
# DataLoader IPC strategy: keep the default 'file_descriptor' (anonymous memfds).
# DO NOT switch to 'file_system' — despite the name it does NOT use /tmp; it calls
# shm_open() which creates named files in /dev/shm, the very space being eaten by
# the WebDataset shards. With shards in /dev/shm and 1M FD limit, file_descriptor
# is the right choice.

import torch.nn as nn
import torch.optim as optim
import timm
import argparse
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import f1_score
import webdataset as wds
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.utils import ModelEmaV3

# ── 1. The Core Physics: Complex-Valued 2D FFT Filter ────────────────────────
class ComplexFourierFilter(nn.Module):
    def __init__(self, dim, h, w):
        super().__init__()
        self.complex_weight = nn.Parameter(
            torch.randn(dim, h, w // 2 + 1, 2, dtype=torch.float32) * 0.02
        )

    def forward(self, x, **kwargs):
        B, N, C = x.shape
        cls_token = x[:, 0:1, :]
        x_spatial = x[:, 1:, :]
        
        H = int((N - 1) ** 0.5)
        W = H
        x_spatial = x_spatial.view(B, H, W, C).permute(0, 3, 1, 2).contiguous().float()
        
        x_fft = torch.fft.rfft2(x_spatial, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x_fft = x_fft * weight
        x_filtered = torch.fft.irfft2(x_fft, s=(H, W), norm='ortho')
        
        x_filtered = x_filtered.to(x.dtype)
        x_filtered = x_filtered.permute(0, 2, 3, 1).view(B, H * W, C)
        return torch.cat([cls_token, x_filtered], dim=1)


# ── 2. Architectural Surgery: ViT -> GFNet ───────────────────────────────────
def build_gfnet(model_name='vit_base_patch16_224', num_classes=7806, resolution=224):
    print(f"[*] Building GFNet from {model_name} base at {resolution}px (global_pool=avg)...")
    # global_pool='avg' is critical: the FFT skips CLS (it has no spatial dims),
    # so a CLS-token classifier head would see zero patch information across all
    # 12 layers. Mean-pooling the patches matches Rao et al. 2021 GFNet.
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes,
                              img_size=resolution, global_pool='avg')
    
    patch_size = model.patch_embed.patch_size[0]
    grid_size = resolution // patch_size
    dim = model.embed_dim
    
    replaced_count = 0
    for block in model.blocks:
        block.attn = ComplexFourierFilter(dim=dim, h=grid_size, w=grid_size)
        replaced_count += 1
        
    print(f"[*] Surgically replaced {replaced_count} Attention layers with 2D FFT Filters.")
    return model


# ── 3. High-Throughput RAM-Disk Loaders ───────────────────────────────────────
def _load_species_mapping(mapping_path="/workspace/plantclef/processed/species_ids_mapping.csv"):
    """Loads mapping from raw Species ID (e.g. 1361087) to zero-indexed class (e.g. 0-7805)."""
    import csv
    mapping = {}
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r') as f:
            reader = csv.reader(f)
            for idx, row in enumerate(reader):
                if row:
                    mapping[int(row[0].strip())] = idx
        print(f"[*] Loaded {len(mapping)} species mappings.")
    else:
        print(f"[!] Mapping file not found at {mapping_path}")
    return mapping

def build_wds_loader(shards_pattern, batch_size, num_workers, resolution, is_train=True):
    import glob
    import io
    from PIL import Image
    
    shard_urls = sorted(glob.glob(shards_pattern))
    if not shard_urls:
        print(f"[!] No shards found for pattern: {shards_pattern}")
        return None

    # WebDataset round-robins shards across workers; more workers than shards
    # leaves the surplus idle. Cap to shard count.
    effective_workers = min(num_workers, len(shard_urls))
    if effective_workers < num_workers:
        print(f"[*] Capping num_workers {num_workers} → {effective_workers} (only {len(shard_urls)} shards)")

    print(f"[*] Found {len(shard_urls)} shards for {shards_pattern}")
    
    dataset = wds.WebDataset(shard_urls, resampled=is_train, shardshuffle=True if is_train else False)
    if is_train:
        dataset = dataset.shuffle(1000)
        
    # IMPORTANT: keep the worker tensor as uint8 to slash the worker→main IPC footprint
    # by 4×. Float conversion + normalization happens on GPU in the train step.
    # Train pipeline: RandomResizedCrop + RandAugment + RandomErasing — heavy regularization
    # to close the 5× train-val F1 gap. RandAugment must run on PIL (before PILToTensor);
    # RandomErasing runs on the uint8 tensor.
    if is_train:
        tfm = transforms.Compose([
            transforms.RandomResizedCrop(resolution, scale=(0.7, 1.0), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.PILToTensor(),
            transforms.RandomErasing(p=0.25, value=0),
        ])
    else:
        tfm = transforms.Compose([
            transforms.Resize((resolution, resolution), antialias=True),
            transforms.PILToTensor(),
        ])
    
    species_map = _load_species_mapping()
    
    def custom_decode(sample):
        img_bytes = sample.get("jpg") or sample.get("png")
        cls_bytes = sample.get("cls") or sample.get("txt")
        
        # Decode image
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        # Decode class (handle both ASCII text and 4-byte little-endian binary)
        if len(cls_bytes) == 4:
            raw_id = int.from_bytes(cls_bytes, byteorder='little')
        else:
            raw_id = int(cls_bytes.decode('utf-8').strip())
            
        # Map raw ID (e.g. 1361087) to class index (0 - 7805)
        # Fallback to 0 to prevent CUDA assert crash if ID is weird
        cls_int = species_map.get(raw_id, 0)
            
        return tfm(img), cls_int
        
    dataset = dataset.map(custom_decode).batched(batch_size)
    loader = wds.WebLoader(
        dataset, batch_size=None, num_workers=effective_workers,
        pin_memory=True,
        persistent_workers=effective_workers > 0,
        prefetch_factor=2 if effective_workers > 0 else None,  # 16 workers × 2 ≈ 20GB in-flight
    )
    # Give the loader a fixed epoch size so tqdm knows when to stop
    if is_train:
        loader = loader.with_epoch(1000) # Process 1000 batches per epoch
    return loader


# GPU-side float-cast + ImageNet normalization (replaces the per-worker ToTensor/Normalize).
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def _to_normalized_float(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """uint8 [B, C, H, W] on host → float32 [B, C, H, W] normalized on device."""
    x = x.to(device, non_blocking=True).float().mul_(1.0 / 255.0)
    mean = _IMAGENET_MEAN.to(device, non_blocking=True)
    std  = _IMAGENET_STD.to(device, non_blocking=True)
    return (x - mean) / std


def cosine_warmup_lr(step: int, warmup_steps: int, total_steps: int, base_lr: float, min_lr: float = 1e-6) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ── 4. Training Loop with Early Stopping & Metrics ────────────────────────────
def train(args):
    # Free speed knobs — set before any tensor allocation.
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing GFNet Training on {device} | Epochs: {args.epochs}")

    # AMP dtype: bf16 on Blackwell/Ampere/Hopper, else fp16. GradScaler is fp16-only —
    # using it with bf16 is a no-op at best and can NaN at worst.
    amp_dtype = torch.bfloat16 if (
        args.use_amp and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    ) else torch.float16
    use_scaler = args.use_amp and amp_dtype is torch.float16
    print(f"[*] AMP: dtype={amp_dtype}  scaler={use_scaler}")

    # Loaders pointing to /dev/shm
    train_loader = build_wds_loader("/dev/shm/shards/train_*.tar", args.batch_size, args.num_workers, args.resolution, is_train=True)
    val_loader   = build_wds_loader("/dev/shm/shards/val_*.tar", args.batch_size, args.num_workers, args.resolution, is_train=False)

    model = build_gfnet(num_classes=args.num_classes, resolution=args.resolution).to(device)

    # EMA must be created BEFORE torch.compile so its deepcopy holds the raw module.
    # Updates still see the live params of the compiled model (compile doesn't relocate
    # parameter storage), so this is the canonical timm recipe.
    ema = ModelEmaV3(model, decay=args.ema_decay) if args.use_ema else None
    if ema is not None:
        print(f"[*] EMA enabled (decay={args.ema_decay}) — val will use EMA weights.")

    if args.compile:
        # `reduce-overhead` is the safe high-throughput choice; `max-autotune` reliably
        # fails Dynamo tracing on torch.fft.rfft2/irfft2.
        model = torch.compile(model, mode="reduce-overhead")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05,
                            betas=(0.9, 0.95), fused=True)

    # Mixup/CutMix produces soft targets, so the train criterion must accept them.
    # Val targets stay hard ints and use the standard CE loss.
    mixup_fn = Mixup(
        mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha,
        prob=1.0, switch_prob=0.5, mode='batch',
        label_smoothing=0.1, num_classes=args.num_classes,
    ) if (args.mixup_alpha > 0 or args.cutmix_alpha > 0) else None
    train_criterion = SoftTargetCrossEntropy() if mixup_fn else nn.CrossEntropyLoss(label_smoothing=0.1)
    val_criterion   = nn.CrossEntropyLoss(label_smoothing=0.1)
    if mixup_fn:
        print(f"[*] Mixup enabled (mixup={args.mixup_alpha}, cutmix={args.cutmix_alpha}).")

    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)
    
    start_epoch = 0
    best_f1 = 0.0
    patience_counter = 0

    os.makedirs("models/gfnet", exist_ok=True)
    latest_ckpt_path = "models/gfnet/latest_gfnet.pt"
    
    # --- RESUME CAPABILITY ---
    if args.resume and os.path.exists(latest_ckpt_path):
        print(f"[*] Resuming from checkpoint: {latest_ckpt_path}")
        checkpoint = torch.load(latest_ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scaler_state_dict' in checkpoint and scaler is not None:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        if ema is not None and 'ema_state_dict' in checkpoint:
            ema.module.load_state_dict(checkpoint['ema_state_dict'])
        start_epoch = checkpoint['epoch']
        best_f1 = checkpoint.get('best_f1', 0.0)
        patience_counter = checkpoint.get('patience_counter', 0)
        print(f"[*] Resumed at epoch {start_epoch+1} with Best F1: {best_f1:.4f}")

    total_steps = 1000 * args.epochs
    warmup_steps = min(500, total_steps // 20)
    global_step = start_epoch * 1000 # Approximation if resuming

    for epoch in range(start_epoch, args.epochs):
        # --- TRAINING ---
        model.train()
        train_loss, train_acc = 0.0, 0.0
        train_preds, train_targets = [], []
        
        pbar = tqdm(total=1000 if train_loader else 50, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        
        iterator = train_loader if train_loader else range(50)
        for batch in iterator:
            if train_loader:
                inputs, targets = batch
            else:
                inputs = torch.randn(args.batch_size, 3, args.resolution, args.resolution)
                targets = torch.randint(0, args.num_classes, (args.batch_size,))
                
            inputs  = _to_normalized_float(inputs, device)   # uint8 → float32, ÷255, normalize
            targets = targets.to(device, non_blocking=True)
            targets_hard = targets  # keep int labels for train metrics before mixup soft-mixes them
            if mixup_fn is not None:
                inputs, targets = mixup_fn(inputs, targets)
            global_step += 1

            lr = cosine_warmup_lr(global_step, warmup_steps, total_steps, args.lr)
            for pg in optimizer.param_groups: pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=args.use_amp):
                outputs = model(inputs)
                loss = train_criterion(outputs, targets)

            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                optimizer.step()

            if ema is not None:
                ema.update(model)

            train_loss += loss.item()
            preds = outputs.argmax(dim=1)
            train_acc += (preds == targets_hard).float().mean().item()
            train_preds.extend(preds.cpu().tolist())
            train_targets.extend(targets_hard.cpu().tolist())
            
            pbar.update(1)
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{lr:.2e}"})
            
        pbar.close()
        avg_train_loss = train_loss / (1000 if train_loader else 50)
        avg_train_acc = train_acc / (1000 if train_loader else 50)
        train_f1 = f1_score(train_targets, train_preds, average='macro', zero_division=0)
        
        # --- VALIDATION ---
        # Use EMA weights when available — they're typically 1-2 F1 points stronger and
        # smooth out the per-epoch val noise dramatically.
        val_model = ema.module if ema is not None else model
        val_model.eval()
        val_loss, val_acc = 0.0, 0.0
        val_preds, val_targets = [], []

        if val_loader:
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
            n_val_batches = 0
            with torch.no_grad():
                for inputs, targets in pbar:
                    inputs  = _to_normalized_float(inputs, device)
                    targets = targets.to(device, non_blocking=True)
                    with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=args.use_amp):
                        outputs = val_model(inputs)
                        loss = val_criterion(outputs, targets)

                    val_loss += loss.item()
                    preds = outputs.argmax(dim=1)
                    val_acc += (preds == targets).float().mean().item()
                    val_preds.extend(preds.cpu().tolist())
                    val_targets.extend(targets.cpu().tolist())
                    n_val_batches += 1

                    pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # WebLoader has no len() when used with WebDataset (streaming through
            # shards round-robin), so count batches manually.
            avg_val_loss = val_loss / max(1, n_val_batches)
            avg_val_acc = val_acc / max(1, n_val_batches)
            val_f1 = f1_score(val_targets, val_preds, average='macro', zero_division=0)
            
            print(f"➜ Train: Loss {avg_train_loss:.4f} | Acc {avg_train_acc:.4f} | F1 {train_f1:.4f}")
            print(f"➜ Val:   Loss {avg_val_loss:.4f} | Acc {avg_val_acc:.4f} | F1 {val_f1:.4f}")
            
            # Early Stopping & Checkpointing
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                # Save EMA weights as best when EMA is on — that's what produced this F1.
                best_state = ema.module.state_dict() if ema is not None else model.state_dict()
                torch.save(best_state, "models/gfnet/best_gfnet.pt")
                print("  ↳ [New Best F1! Checkpoint saved.]")
            else:
                patience_counter += 1
                print(f"  ↳ [No improvement. Patience: {patience_counter}/{args.patience}]")

            # Always save the latest state for resuming
            ckpt = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_f1': best_f1,
                'patience_counter': patience_counter,
            }
            if ema is not None:
                ckpt['ema_state_dict'] = ema.module.state_dict()
            torch.save(ckpt, latest_ckpt_path)
                
            if patience_counter >= args.patience:
                print("\n[!] Early Stopping triggered.")
                break
        else:
            print(f"➜ Train (Synthetic): Loss {avg_train_loss:.4f} | Acc {avg_train_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--num_classes", type=int, default=7806)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    # GFNet replaces 12 attention layers with random-init FFT filters, so the model is
    # ~50% from-scratch. 1e-3 is the original GFNet/ViT-from-scratch recipe; the previous
    # 5e-4 (or worse, 5e-5) was too low — train F1 was crawling at +0.005/epoch.
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Max grad norm (paper §138)")
    parser.add_argument("--use_amp",   action="store_true", default=True)
    parser.add_argument("--compile",   action="store_true", default=True)

    # Regularization knobs — these address the 5× train-val F1 gap seen in the long run.
    parser.add_argument("--mixup_alpha",  type=float, default=0.2,
                        help="Mixup alpha. 0 disables mixup (cutmix can still apply).")
    parser.add_argument("--cutmix_alpha", type=float, default=1.0,
                        help="CutMix alpha. 0 disables cutmix. If both are 0, mixup is fully off.")
    parser.add_argument("--use_ema",      action="store_true", default=True,
                        help="Maintain EMA weights and validate against them.")
    parser.add_argument("--ema_decay",    type=float, default=0.9998)

    train(parser.parse_args())
