"""MambaOut trainer — gated CNN that matches VMamba accuracy on ImageNet-scale tasks.

True VMamba (selective state-space) isn't in this timm build. MambaOut (Yu et al. 2024,
"MambaOut: Do We Really Need Mamba for Vision?") shows that for non-causal,
fixed-length tasks like image classification, a gated ConvNeXt-style block reaches
the same accuracy as VMamba while being faster — no SSM scan needed.

Default backbone: mambaout_base (~85 M params, ~ConvNeXt-B speed).
Other reasonable picks: mambaout_small (faster), mambaout_tiny (very fast for sweeps).
"""
import os
import math
import torch
import torch.multiprocessing  # noqa: F401  — keep default 'file_descriptor'; see train_gfnet.py
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


# ── 1. Backbone ──────────────────────────────────────────────────────────────
def build_mambaout(model_name='mambaout_base', num_classes=7806, resolution=224):
    print(f"[*] Building {model_name} at {resolution}px (num_classes={num_classes})...")
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    return model


# ── 2. Loaders ───────────────────────────────────────────────────────────────
def _load_species_mapping(mapping_path="/workspace/plantclef/processed/species_ids_mapping.csv"):
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

    effective_workers = min(num_workers, len(shard_urls))
    if effective_workers < num_workers:
        print(f"[*] Capping num_workers {num_workers} → {effective_workers} (only {len(shard_urls)} shards)")

    print(f"[*] Found {len(shard_urls)} shards for {shards_pattern}")

    dataset = wds.WebDataset(shard_urls, resampled=is_train, shardshuffle=is_train)
    if is_train:
        dataset = dataset.shuffle(1000)

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
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        if len(cls_bytes) == 4:
            raw_id = int.from_bytes(cls_bytes, byteorder='little')
        else:
            raw_id = int(cls_bytes.decode('utf-8').strip())
        cls_int = species_map.get(raw_id, 0)
        return tfm(img), cls_int

    dataset = dataset.map(custom_decode).batched(batch_size)
    loader = wds.WebLoader(
        dataset, batch_size=None, num_workers=effective_workers,
        pin_memory=True,
        persistent_workers=effective_workers > 0,
        prefetch_factor=2 if effective_workers > 0 else None,
    )
    if is_train:
        loader = loader.with_epoch(1000)
    return loader


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def _to_normalized_float(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    x = x.to(device, non_blocking=True).float().mul_(1.0 / 255.0)
    mean = _IMAGENET_MEAN.to(device, non_blocking=True)
    std  = _IMAGENET_STD.to(device, non_blocking=True)
    return (x - mean) / std


def cosine_warmup_lr(step: int, warmup_steps: int, total_steps: int, base_lr: float, min_lr: float = 1e-6) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ── 3. Train loop ─────────────────────────────────────────────────────────────
def train(args):
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing MambaOut Training on {device} | Epochs: {args.epochs}")

    amp_dtype = torch.bfloat16 if (
        args.use_amp and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    ) else torch.float16
    use_scaler = args.use_amp and amp_dtype is torch.float16
    print(f"[*] AMP: dtype={amp_dtype}  scaler={use_scaler}")

    train_loader = build_wds_loader("/dev/shm/shards/train_*.tar", args.batch_size, args.num_workers, args.resolution, is_train=True)
    val_loader   = build_wds_loader("/dev/shm/shards/val_*.tar",   args.batch_size, args.num_workers, args.resolution, is_train=False)

    model = build_mambaout(model_name=args.model, num_classes=args.num_classes, resolution=args.resolution).to(device)

    ema = ModelEmaV3(model, decay=args.ema_decay) if args.use_ema else None
    if ema is not None:
        print(f"[*] EMA enabled (decay={args.ema_decay}) — val will use EMA weights.")

    if args.compile:
        # MambaOut is plain conv + gating — max-autotune is safe and fastest.
        model = torch.compile(model, mode="max-autotune")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05,
                            betas=(0.9, 0.95), fused=True)

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

    out_dir = "models/mambaout"
    os.makedirs(out_dir, exist_ok=True)
    latest_ckpt_path = f"{out_dir}/latest_mambaout.pt"
    best_ckpt_path   = f"{out_dir}/best_mambaout.pt"

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
    global_step = start_epoch * 1000

    for epoch in range(start_epoch, args.epochs):
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

            inputs  = _to_normalized_float(inputs, device)
            targets = targets.to(device, non_blocking=True)
            targets_hard = targets
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
        avg_train_acc  = train_acc  / (1000 if train_loader else 50)
        train_f1 = f1_score(train_targets, train_preds, average='macro', zero_division=0)

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

            avg_val_loss = val_loss / max(1, n_val_batches)
            avg_val_acc  = val_acc  / max(1, n_val_batches)
            val_f1 = f1_score(val_targets, val_preds, average='macro', zero_division=0)

            print(f"➜ Train: Loss {avg_train_loss:.4f} | Acc {avg_train_acc:.4f} | F1 {train_f1:.4f}")
            print(f"➜ Val:   Loss {avg_val_loss:.4f} | Acc {avg_val_acc:.4f} | F1 {val_f1:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                best_state = ema.module.state_dict() if ema is not None else model.state_dict()
                torch.save(best_state, best_ckpt_path)
                print("  ↳ [New Best F1! Checkpoint saved.]")
            else:
                patience_counter += 1
                print(f"  ↳ [No improvement. Patience: {patience_counter}/{args.patience}]")

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
    parser.add_argument("--model",       type=str, default="mambaout_base",
                        help="timm model name. Try mambaout_small (faster) or mambaout_tiny (sweeps).")
    parser.add_argument("--batch_size",  type=int, default=1024)
    parser.add_argument("--resolution",  type=int, default=224)
    parser.add_argument("--num_classes", type=int, default=7806)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--epochs",      type=int, default=150)
    parser.add_argument("--patience",    type=int, default=10)
    parser.add_argument("--resume",      action="store_true")
    # MambaOut is fully ImageNet-pretrained — 5e-4 is the standard fine-tuning LR.
    parser.add_argument("--lr",          type=float, default=5e-4)
    parser.add_argument("--grad_clip",   type=float, default=1.0)
    parser.add_argument("--use_amp",     action="store_true", default=True)
    parser.add_argument("--compile",     action="store_true", default=True)

    parser.add_argument("--mixup_alpha",  type=float, default=0.2)
    parser.add_argument("--cutmix_alpha", type=float, default=1.0)
    parser.add_argument("--use_ema",      action="store_true", default=True)
    parser.add_argument("--ema_decay",    type=float, default=0.9998)

    train(parser.parse_args())
