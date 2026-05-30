import torch
import torch.nn.functional as F
from torch.amp import autocast
from torchmetrics.classification import (MulticlassF1Score, MulticlassPrecision,
                                         MulticlassRecall, MulticlassAccuracy)
from tqdm import tqdm
import wandb

import config as _cfg
BATCH_SIZE      = getattr(_cfg, "BATCH_SIZE",      384)
CHUNK_SIZE      = getattr(_cfg, "CHUNK_SIZE",      32)
P2_CHUNK_SIZE   = getattr(_cfg, "P2_CHUNK_SIZE",   32)
MAX_VAL_BATCHES = getattr(_cfg, "MAX_VAL_BATCHES", 100)
PCA_COMPONENTS  = getattr(_cfg, "PCA_COMPONENTS",  512)
from .cache import chunked_backbone_forward, CachedFeatureDataset, CachedPCADataset
from .checkpoints import save_progress_checkpoint


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(model, loader, criterion, num_classes, device, pca_layer=None):
    """
    GPU-accelerated validation via torchmetrics.
    All state accumulates on GPU; single CPU transfer at .compute().
    Stops at MAX_VAL_BATCHES for speed (~25,600 images, within 0.5% of full val).

    Args:
        pca_layer (nn.Module): Optional GPU-native PCA transformation. If provided,
                               validation will use the model's phase1_head.
    """
    model.eval()
    val_loss = 0.0
    f1_m   = MulticlassF1Score(num_classes=num_classes, average='macro').to(device)
    prec_m = MulticlassPrecision(num_classes=num_classes, average='macro').to(device)
    rec_m  = MulticlassRecall(num_classes=num_classes, average='macro').to(device)
    acc_m  = MulticlassAccuracy(num_classes=num_classes, average='micro').to(device)

    val_chunk = CHUNK_SIZE * 2   # no_grad -> 2x chunk is safe
    i = 0

    # Ensure PCA layer is in eval mode if provided
    if pca_layer is not None:
        pca_layer.eval()

    with torch.no_grad():
        for i, data in enumerate(tqdm(loader, desc="Validating", unit="batch", leave=False)):
            if i >= MAX_VAL_BATCHES:
                break
            images = data[0]['data'].to(memory_format=torch.channels_last)
            labels = data[0]['label'].squeeze().long()

            # Blackwell: Use BF16 for backbone validation
            mixed_precision_dtype = torch.bfloat16

            with autocast(device_type='cuda', dtype=mixed_precision_dtype):
                feat_bio  = chunked_backbone_forward(model.bioclip,  images, val_chunk)
                feat_dino = chunked_backbone_forward(model.dinov2,   images, val_chunk)
                feat_conv = chunked_backbone_forward(model.convnext, images, val_chunk)
                
                # Concatenate raw backbone features
                fused_raw = torch.cat([feat_bio, feat_dino, feat_conv], dim=1)

                if pca_layer is not None:
                    # Phase 1 path
                    feat_pca  = pca_layer(fused_raw)
                    outputs   = model.phase1_head(feat_pca)
                else:
                    # Phase 2 path
                    fused_proj = model.proj_grouped(fused_raw)
                    outputs    = model.classifier(F.normalize(fused_proj, dim=1))
                
                # Handle Asymmetric Loss (requires one-hot targets)
                from .losses import AsymmetricLoss
                if isinstance(criterion, AsymmetricLoss):
                    targets = F.one_hot(labels, num_classes=num_classes).float()
                    loss = criterion(outputs, targets)
                else:
                    loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            f1_m.update(predicted, labels)
            prec_m.update(predicted, labels)
            rec_m.update(predicted, labels)
            acc_m.update(predicted, labels)

    metrics = {
        "loss":      val_loss / (i + 1),
        "acc":       acc_m.compute().item()  * 100.0,
        "f1":        f1_m.compute().item(),
        "precision": prec_m.compute().item(),
        "recall":    rec_m.compute().item(),
    }
    for m in [f1_m, prec_m, rec_m, acc_m]:
        m.reset()
    loader.reset()
    return metrics


# ---------------------------------------------------------------------------
# Phase 1: cached head training
# ---------------------------------------------------------------------------

def run_phase1_cached(model, cache, criterion, epoch, num_classes, device):
    """
    Phase 1 head training on cached features.

    If cache contains 'features_pca' (PCA-compressed features), trains the
    model's phase1_head -- a lightweight linear probe on PCA_COMPONENTS-d
    features.  This is ~3x faster than training the full projection heads on
    raw backbone features since the input is much smaller (512-d vs 3072-d).

    If no PCA features, falls back to training proj_bio/dino/conv + classifier
    on raw backbone features (original behaviour).
    """
    model.train()
    acc_m = MulticlassAccuracy(num_classes=num_classes, average='micro').to(device)

    use_pca = 'features_pca' in cache
    if use_pca:
        dataset = CachedPCADataset(cache)
        mode    = "PCA"
    else:
        dataset = CachedFeatureDataset(cache)
        mode    = "raw"

    # Blackwell: If features are already on GPU, num_workers=0 is faster (no pickling)
    is_on_gpu = False
    if use_pca and cache['features_pca'].is_cuda:
        is_on_gpu = True
    elif not use_pca and cache['bio'].is_cuda:
        is_on_gpu = True

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE * 4,
        shuffle=True, num_workers=0 if is_on_gpu else 4, 
        pin_memory=not is_on_gpu
    )

    running_loss = 0.0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader),
                desc=f"Epoch {epoch} [Phase1-{mode}]")

    if use_pca:
        for i, (feat_pca, labels) in pbar:
            feat_pca = feat_pca.to(device, non_blocking=True)
            labels   = labels.to(device,   non_blocking=True)

            with autocast(device_type='cuda', dtype=torch.bfloat16):
                # phase1_head: simple linear probe on PCA features
                outputs = model.phase1_head(feat_pca)
                loss    = criterion(outputs, labels)

            model.backward(loss)
            model.step()

            _, predicted = outputs.max(1)
            acc_m.update(predicted, labels)
            running_loss += loss.item()

            if i % 10 == 0:
                pbar.set_postfix({
                    "Loss": f"{running_loss/(i+1):.4f}",
                    "Acc":  f"{acc_m.compute().item()*100:.2f}%"
                })
    else:
        for i, (feat_bio, feat_dino, feat_conv, labels) in pbar:
            feat_bio  = feat_bio.to(device,  non_blocking=True)
            feat_dino = feat_dino.to(device, non_blocking=True)
            feat_conv = feat_conv.to(device, non_blocking=True)
            labels    = labels.to(device,    non_blocking=True)

            with autocast(device_type='cuda', dtype=torch.bfloat16):
                fused_raw  = torch.cat([feat_bio, feat_dino, feat_conv], dim=1)
                fused_proj = model.proj_grouped(fused_raw)
                outputs    = model.classifier(F.normalize(fused_proj, dim=1))
                
                # Handle Asymmetric Loss (requires one-hot targets)
                from .losses import AsymmetricLoss
                if isinstance(criterion, AsymmetricLoss):
                    targets = F.one_hot(labels, num_classes=num_classes).float()
                    loss = criterion(outputs, targets)
                else:
                    loss = criterion(outputs, labels)

            model.backward(loss)
            model.step()

            _, predicted = outputs.max(1)
            acc_m.update(predicted, labels)
            running_loss += loss.item()

            if i % 10 == 0:
                pbar.set_postfix({
                    "Loss": f"{running_loss/(i+1):.4f}",
                    "Acc":  f"{acc_m.compute().item()*100:.2f}%"
                })

    train_acc = acc_m.compute().item() * 100.0
    acc_m.reset()
    return train_acc


# ---------------------------------------------------------------------------
# Phase 2: LoRA fine-tuning with BioCLIP taxonomic feature distillation
# ---------------------------------------------------------------------------

# Distillation weight -- blend between hard label loss and BioCLIP alignment.
# 0.3 means 70% ASL+LogitAdj, 30% feature distillation.
# BioCLIP's Tree-of-Life features encode taxonomic similarity so this guides
# DINOv2 + ConvNeXt to learn a feature space consistent with botanical taxonomy.
KD_ALPHA = 0.3


def run_epoch(model, train_loader, criterion, epoch, num_classes, device, optimizer=None, start_step=0, p2_chunk_size=None):
    """
    Phase 2 LoRA training with BioCLIP taxonomic feature distillation.
    Supports both raw models (with explicit optimizer) and DeepSpeed engines.
    """
    model.train()
    acc_m = MulticlassAccuracy(num_classes=num_classes, average='micro').to(device)

    # Handle DeepSpeed vs Raw Model
    is_deepspeed = hasattr(model, 'backward')
    raw_model    = model.module if is_deepspeed else model
    
    # Use provided chunk size or default from config
    chunk_size = p2_chunk_size if p2_chunk_size is not None else P2_CHUNK_SIZE

    pbar         = tqdm(enumerate(train_loader), total=len(train_loader),
                        desc=f"Epoch {epoch} [Train]", initial=start_step)
    running_loss     = 0.0
    running_kd_loss  = 0.0

    total_steps = len(train_loader)
    save_interval = max(1, total_steps // 20)

    for i, data in pbar:
        if i < start_step:
            continue
        images = data[0]['data'].to(memory_format=torch.channels_last)
        labels = data[0]['label'].squeeze().long()

        # Blackwell: Use FP8 autocast if enabled in config
        # e4m3fn is best for forward/backward pass on Blackwell
        # Fallback to BF16 if FP8 is not supported or enabled
        mixed_precision_dtype = torch.bfloat16
        if getattr(_cfg, 'USE_FP8', False) and hasattr(torch, 'float8_e4m3fn'):
            mixed_precision_dtype = torch.float8_e4m3fn

        with autocast(device_type='cuda', dtype=mixed_precision_dtype):
            # If optimizer is provided (Warmup), all backbones are frozen.
            # We can skip gradient tracking for ALL backbones.
            backbone_ctx = torch.no_grad() if optimizer is not None else torch.enable_grad()
            
            if hasattr(raw_model, 'has_ext') and raw_model.has_ext:
                # --- OPTIMIZED BLACKWELL PATH ---
                import plantclef_ext
                with backbone_ctx:
                    s0 = torch.cuda.ExternalStream(raw_model.orchestrator.get_stream(0))
                    s1 = torch.cuda.ExternalStream(raw_model.orchestrator.get_stream(1))
                    s2 = torch.cuda.ExternalStream(raw_model.orchestrator.get_stream(2))

                    with torch.cuda.stream(s0):
                        feat_bio_raw = chunked_backbone_forward(raw_model.bioclip, images, chunk_size)
                    with torch.cuda.stream(s1):
                        feat_dino_raw = chunked_backbone_forward(raw_model.dinov2, images, chunk_size)
                    with torch.cuda.stream(s2):
                        feat_conv_raw = chunked_backbone_forward(raw_model.convnext, images, chunk_size)
                    
                    raw_model.orchestrator.synchronize()

                # Fused Slotted Projection (Optimized Blackwell Path)
                linear = raw_model.proj_grouped[0]
                ln     = raw_model.proj_grouped[1]
                fused_proj = plantclef_ext.fused_projection(
                    feat_bio_raw.float(), feat_dino_raw.float(), feat_conv_raw.float(),
                    linear.weight.float(), linear.bias.float(), 
                    ln.weight.float(), ln.bias.float()
                )
                # Convert back to autocast dtype for the rest of the forward pass
                fused_proj = fused_proj.to(feat_bio_raw.dtype)
            else:
                # Standard sequential fallback
                with backbone_ctx:
                    feat_bio_raw  = chunked_backbone_forward(raw_model.bioclip,  images, chunk_size)
                    feat_dino_raw = chunked_backbone_forward(raw_model.dinov2,   images, chunk_size)
                    feat_conv_raw = chunked_backbone_forward(raw_model.convnext, images, chunk_size)

                fused_raw  = torch.cat([feat_bio_raw, feat_dino_raw, feat_conv_raw], dim=1)
                fused_proj = raw_model.proj_grouped(fused_raw)
            
            # Slice for KD distillation (each backbone still aligns to BioCLIP)
            # BioCLIP projected is 0:512, DINO is 512:1024, ConvNeXt is 1024:1536
            feat_bio  = F.normalize(fused_proj[:, 0:512],    dim=1)
            feat_dino = F.normalize(fused_proj[:, 512:1024], dim=1)
            feat_conv = F.normalize(fused_proj[:, 1024:1536],dim=1)

            feat_bio_teacher = feat_bio.detach()
            outputs   = raw_model.classifier(torch.cat([feat_bio, feat_dino, feat_conv], dim=1))
            
            # Handle Asymmetric Loss (requires one-hot targets)
            from .losses import AsymmetricLoss
            if isinstance(criterion, AsymmetricLoss):
                targets = F.one_hot(labels, num_classes=num_classes).float()
                hard_loss = criterion(outputs, targets)
            else:
                hard_loss = criterion(outputs, labels)

            kd_loss_dino = (1 - F.cosine_similarity(feat_dino, feat_bio_teacher)).mean()
            kd_loss_conv = (1 - F.cosine_similarity(feat_conv, feat_bio_teacher)).mean()
            kd_loss      = (kd_loss_dino + kd_loss_conv) / 2

            loss = (1 - KD_ALPHA) * hard_loss + KD_ALPHA * kd_loss

        if is_deepspeed:
            model.backward(loss)
            model.step()
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        _, predicted = outputs.max(1)
        acc_m.update(predicted, labels)
        running_loss    += loss.item()
        running_kd_loss += kd_loss.item()

        if i % 10 == 0:
            # Correct average for resumed runs
            steps_this_session = i - start_step + 1
            pbar.set_postfix({
                "Loss":    f"{running_loss / steps_this_session:.4f}",
                "KD":      f"{running_kd_loss / steps_this_session:.4f}",
                "Acc":     f"{acc_m.compute().item()*100:.2f}%"
            })

        # Progress saving (every 5%)
        if (i + 1) % save_interval == 0 and (i + 1) < total_steps:
            save_progress_checkpoint(model, epoch, i + 1, total_steps)

    train_loader.reset()
    train_acc = acc_m.compute().item() * 100.0
    acc_m.reset()

    # Get LR for logging
    if is_deepspeed:
        current_lr = model.get_lr()[0]
    elif optimizer is not None:
        current_lr = optimizer.param_groups[0]['lr']
    else:
        current_lr = 0.0

    wandb.log({
        "epoch":      epoch,
        "train_acc":  train_acc,
        "lr":         current_lr,
        "kd_loss":    running_kd_loss / (i + 1),
    })
    print(f"[Epoch {epoch}] Train Acc: {train_acc:.2f}%  "
          f"KD Loss: {running_kd_loss/(i+1):.4f}")
    return train_acc
