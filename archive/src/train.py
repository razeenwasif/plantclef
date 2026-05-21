import os
import sys
# Ensure src/ is on the path so local packages (training/, models/, data/)
# take precedence over any same-named packages installed globally
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:256"
)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam
import wandb
import cudf
from tqdm import tqdm
from dotenv import load_dotenv
import gc

try:
    import plantclef_ext
except ImportError:
    plantclef_ext = None

load_dotenv()

from models.ensemble import PlantEnsemble
from data.dataloader import get_dali_loaders
from training import (
    LogitAdjustmentLoss, AsymmetricLoss, EarlyStopping,
    extract_and_cache_features, CachedFeatureDataset,
    validate, run_phase1_cached, run_epoch,
    load_phase1_checkpoint, save_phase1_checkpoint,
    load_phase1_heads_for_phase2,
    load_phase2_checkpoint, save_epoch_checkpoint, save_deepspeed_checkpoint,
    phase1_is_complete, fit_and_save, load_pca, apply_pca,
)
from config import (
    RAW_CSV, IMG_DIR, CLEANED_CSV, BATCH_SIZE, P2_BATCH_SIZE, P2_CHUNK_SIZE,
    RESOLUTION, MODE, BIOCLIP_NAME, DINOV2_NAME, CONVNEXT_NAME,
    ACCUMULATION_STEPS, EPOCHS_PHASE1, EPOCHS_PHASE2, P2_SAMPLES_PER_EPOCH,
    VAL_EVERY_N_EPOCHS, MAX_VAL_BATCHES, PATIENCE,
    LORA_R, LORA_ALPHA, LORA_DROPOUT,
    FEATURE_CACHE_PATH, P1_CKPT_PATH, P2_CKPT_DIR, P2_EPOCH_CKPT,
    PCA_TRANSFORM_PATH,
)

# ---------------------------------------------------------------------------
# Hardware flags
# ---------------------------------------------------------------------------
torch.backends.cudnn.benchmark           = True
torch.backends.cuda.matmul.allow_tf32    = True
torch.backends.cudnn.allow_tf32          = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_math_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

# ---------------------------------------------------------------------------
# DeepSpeed configs
# ---------------------------------------------------------------------------
# Blackwell Optimisation: Disable all CPU offloading to leverage 32GB VRAM
DS_CONFIG_P1 = {
    "zero_optimization": {
        "stage": 1,
        # Offload disabled for 5090 Blackwell efficiency
    },
    "bf16": {"enabled": True},
    "gradient_accumulation_steps": ACCUMULATION_STEPS,
    "train_micro_batch_size_per_gpu": BATCH_SIZE,
    "steps_per_print": 50,
    "wall_clock_breakdown": False,
    "distributed_backend": "nccl",
}

# Phase 2: no CPU offload -- LoRA makes optimizer states tiny, keep on GPU
DS_CONFIG_P2 = {
    "zero_optimization": {
        "stage": 1
    },
    "zero_allow_untested_optimizer": True,
    "bf16": {"enabled": True},
    "gradient_accumulation_steps": ACCUMULATION_STEPS,
    "train_micro_batch_size_per_gpu": P2_BATCH_SIZE,
    "steps_per_print": 50,
    "wall_clock_breakdown": False,
    "distributed_backend": "nccl",
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_pca_module(pca, device):
    """
    Convert a fitted sklearn IncrementalPCA into a torch.nn.Module
    for fast GPU-native transformation.
    """
    import torch.nn as nn
    n_components = pca.n_components_
    d_in = pca.components_.shape[1]
    
    layer = nn.Linear(d_in, n_components)
    layer.weight.data = torch.from_numpy(pca.components_).float()
    layer.bias.data   = torch.from_numpy(-pca.mean_ @ pca.components_.T).float()
    
    return layer.to(device)


def train():
    """
    Progressive 2-Stage Training:

    Phase 1 — Feature Caching + Head Warmup
      Auto-skipped if a complete Phase 1 checkpoint already exists.
      Backbones frozen; features extracted once (~1-1.5 hrs at 384px).
      Only projection heads + classifier train (~30-40 min for 10 epochs).

    Phase 2 — LoRA Fine-Tuning
      LoRA adapters on DINOv2 + ConvNeXt (~5M trainable params vs ~800M).
      BioCLIP fully frozen throughout.
      batch=256, 3 epochs -- target < 24 hours total.
    """
    wandb.init(
        project="plantclef-2026",
        name=f"ensemble-lora-{MODE.lower()}",
        config={
            "resolution":          RESOLUTION,
            "batch_size":          BATCH_SIZE,
            "p2_batch_size":       P2_BATCH_SIZE,
            "epochs_phase1":       EPOCHS_PHASE1,
            "epochs_phase2":       EPOCHS_PHASE2,
            "lora_r":              LORA_R,
            "lora_alpha":          LORA_ALPHA,
            "lora_dropout":        LORA_DROPOUT,
            "lr_phase1":           1e-3,
            "lr_phase2_lora":      1e-4,
            "lr_phase2_head":      2e-4,
            "patience":            PATIENCE,
            "bioclip_backbone":    BIOCLIP_NAME,
            "dinov2_backbone":     DINOV2_NAME,
            "convnext_backbone":   CONVNEXT_NAME,
            "phase2_strategy":     "LoRA(DINOv2+ConvNeXt)+FrozenBioCLIP",
        }
    )
    config = wandb.config
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    csv_path = CLEANED_CSV if os.path.exists(CLEANED_CSV) else RAW_CSV
    os.makedirs("models", exist_ok=True)

    # Load class counts for Logit Adjustment then free GPU RAM immediately
    df     = cudf.read_csv(csv_path, sep=';')
    counts = df['species_id'].value_counts().sort_index().to_arrow().to_pylist()
    del df
    torch.cuda.empty_cache()

    # Asymmetric Loss with Fused CUDA Kernel for multi-label long-tail
    num_classes = len(counts)
    logit_adj   = (1.0 * torch.log(torch.tensor(counts, dtype=torch.float32) / sum(counts))).to(DEVICE)
    criterion   = AsymmetricLoss(logit_adjustments=logit_adj, use_fused=True)

    # -----------------------------------------------------------------------
    # PHASE 1: Feature Caching + Head Warmup
    # Auto-skipped if Phase 1 already completed (checkpoint covers all epochs)
    # -----------------------------------------------------------------------
    skip_phase1 = phase1_is_complete()
    
    # Load model and apply Blackwell-native compilation
    model = PlantEnsemble(
        num_classes=num_classes, 
        input_res=RESOLUTION,
        bioclip_name=BIOCLIP_NAME,
        dinov2_name=DINOV2_NAME,
        convnext_name=CONVNEXT_NAME,
    ).to(DEVICE).to(memory_format=torch.channels_last)
    
    # Blackwell Optimisation: max-autotune uses Triton for specialized Blackwell kernels
    if getattr(config, 'USE_COMPILE', False):
        print("[Blackwell] Applying torch.compile(model, mode='max-autotune')...")
        model = torch.compile(model, mode="max-autotune")

    if skip_phase1:
        print(f"\n[Phase1] Complete checkpoint found -- skipping Phase 1.")
        # Need to initialize loaders to get total samples etc, though we skip P1
        _t_loader, _v_loader, _ = get_dali_loaders(csv_path, IMG_DIR, batch_size=BATCH_SIZE)
        model.set_grad_checkpointing(True)
        model.freeze_backbones()
        del _t_loader, _v_loader
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print("\n--- PHASE 1: Feature Caching + Head Warmup ---")
        train_loader, val_loader, _ = get_dali_loaders(
            csv_path, IMG_DIR, batch_size=BATCH_SIZE,
            resolution=RESOLUTION, sampling_mode='natural'
        )
        
        model.set_grad_checkpointing(True)
        model.freeze_backbones()

        # Load or extract feature cache
        if os.path.exists(FEATURE_CACHE_PATH):
            print(f"[Feature Cache] Loading from {FEATURE_CACHE_PATH}...")
            with tqdm(total=1, desc="Loading feature cache", unit="file") as pbar:
                cache = torch.load(FEATURE_CACHE_PATH, weights_only=False)
                pbar.update(1)
            
            # Blackwell: Load entire 22.5GB cache to GPU to eliminate PCIe bottlenecks
            # The 5090 has 32GB VRAM, plenty for the 22.5GB cache + model weights.
            if getattr(config, 'LOAD_CACHE_TO_GPU', False):
                print("[Blackwell] Moving feature cache to GPU memory...")
                for k in cache:
                    if torch.is_tensor(cache[k]):
                        cache[k] = cache[k].to(DEVICE, non_blocking=True)
        else:
            cache = extract_and_cache_features(model, train_loader, DEVICE, FEATURE_CACHE_PATH)

        # Fit PCA on cached features (or load existing transform)
        if 'features_pca' not in cache:
            if os.path.exists(PCA_TRANSFORM_PATH):
                print("[PCA] Loading existing transform...")
                pca   = load_pca(PCA_TRANSFORM_PATH)
                cache = apply_pca(cache, pca)
            else:
                print("[PCA] Fitting PCA on cached features...")
                cache, pca = fit_and_save(cache)
            torch.save(cache, FEATURE_CACHE_PATH)
            print("[PCA] Cache updated with PCA features.")
        else:
            # Need the PCA object to build the validation module
            pca = load_pca(PCA_TRANSFORM_PATH)

        # GPU-native PCA for validation
        pca_module = get_pca_module(pca, DEVICE)

        # Phase 1 trains only the phase1_head (linear probe on PCA features).
        # Much faster than training proj heads on raw features (~3x smaller input).
        # Phase 2 still uses full proj heads + ensemble (unaffected by this).
        phase1_params  = list(model.phase1_head.parameters())
        optimizer_p1   = optim.AdamW(phase1_params, lr=config.lr_phase1,
                                     weight_decay=0.05, fused=True)

        from training import CachedPCADataset
        cache_dataset      = CachedPCADataset(cache) if 'features_pca' in cache else CachedFeatureDataset(cache)
        steps_per_epoch_p1 = len(cache_dataset) // (BATCH_SIZE * 4 * ACCUMULATION_STEPS)
        total_steps_p1     = (steps_per_epoch_p1 + 5) * EPOCHS_PHASE1

        scheduler_p1 = OneCycleLR(optimizer_p1, max_lr=config.lr_phase1,
                                   total_steps=total_steps_p1, pct_start=0.3,
                                   anneal_strategy='cos')

        model_engine_p1, optimizer_p1, _, scheduler_p1 = deepspeed.initialize(
            model=model, optimizer=optimizer_p1,
            lr_scheduler=scheduler_p1, config=DS_CONFIG_P1,
        )

        best_val_acc   = 0.0
        start_epoch_p1, best_val_acc = load_phase1_checkpoint(model_engine_p1, DEVICE)

        for epoch in range(start_epoch_p1, EPOCHS_PHASE1):
            train_acc = run_phase1_cached(
                model_engine_p1, cache, criterion, epoch, num_classes, DEVICE
            )
            # Use pca_module during Phase 1 validation to evaluate the correct head
            metrics = validate(model_engine_p1.module, val_loader, criterion,
                               num_classes, DEVICE, pca_layer=pca_module)
            print(f"\n[Phase1 Epoch {epoch}] Train: {train_acc:.2f}%  "
                  f"Val: {metrics['acc']:.2f}%  F1: {metrics['f1']:.4f}")
            wandb.log({"epoch": epoch, "train_acc": train_acc,
                       **{f"val_{k}": v for k, v in metrics.items()}})

            if metrics['acc'] > best_val_acc:
                best_val_acc = metrics['acc']
            save_phase1_checkpoint(model_engine_p1, epoch, best_val_acc)

        # Free Phase 1 engine + data before Phase 2
        model = model_engine_p1.module
        del model_engine_p1, optimizer_p1, scheduler_p1
        del cache_dataset, cache, train_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # PHASE 2: LoRA Fine-Tuning
    # -----------------------------------------------------------------------
    print("\n--- PHASE 2: LoRA Fine-Tuning (Long-Tail Calibration) ---")

    train_loader, val_loader, _ = get_dali_loaders(
        csv_path, IMG_DIR, batch_size=P2_BATCH_SIZE,
        resolution=RESOLUTION, sampling_mode='natural',
        samples_per_epoch=P2_SAMPLES_PER_EPOCH
    )

    # FAST WARMUP: Train main classifier on cached features if needed
    fast_warmup_done = False
    
    # Check if we should skip warmup (if we are already deep into Phase 2)
    has_p2_ckpt = any(os.path.exists(os.path.join("models", f)) for f in os.listdir("models") 
                      if f.startswith("phase2_checkpoint_ep") and "_step_final" in f)
    
    if os.path.exists(FEATURE_CACHE_PATH) and not has_p2_ckpt:
        print("\n--- PHASE 2A: Fast Head Warmup (Using Cache) ---")
        raw_cache = torch.load(FEATURE_CACHE_PATH, weights_only=False)
        if 'features_pca' in raw_cache:
            pca_features = raw_cache.pop('features_pca')
            
            head_params = list(model.proj_grouped.parameters()) + list(model.classifier.parameters())
            warmup_opt  = optim.AdamW(head_params, lr=1e-3, weight_decay=0.01, fused=True)
            model_engine_warmup, _, _, _ = deepspeed.initialize(
                model=model, optimizer=warmup_opt, config=DS_CONFIG_P1
            )

            # --- CACHE VALIDATION FEATURES ---
            print("[Warmup] Pre-extracting validation features for instant metrics...")
            val_cache = {'bio': [], 'dino': [], 'conv': [], 'label': []}
            model.eval()
            with torch.no_grad():
                for i, v_data in enumerate(tqdm(val_loader, desc="Caching Val", leave=False)):
                    if i >= MAX_VAL_BATCHES: break
                    v_imgs = v_data[0]['data'].to(DEVICE, memory_format=torch.channels_last)
                    v_lbls = v_data[0]['label'].to(DEVICE).squeeze().long()

                    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                        # Use parallel streams if available
                        if hasattr(model, 'has_ext') and model.has_ext:
                            s0 = torch.cuda.ExternalStream(model.orchestrator.get_stream(0))
                            s1 = torch.cuda.ExternalStream(model.orchestrator.get_stream(1))
                            s2 = torch.cuda.ExternalStream(model.orchestrator.get_stream(2))
                            with torch.cuda.stream(s0): b = model.bioclip(v_imgs)
                            with torch.cuda.stream(s1): d = model.dinov2(v_imgs)
                            with torch.cuda.stream(s2): c = model.convnext(v_imgs)
                            model.orchestrator.synchronize()
                        else:
                            b, d, c = model.bioclip(v_imgs), model.dinov2(v_imgs), model.convnext(v_imgs)

                    val_cache['bio'].append(b.cpu()); val_cache['dino'].append(d.cpu())
                    val_cache['conv'].append(c.cpu()); val_cache['label'].append(v_lbls.cpu())

            val_cache = {k: torch.cat(v) for k, v in val_cache.items()}
            print(f"[Warmup] Cached {len(val_cache['label'])} validation samples.")

            # 50 epochs on cached features to fully initialize the main head
            # Added early stopping to skip if it plateaus early
            print("Starting Extended Fast Warmup (Target: 50 Epochs)...")
            best_w_acc = 0.0
            w_patience = 0
            for w_epoch in range(50):
                run_phase1_cached(model_engine_warmup, raw_cache, criterion, 
                                  f"Warmup-{w_epoch}", num_classes, DEVICE)

                # --- INSTANT VALIDATION ---
                model.eval()
                with torch.no_grad():
                    v_b, v_d, v_c = val_cache['bio'].to(DEVICE), val_cache['dino'].to(DEVICE), val_cache['conv'].to(DEVICE)
                    v_l = val_cache['label'].to(DEVICE)
                    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                        if hasattr(model, 'has_ext') and model.has_ext:
                            v_proj = plantclef_ext.fused_projection(
                                v_b.float(), v_d.float(), v_c.float(), 
                                model.proj_grouped[0].weight.float(), 
                                model.proj_grouped[0].bias.float(), 
                                model.proj_grouped[1].weight.float(), 
                                model.proj_grouped[1].bias.float()
                            )
                            v_proj = v_proj.to(torch.bfloat16)
                        else:
                            v_proj = model.proj_grouped(torch.cat([v_b, v_d, v_c], dim=1))
                        v_out = model.classifier(F.normalize(v_proj, dim=1))
                        v_acc = (v_out.argmax(1) == v_l).float().mean().item() * 100

                print(f"Warmup-{w_epoch} Val Acc: {v_acc:.2f}%")

                if v_acc > best_w_acc + 0.05:
                    best_w_acc = v_acc
                    w_patience = 0
                else:
                    w_patience += 1
                    if w_patience >= 3:
                        print(f"Warmup plateau detected at epoch {w_epoch}. Converged at {v_acc:.2f}%.")
                        break
            # Clean up warmup engine
            model = model_engine_warmup.module
            del model_engine_warmup, warmup_opt, val_cache

            raw_cache['features_pca'] = pca_features
            del raw_cache
            gc.collect()
            torch.cuda.empty_cache()
            fast_warmup_done = True
            print("Fast Warmup Complete. Classifier is now fully initialized.")
            
            # Validate after warmup to see real progress
            print("[Warmup] Running validation...")
            metrics = validate(model, val_loader, criterion, num_classes, DEVICE)
            print(f"Post-Warmup Accuracy: {metrics['acc']:.2f}%")

    # Note: load_phase1_heads_for_phase2 removed because it was overwriting 
    # the Fast Warmup with untrained random weights from the P1 probe checkpoint.

    # -----------------------------------------------------------------------
    # PHASE 2 - Part A: Head Warmup (Frozen Backbones)
    # -----------------------------------------------------------------------
    # Resume check BEFORE warmup
    resumed_epoch, _, resumed_step = load_phase2_checkpoint(model, DEVICE, tag="checkpoint_latest")
    
    if (resumed_epoch is None or resumed_epoch < EPOCHS_PHASE1) and not fast_warmup_done:
        print("\n--- PHASE 2A: Head Warmup (Frozen Backbones) ---")
        model.freeze_backbones()
        
        warmup_optimizer = optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=2e-4, weight_decay=0.01
        )
        
        # One warmup epoch (Epoch -1)
        start_step = resumed_step if (resumed_epoch == -1 or resumed_epoch == EPOCHS_PHASE1 - 1) else 0
        run_epoch(model, train_loader, criterion, -1, num_classes, DEVICE, 
                  optimizer=warmup_optimizer, start_step=start_step, 
                  p2_chunk_size=P2_BATCH_SIZE)
        print("Warmup complete. Classifier is now initialized.")
    else:
        print(f"\n[Phase2A] Skipping warmup (Already at epoch {resumed_epoch})")

    # -----------------------------------------------------------------------
    # PHASE 2 - Part B: LoRA Fine-Tuning
    # -----------------------------------------------------------------------
    print("\n--- PHASE 2B: LoRA Fine-Tuning (Long-Tail Calibration) ---")
    
    # Apply LoRA to DINOv2 + ConvNeXt; BioCLIP stays frozen
    model.apply_lora(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT)

    # Separate optimizer param groups: lower LR for LoRA, higher for heads
    lora_params = [p for name, p in model.named_parameters()
                   if 'lora_' in name and p.requires_grad]
    head_params = [p for name, p in model.named_parameters()
                   if 'lora_' not in name and p.requires_grad]

    # Freeze BioCLIP explicitly (LoRA was not applied to it)
    for param in model.bioclip.parameters():
        param.requires_grad = False

    # Phase 2 optimizer
    optimizer_p2 = optim.AdamW([
        {'params': lora_params, 'lr': config.lr_phase2_lora},
        {'params': head_params, 'lr': config.lr_phase2_head},
    ], weight_decay=0.01, fused=True)

    # Wrap with Lookahead to find flatter minima
    from timm.optim import Lookahead
    optimizer_p2 = Lookahead(optimizer_p2, alpha=0.5, k=6)

    steps_per_epoch_p2 = len(train_loader)
    total_steps_p2     = (steps_per_epoch_p2 + 5) * EPOCHS_PHASE2

    scheduler_p2 = OneCycleLR(
        optimizer_p2,
        max_lr=[config.lr_phase2_lora, config.lr_phase2_head],
        total_steps=total_steps_p2,
        pct_start=0.1,
        anneal_strategy='cos',
    )

    model_engine_p2, optimizer_p2, _, scheduler_p2 = deepspeed.initialize(
        model=model, optimizer=optimizer_p2,
        lr_scheduler=scheduler_p2, config=DS_CONFIG_P2,
    )

    early_stopping = EarlyStopping(patience=PATIENCE, min_delta=0.001)
    best_val_acc   = 0.0
    start_epoch    = EPOCHS_PHASE1
    final_epoch    = EPOCHS_PHASE1 + EPOCHS_PHASE2 - 1

    # Resume from checkpoint if one exists
    resumed_epoch, resumed_acc, resumed_step = load_phase2_checkpoint(model_engine_p2, DEVICE, tag="checkpoint_latest")
    if resumed_epoch is not None:
        # If the checkpoint is from the warmup (e.g. -1), we start at the 
        # actual Phase 2 start epoch (EPOCHS_PHASE1). 
        # If it's a real Phase 2 epoch (e.g. 10+), we resume there.
        start_epoch  = max(EPOCHS_PHASE1, resumed_epoch)
        best_val_acc = resumed_acc

    for epoch in range(start_epoch, EPOCHS_PHASE1 + EPOCHS_PHASE2):
        # Only use resumed_step for the very first epoch we resume into
        current_start_step = resumed_step if epoch == resumed_epoch else 0
        run_epoch(model_engine_p2, train_loader, criterion, epoch, num_classes, DEVICE, 
                  start_step=current_start_step)

        # Lightweight per-epoch checkpoint every epoch
        save_epoch_checkpoint(model_engine_p2, epoch, best_val_acc)

        if epoch % VAL_EVERY_N_EPOCHS == 0 or epoch == final_epoch:
            metrics = validate(
                model_engine_p2.module, val_loader, criterion, num_classes, DEVICE
            )
            val_acc = metrics['acc']

            print(f"\n[Epoch {epoch}] Val Loss: {metrics['loss']:.4f}  "
                  f"Acc: {metrics['acc']:.2f}%  F1: {metrics['f1']:.4f}  "
                  f"Prec: {metrics['precision']:.4f}  Rec: {metrics['recall']:.4f}")

            wandb.log({"epoch": epoch, **{f"val_{k}": v for k, v in metrics.items()}})

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_deepspeed_checkpoint(
                    model_engine_p2, "models/", "best_calibrated", epoch, best_val_acc
                )
                print(f"--> New Best (Acc: {val_acc:.2f}%)")

            save_deepspeed_checkpoint(
                model_engine_p2, P2_CKPT_DIR, "checkpoint_latest", epoch, best_val_acc
            )

            early_stopping(val_acc)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch}. Best: {best_val_acc:.2f}%")
                break

    save_deepspeed_checkpoint(
        model_engine_p2, "models/", "final", epoch, best_val_acc
    )
    wandb.finish()


if __name__ == "__main__":
    train()
