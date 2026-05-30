"""
DINO-style SSL continual pretraining of BioCLIP-2.5 ViT-H/14 on
geometry-corrected LUCAS pseudo-quadrats.

Why
---
EDA showed LUCAS *aspect-/scale-corrected* is the only unlabeled in-domain
proxy we have for the test distribution. 008 Phase B v3 self-distilled labels
on raw LUCAS and crashed to 0.227 because the aspect/scale mismatch made the
teacher's pseudo-labels noise. Here we *do not use labels* — we adapt the
encoder via DINO's self-distillation loss so its representation moves toward
quadrat-shaped imagery. The supervised head is fine-tuned afterwards by
`finetune_after_ssl.py` using the team-best 010 last_blocks recipe.

Design choices (deliberately conservative; this is a partial-SSL adapt, not
a from-scratch DINOv2 run):

* **Partial unfreeze.** Only the last `--unfreeze-blocks` ViT blocks +
  `ln_post` + `proj` train. Matches 010's `unfreeze_n=4` sweet spot.
  BioCLIP's earlier blocks carry the Tree-of-Life prior we don't want to lose.

* **Multicrop, all at 224 px.** 2 globals + N locals, all at the encoder's
  native input size. Avoids interpolating ViT-H/14 patch position embeddings
  (the model is pretrained at 14×14 patches → 16×16 grid at 224 px). Local
  vs global is encoded by RandomResizedCrop scale ranges, not by image size.
  This is a deviation from canonical DINO (which uses 96 px locals) but is
  the safe choice for a HuggingFace pretrained ViT-H.

* **EMA teacher** with cosine momentum schedule 0.996 → 1.0.
* **Centering** on teacher logits with EMA buffer (no batch-norm on out).
* **bf16** mixed precision; fp32 master weights via param's own dtype.
* **Gradient checkpointing** on the student's unfrozen ViT blocks.

Run
---
    torchrun --nproc-per-node=1 ssl_train.py \\
        --data-dir /workspace/plantclef/processed/lucas_aspect_corrected \\
        --out-dir src_experiments/012_bioclip25_ssl_lucas/outputs \\
        --epochs 5 --batch-size 16 --grad-accum 4 \\
        --unfreeze-blocks 4 --num-locals 6

A 5-epoch run on ~212K LUCAS images at effective batch 64 takes ~6-10 hours
on a 5090 (single GPU). Output: `ssl_bioclip25_backbone_ep{N}.pt` containing
just the BioCLIP visual encoder state_dict (consumable by step 3).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image

LOG = logging.getLogger("ssl_train")


# ---------------------------------------------------------------------------
# OpenCLIP / BioCLIP normalization stats (matches src_experiments/006).
# ---------------------------------------------------------------------------
BIOCLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
BIOCLIP_STD  = (0.26862954, 0.26130258, 0.27577711)
BIOCLIP25_MODEL_NAME = "hf-hub:imageomics/bioclip-2.5-vith14"

IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


# ---------------------------------------------------------------------------
# Dataset: LUCAS images → DINO multicrop views
# ---------------------------------------------------------------------------

class MultiCropTransform:
    """
    Two global crops + N local crops, all at 224 px to match ViT-H/14's
    pretrained position embedding grid.

    Globals see 50-100% of the image; locals see 5-50%. Heavier augmentation
    on locals (ColorJitter, GaussianBlur) to match canonical DINO recipe.
    """

    def __init__(self, num_locals: int = 6, img_size: int = 224) -> None:
        self.num_locals = num_locals
        norm = T.Normalize(mean=BIOCLIP_MEAN, std=BIOCLIP_STD)

        flip_color = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(
                [T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8,
            ),
            T.RandomGrayscale(p=0.2),
        ])

        self.global_t = T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.5, 1.0),
                                interpolation=T.InterpolationMode.BICUBIC),
            flip_color,
            T.RandomApply([T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
            T.ToTensor(),
            norm,
        ])

        self.local_t = T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.05, 0.5),
                                interpolation=T.InterpolationMode.BICUBIC),
            flip_color,
            T.RandomApply([T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
            T.ToTensor(),
            norm,
        ])

    def __call__(self, img: Image.Image) -> list[torch.Tensor]:
        crops = [self.global_t(img), self.global_t(img)]
        crops += [self.local_t(img) for _ in range(self.num_locals)]
        return crops


class LucasDataset(Dataset):
    def __init__(self, root: Path, num_locals: int = 6) -> None:
        self.paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_SUFFIXES)
        if not self.paths:
            raise RuntimeError(f"No images under {root}")
        self.tfm = MultiCropTransform(num_locals=num_locals)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> list[torch.Tensor]:
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # Defensive: skip a corrupt file by returning a deterministic neighbour.
            img = Image.open(self.paths[(idx + 1) % len(self.paths)]).convert("RGB")
        return self.tfm(img)


def multicrop_collate(batch: list[list[torch.Tensor]]) -> list[torch.Tensor]:
    """
    Re-arrange list-of-lists so each crop index is its own batched tensor.
    Returns ``crops_per_view``, where ``crops_per_view[i]`` is shape
    ``(B, 3, 224, 224)`` for view i across the whole batch.
    """
    n_views = len(batch[0])
    return [torch.stack([sample[i] for sample in batch], dim=0) for i in range(n_views)]


# ---------------------------------------------------------------------------
# Model: BioCLIP visual encoder + DINO projection head
# ---------------------------------------------------------------------------

class DINOHead(nn.Module):
    """
    DINO projection head: 3-layer MLP → L2 norm → weight-normed final layer.
    Output dim 65536 from the original DINO paper; that is overkill for
    partial-SSL adapt, but it costs ~33 M params (independent of backbone) so
    we keep it for fidelity to the published recipe.
    """
    def __init__(self, in_dim: int, out_dim: int = 65536,
                 hidden_dim: int = 2048, bottleneck_dim: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)
        self.last_layer.weight_g.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last_layer(x)


class BioCLIPVisual(nn.Module):
    """
    Wraps OpenCLIP BioCLIP-2.5 image encoder and exposes a clean (B, embed_dim)
    forward. Selectively unfreezes the last `n` ViT blocks + ln_post + proj.

    All-frozen blocks remain in eval() to disable any dropout / silent
    train-time path; the unfrozen blocks behave normally.
    """
    def __init__(self, model_name: str = BIOCLIP25_MODEL_NAME,
                 unfreeze_blocks: int = 4) -> None:
        super().__init__()
        import open_clip
        clip_model, _, _ = open_clip.create_model_and_transforms(model_name)
        self.backbone = clip_model
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        embed_dim = self._probe_embed_dim()
        self.embed_dim = embed_dim
        LOG.info(f"BioCLIP-2.5 loaded, embed_dim={embed_dim}")

        if unfreeze_blocks > 0:
            visual = self.backbone.visual
            resblocks = visual.transformer.resblocks
            n_total = len(resblocks)
            n_unfreeze = min(unfreeze_blocks, n_total)
            for i, block in enumerate(resblocks):
                if i >= n_total - n_unfreeze:
                    for p in block.parameters():
                        p.requires_grad_(True)
            for attr in ("ln_post", "proj"):
                obj = getattr(visual, attr, None)
                if obj is None:
                    continue
                if isinstance(obj, nn.Parameter):
                    obj.requires_grad_(True)
                elif isinstance(obj, nn.Module):
                    for p in obj.parameters():
                        p.requires_grad_(True)
            n_train = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
            LOG.info(f"Unfroze last {n_unfreeze}/{n_total} blocks + ln_post/proj  → {n_train:,} params")

    def _probe_embed_dim(self) -> int:
        device = next(self.backbone.parameters()).device
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=device)
            try:
                feat = self.backbone.encode_image(dummy, normalize=False)
            except TypeError:
                feat = self.backbone.encode_image(dummy)
        return feat.shape[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            return self.backbone.encode_image(x, normalize=False)
        except TypeError:
            return self.backbone.encode_image(x)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        if mode:
            for module in self.backbone.modules():
                own = any(p.requires_grad for p in module.parameters(recurse=False))
                if own:
                    module.train(True)
        return self


# ---------------------------------------------------------------------------
# DINO loss with centering
# ---------------------------------------------------------------------------

class DINOLoss(nn.Module):
    """
    Cross-entropy of student logits against teacher logits, with EMA centering
    on the teacher and temperature warmup.

    For each (student_view, teacher_view) pair where student_view ≠ teacher_view,
    minimise CE(softmax((teacher - center) / temp_t), softmax(student / temp_s)).
    """
    def __init__(self, out_dim: int, n_global_views: int = 2,
                 student_temp: float = 0.1,
                 teacher_temp_start: float = 0.04,
                 teacher_temp_end: float = 0.04,
                 warmup_epochs: int = 0,
                 total_epochs: int = 5,
                 center_momentum: float = 0.9) -> None:
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.n_global_views = n_global_views
        self.register_buffer("center", torch.zeros(1, out_dim))
        # piecewise linear teacher temp warmup
        ramp = np.linspace(teacher_temp_start, teacher_temp_end, max(1, warmup_epochs))
        flat = np.full(max(0, total_epochs - warmup_epochs), teacher_temp_end)
        self.teacher_temp_schedule = np.concatenate([ramp, flat]).astype(np.float32)

    def teacher_temp(self, epoch: int) -> float:
        idx = min(epoch, len(self.teacher_temp_schedule) - 1)
        return float(self.teacher_temp_schedule[idx])

    def forward(self, student_out: torch.Tensor, teacher_out: torch.Tensor,
                epoch: int) -> torch.Tensor:
        """
        student_out : (n_views_total * B, out_dim)
        teacher_out : (n_global_views * B, out_dim)  — only globals go through teacher
        """
        s = student_out / self.student_temp
        s_chunks = s.chunk(student_out.shape[0] // (teacher_out.shape[0] // self.n_global_views))

        teacher_temp = self.teacher_temp(epoch)
        t = F.softmax((teacher_out - self.center) / teacher_temp, dim=-1).detach()
        t_chunks = t.chunk(self.n_global_views)

        total_loss = torch.zeros((), device=student_out.device, dtype=student_out.dtype)
        n_terms = 0
        for tg, t_view in enumerate(t_chunks):
            for sv, s_view in enumerate(s_chunks):
                if sv == tg:
                    continue
                loss = -(t_view * F.log_softmax(s_view, dim=-1)).sum(dim=-1).mean()
                total_loss = total_loss + loss
                n_terms += 1
        total_loss = total_loss / max(1, n_terms)

        # EMA-update centering buffer
        with torch.no_grad():
            batch_center = teacher_out.mean(dim=0, keepdim=True)
            self.center.mul_(self.center_momentum).add_(batch_center, alpha=1 - self.center_momentum)

        return total_loss


# ---------------------------------------------------------------------------
# EMA helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def ema_update(student: nn.Module, teacher: nn.Module, momentum: float) -> None:
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(momentum).add_(ps.data, alpha=1 - momentum)


def cosine_schedule(base: float, final: float, total_steps: int) -> np.ndarray:
    steps = np.arange(total_steps)
    return final + 0.5 * (base - final) * (1 + np.cos(np.pi * steps / max(1, total_steps - 1)))


# ---------------------------------------------------------------------------
# Argument parsing & main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True,
                   help="Geometry-corrected LUCAS directory from prepare_lucas.py.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--model-name", default=BIOCLIP25_MODEL_NAME)
    p.add_argument("--unfreeze-blocks", type=int, default=4)
    p.add_argument("--num-locals", type=int, default=6)

    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16,
                   help="Micro-batch (samples per fwd). Effective = micro * grad_accum.")
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=8)

    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lr-final", type=float, default=1e-6)
    p.add_argument("--warmup-epochs", type=int, default=1)
    p.add_argument("--weight-decay", type=float, default=0.04)

    p.add_argument("--out-dim", type=int, default=65536, help="DINO head output dim.")
    p.add_argument("--student-temp", type=float, default=0.1)
    p.add_argument("--teacher-temp", type=float, default=0.04)
    p.add_argument("--ema-momentum-start", type=float, default=0.996)
    p.add_argument("--ema-momentum-end", type=float, default=1.0)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-every", type=int, default=1, help="epochs")
    p.add_argument("--clip-grad", type=float, default=3.0)
    p.add_argument("--log-every", type=int, default=20)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "ssl_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOG.info(f"Device: {device}")

    # ----- data -----
    ds = LucasDataset(Path(args.data_dir), num_locals=args.num_locals)
    LOG.info(f"LUCAS: {len(ds):,} images, num_locals={args.num_locals}")
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=multicrop_collate,
        persistent_workers=args.num_workers > 0,
    )

    # ----- model: student + teacher -----
    student_backbone = BioCLIPVisual(args.model_name, unfreeze_blocks=args.unfreeze_blocks).to(device)
    teacher_backbone = BioCLIPVisual(args.model_name, unfreeze_blocks=0).to(device)

    student_head = DINOHead(student_backbone.embed_dim, out_dim=args.out_dim).to(device)
    teacher_head = DINOHead(teacher_backbone.embed_dim, out_dim=args.out_dim).to(device)

    # initialise teacher = student
    teacher_backbone.load_state_dict(student_backbone.state_dict())
    teacher_head.load_state_dict(student_head.state_dict())
    for p in teacher_backbone.parameters():
        p.requires_grad_(False)
    for p in teacher_head.parameters():
        p.requires_grad_(False)

    # gradient checkpointing on the unfrozen ViT blocks (memory)
    try:
        student_backbone.backbone.visual.transformer.grad_checkpointing = True
        LOG.info("Enabled gradient checkpointing on student ViT.")
    except Exception as e:
        LOG.warning(f"Could not toggle grad-checkpointing: {e}")

    n_global = 2
    n_total = n_global + args.num_locals
    LOG.info(f"Multicrop: {n_global} globals + {args.num_locals} locals = {n_total} views/sample")

    # ----- loss / optim -----
    loss_fn = DINOLoss(
        out_dim=args.out_dim,
        n_global_views=n_global,
        student_temp=args.student_temp,
        teacher_temp_start=args.teacher_temp,
        teacher_temp_end=args.teacher_temp,
        warmup_epochs=0,
        total_epochs=args.epochs,
    ).to(device)

    trainable = (
        [p for p in student_backbone.parameters() if p.requires_grad]
        + list(student_head.parameters())
    )
    n_train_params = sum(p.numel() for p in trainable)
    LOG.info(f"Trainable params: {n_train_params:,}")

    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = len(loader) // args.grad_accum
    total_optim_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs
    lr_warmup = np.linspace(args.lr * 1e-3, args.lr, max(1, warmup_steps))
    lr_cos = cosine_schedule(args.lr, args.lr_final, max(1, total_optim_steps - warmup_steps))
    lr_schedule = np.concatenate([lr_warmup, lr_cos])

    momentum_schedule = cosine_schedule(args.ema_momentum_start, args.ema_momentum_end, total_optim_steps)

    # ----- training loop -----
    student_backbone.train()
    student_head.train()
    teacher_backbone.eval()
    teacher_head.eval()

    optim.zero_grad(set_to_none=True)
    optim_step = 0
    t_start = time.time()

    for epoch in range(args.epochs):
        running = 0.0
        n_micro = 0

        for it, crops in enumerate(loader):
            crops = [c.to(device, non_blocking=True) for c in crops]
            globals_ = torch.cat(crops[:n_global], dim=0)        # (n_global*B, 3, H, W)
            all_views = torch.cat(crops, dim=0)                  # (n_total*B, 3, H, W)

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                t_feat = teacher_backbone(globals_)
                t_out = teacher_head(t_feat).float()

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                s_feat = student_backbone(all_views)
                s_out = student_head(s_feat).float()

                loss = loss_fn(s_out, t_out, epoch=epoch)
                loss = loss / args.grad_accum

            loss.backward()
            running += loss.item() * args.grad_accum
            n_micro += 1

            if (it + 1) % args.grad_accum == 0:
                lr = float(lr_schedule[min(optim_step, len(lr_schedule) - 1)])
                for g in optim.param_groups:
                    g["lr"] = lr

                if args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(trainable, args.clip_grad)

                optim.step()
                optim.zero_grad(set_to_none=True)

                m = float(momentum_schedule[min(optim_step, len(momentum_schedule) - 1)])
                ema_update(student_backbone, teacher_backbone, m)
                ema_update(student_head, teacher_head, m)
                optim_step += 1

                if optim_step % args.log_every == 0:
                    elapsed = time.time() - t_start
                    LOG.info(
                        f"ep={epoch} step={optim_step}/{total_optim_steps}  "
                        f"loss={running / n_micro:.4f}  lr={lr:.2e}  ema_m={m:.4f}  "
                        f"elapsed={elapsed/60:.1f}m"
                    )
                    running = 0.0
                    n_micro = 0

        # ----- save -----
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            ckpt_path = out_dir / f"ssl_bioclip25_backbone_ep{epoch+1}.pt"
            torch.save({
                "epoch": epoch + 1,
                "model_name": args.model_name,
                # save the FULL backbone state_dict (student is the live model);
                # downstream finetune_after_ssl.py loads this and replaces the
                # OpenCLIP weights before unfreezing for supervised training.
                "backbone_state_dict": student_backbone.backbone.state_dict(),
                "ssl_args": vars(args),
            }, ckpt_path)
            LOG.info(f"Saved {ckpt_path}")

    LOG.info("SSL pretraining done.")


if __name__ == "__main__":
    main()
