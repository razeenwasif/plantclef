"""
PlantNet 2024 DINOv2 backbone + multi-label classifier head for PlantCLEF 2026.

Backbone
--------
`vit_base_patch14_reg4_dinov2.lvd142m` (timm), initialised from the official
PlantCLEF 2024 checkpoint `vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all`.
The PC24 checkpoint was fine-tuned (backbone + head) on PlantNet 2024 single-plant
data for 7806 species. 7804 of those species are identical to PC26's 7806 set —
so we remap the pretrained head row-wise into our sorted-species ordering and
drop the 2 dead classes. This gives us a classifier head that already carries
per-species signal, rather than a fresh random head.

Architecture
------------
timm ViT-B/14 reg4 backbone
    -> global pool over patch tokens (timm default) -> 768-d
    -> Linear(768, 7806)  [weights init'd from PC24 head, remapped]

At inference we apply sigmoid (not softmax) so the same logits serve multi-label.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


DEFAULT_TIMM_NAME = "vit_base_patch14_reg4_dinov2.lvd142m"
DEFAULT_EMBED_DIM = 768


# ---------------------------------------------------------------------------
# Head remapping
# ---------------------------------------------------------------------------

def load_pc24_class_mapping(path: str | Path) -> list[str]:
    """Read PlantCLEF 2024 class_mapping.txt (one species_id per line)."""
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def build_head_remap(
    pc24_class_list: list[str],
    target_species_ids: list[str],
) -> tuple[torch.Tensor, list[int]]:
    """
    Return (row-index tensor, list of missing indices into target_species_ids).

    row[j] = i such that pc24_class_list[i] == target_species_ids[j], or -1 if
    the target species is absent from PC24.
    """
    lut = {sid: i for i, sid in enumerate(pc24_class_list)}
    rows: list[int] = []
    missing: list[int] = []
    for j, sid in enumerate(target_species_ids):
        i = lut.get(str(sid), -1)
        rows.append(i)
        if i < 0:
            missing.append(j)
    return torch.tensor(rows, dtype=torch.long), missing


# ---------------------------------------------------------------------------
# Backbone + head
# ---------------------------------------------------------------------------

class PlantNetDINOv2MultiLabel(nn.Module):
    """
    PlantNet 2024 DINOv2 ViT-B/14 + single-Linear multi-label head.

    Parameters
    ----------
    n_classes         : 7806 for PlantCLEF 2026.
    pc24_checkpoint   : Path to PlantNet `model_best.pth.tar`. Optional — if
                        omitted we start from timm's lvd142m SSL weights with a
                        fresh head.
    pc24_class_file   : Path to PlantNet `class_mapping.txt`.
    target_species_ids: Sorted species IDs matching our inference pipeline. Used
                        to remap the PC24 head rows into our ordering.
    img_size          : Input resolution. Must be a multiple of 14.
    use_ema           : Prefer `state_dict_ema` from the PC24 checkpoint.
    """

    def __init__(
        self,
        n_classes: int,
        pc24_checkpoint: Optional[str | Path] = None,
        pc24_class_file: Optional[str | Path] = None,
        target_species_ids: Optional[list[str]] = None,
        img_size: int = 336,
        use_ema: bool = True,
        backbone_name: str = DEFAULT_TIMM_NAME,
    ) -> None:
        super().__init__()
        import timm

        if img_size % 14 != 0:
            raise ValueError(f"img_size must be a multiple of 14 (got {img_size}).")

        self.backbone_name = backbone_name
        self.n_classes = n_classes
        self.img_size = img_size

        # We use timm's native classifier interface (ViT's built-in Linear head).
        # Creating with num_classes=n_classes lets us later load PC24 weights
        # with a remap; the backbone weights still load from the checkpoint.
        self.model = timm.create_model(
            backbone_name,
            pretrained=(pc24_checkpoint is None),
            num_classes=n_classes,
            img_size=img_size,
        )
        self.embed_dim = getattr(self.model, "num_features", DEFAULT_EMBED_DIM)

        if pc24_checkpoint is not None:
            self._load_pc24(
                Path(pc24_checkpoint),
                Path(pc24_class_file) if pc24_class_file else None,
                target_species_ids,
                use_ema=use_ema,
            )

        logger.info(
            f"PlantNetDINOv2MultiLabel: {backbone_name}, "
            f"n_classes={n_classes}, img_size={img_size}, "
            f"embed_dim={self.embed_dim}"
        )

    # ------------------------------------------------------------------

    def _load_pc24(
        self,
        ckpt_path: Path,
        class_file: Optional[Path],
        target_species_ids: Optional[list[str]],
        use_ema: bool,
    ) -> None:
        if not ckpt_path.exists():
            raise FileNotFoundError(f"PC24 checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = None
        if isinstance(ckpt, dict):
            if use_ema and "state_dict_ema" in ckpt:
                sd = ckpt["state_dict_ema"]
                logger.info("  using state_dict_ema from PC24 checkpoint.")
            elif "state_dict" in ckpt:
                sd = ckpt["state_dict"]
            else:
                sd = ckpt
        else:
            sd = ckpt

        # timm serializes EMA weights with a "module." prefix sometimes — strip.
        if any(k.startswith("module.") for k in sd.keys()):
            sd = {k[len("module."):]: v for k, v in sd.items()}

        # Resize pos_embed to the current img_size if needed. PC24 trained at
        # 518×518 (1369 patches + 1 CLS + 4 reg = 1374). We may want 336×336
        # (576 patches). timm ships a utility for this.
        target_pos = self.model.pos_embed
        src_pos = sd.get("pos_embed")
        if src_pos is not None and src_pos.shape != target_pos.shape:
            from timm.layers.pos_embed import resample_abs_pos_embed
            num_prefix = target_pos.shape[1] - (self.img_size // 14) ** 2
            src_prefix_tokens = src_pos.shape[1] - int(
                ((src_pos.shape[1] - num_prefix) ** 0.5 + 0.5)
            ) ** 2
            # Both models have the same prefix count (1 CLS + 4 reg = 5).
            new_size = (self.img_size // 14, self.img_size // 14)
            resized = resample_abs_pos_embed(
                src_pos,
                new_size=new_size,
                num_prefix_tokens=num_prefix,
                interpolation="bicubic",
                antialias=True,
            )
            logger.info(
                f"  resized pos_embed: {tuple(src_pos.shape)} -> {tuple(resized.shape)} "
                f"(num_prefix={num_prefix})"
            )
            sd["pos_embed"] = resized

        # Head row remap (PC24 → target_species_ids)
        remapped_head_w = None
        remapped_head_b = None
        if class_file and target_species_ids is not None:
            pc24_classes = load_pc24_class_mapping(class_file)
            rows, missing = build_head_remap(pc24_classes, target_species_ids)
            head_w = sd.get("head.weight")
            head_b = sd.get("head.bias")
            if head_w is None or head_b is None:
                logger.warning("  PC24 ckpt missing head.weight/bias; leaving head randomly initialised.")
            else:
                # Clamp -1 indices to 0 then zero-out those rows
                safe_rows = rows.clamp(min=0)
                remapped_head_w = head_w[safe_rows].clone()
                remapped_head_b = head_b[safe_rows].clone()
                for j in missing:
                    remapped_head_w[j].zero_()
                    remapped_head_b[j].zero_()
                logger.info(
                    f"  head remapped: {len(target_species_ids) - len(missing)}/{len(target_species_ids)} "
                    f"rows copied from PC24, {len(missing)} random-init."
                )
        else:
            logger.info("  no class_file/target_species_ids — head loads as-is (requires same ordering).")

        # Drop the PC24 head (shape mismatch after remap); load the rest strict=False.
        sd_no_head = {k: v for k, v in sd.items() if not k.startswith("head.")}
        result = self.model.load_state_dict(sd_no_head, strict=False)
        logger.info(
            f"  backbone load: missing={len(result.missing_keys)}, "
            f"unexpected={len(result.unexpected_keys)}"
        )
        if result.unexpected_keys[:5]:
            logger.info(f"    unexpected sample: {result.unexpected_keys[:5]}")

        # Apply remapped head weights.
        if remapped_head_w is not None:
            with torch.no_grad():
                self.model.get_classifier().weight.copy_(remapped_head_w)
                self.model.get_classifier().bias.copy_(remapped_head_b)
        elif "head.weight" in sd and sd["head.weight"].shape == self.model.get_classifier().weight.shape:
            with torch.no_grad():
                self.model.get_classifier().weight.copy_(sd["head.weight"])
                self.model.get_classifier().bias.copy_(sd["head.bias"])
            logger.info("  head loaded as-is (shapes matched, ordering assumed identical).")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Raw logits (B, n_classes). Apply sigmoid downstream."""
        return self.model(pixel_values)

    def forward_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Backbone pooled features (B, embed_dim). Useful for caching."""
        feats = self.model.forward_features(pixel_values)
        # Match timm's ViT head pooling: global avg over non-prefix tokens.
        return self.model.forward_head(feats, pre_logits=True)

    # ------------------------------------------------------------------
    # Parameter groups / freeze helpers
    # ------------------------------------------------------------------

    def head_parameters(self):
        return self.model.get_classifier().parameters()

    def backbone_parameters(self):
        head = self.model.get_classifier()
        head_ids = {id(p) for p in head.parameters()}
        for p in self.model.parameters():
            if id(p) not in head_ids:
                yield p

    def freeze_backbone(self) -> None:
        head_ids = {id(p) for p in self.model.get_classifier().parameters()}
        for p in self.model.parameters():
            if id(p) not in head_ids:
                p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = True

    # ------------------------------------------------------------------
    # LoRA
    # ------------------------------------------------------------------

    def apply_lora(
        self,
        r: int = 32,
        alpha: int = 64,
        dropout: float = 0.05,
        target_modules: Optional[list[str]] = None,
    ) -> None:
        """
        Wrap attention + MLP projections with LoRA via PEFT. We scope LoRA to
        `self.model` (the timm ViT), NOT the head — the head is small enough to
        train fully.
        """
        from peft import LoraConfig, get_peft_model

        if target_modules is None:
            target_modules = ["qkv", "proj", "fc1", "fc2"]

        if hasattr(self.model, "set_grad_checkpointing"):
            try:
                self.model.set_grad_checkpointing(False)
            except Exception:
                pass

        # We want the classifier (head) to stay fully trainable and outside the
        # LoRA wrap. PEFT wraps the whole module passed in and freezes non-target
        # params; to keep the head trainable we detach it first, wrap the rest,
        # then re-attach.
        head = self.model.get_classifier()
        # Make head trainable before freezing via PEFT (PEFT freezes non-target
        # params inside the wrapped module; we'll unfreeze the head after).
        config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            bias="none",
            modules_to_save=["head"],  # PEFT keeps these trainable + saves them
        )
        self.model = get_peft_model(self.model, config)
        logger.info(
            f"LoRA applied: r={r}, alpha={alpha}, dropout={dropout}, "
            f"targets={target_modules}, head kept trainable via modules_to_save."
        )


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------

def build_default_transform(img_size: int = 336):
    """ImageNet normalisation; center-crop at img_size after resize."""
    from torchvision import transforms

    if img_size % 14 != 0:
        raise ValueError(f"img_size must be a multiple of 14 (got {img_size}).")

    return transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def build_train_transform(img_size: int = 336):
    """Training transform: RandomResizedCrop + flip, ImageNet normalisation."""
    from torchvision import transforms

    if img_size % 14 != 0:
        raise ValueError(f"img_size must be a multiple of 14 (got {img_size}).")

    return transforms.Compose([
        transforms.RandomResizedCrop(
            img_size,
            scale=(0.6, 1.0),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
