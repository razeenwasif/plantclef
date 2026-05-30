"""
Architecture dispatcher.

Peeks at the checkpoint's state_dict keys and routes to the right model module:
  - 010 / shared_mlp head (model_010.BioCLIP25MultiTask)
  - i002 / per-head MLPs (model_i002.BioCLIP25MultiTask)
"""
from __future__ import annotations
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import model_010
import model_i002

BIOCLIP25_MODEL_NAME = model_010.BIOCLIP25_MODEL_NAME

logger = logging.getLogger(__name__)


def _detect_arch(state_dict_keys) -> str:
    keys = list(state_dict_keys)
    has_shared = any(k.startswith("shared_mlp.") for k in keys)
    has_species_mlp = any(k.startswith("species_mlp.") for k in keys)
    if has_species_mlp:
        return "i002"
    if has_shared:
        return "010"
    raise ValueError(
        "Cannot detect architecture: state_dict has neither shared_mlp.* nor species_mlp.* keys"
    )


def load_checkpoint_model(checkpoint_path: str, device: str = "cpu"):
    p = Path(checkpoint_path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")

    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]
    if all(k.startswith("module.") for k in state_dict):
        peek_keys = [k[len("module."):] for k in state_dict]
    else:
        peek_keys = list(state_dict)

    arch = _detect_arch(peek_keys)
    logger.info(f"Detected architecture: {arch}")

    if arch == "i002":
        return model_i002.load_checkpoint_model(checkpoint_path, device=device)
    return model_010.load_checkpoint_model(checkpoint_path, device=device)


def resize_visual_pos_embed(model, target_img_size: int, patch_size: int = 14) -> None:
    """In-place bicubic interpolation of OpenCLIP visual.positional_embedding so the
    model accepts target_img_size square inputs. Safe to call when sizes already match
    (no-op). Both BioCLIP25MultiTask variants wrap OpenCLIP's clip_model under
    `model.backbone`, with the visual transformer at `model.backbone.visual`."""
    visual = model.backbone.visual
    pos = visual.positional_embedding
    pos_dim_3 = pos.dim() == 3
    if pos_dim_3:
        pos = pos.squeeze(0)

    D = pos.shape[-1]
    cls = pos[:1]
    patches = pos[1:]
    old_grid = int(round(patches.shape[0] ** 0.5))
    new_grid = target_img_size // patch_size

    if old_grid == new_grid:
        logger.info(f"pos_embed grid already matches ({old_grid}×{old_grid}); no resize")
        return

    logger.info(f"Interpolating pos_embed: {old_grid}×{old_grid} → {new_grid}×{new_grid}")
    patches = patches.reshape(1, old_grid, old_grid, D).permute(0, 3, 1, 2).float()
    patches = F.interpolate(patches, size=(new_grid, new_grid), mode="bicubic", align_corners=False)
    patches = patches.permute(0, 2, 3, 1).reshape(-1, D)
    new_pos = torch.cat([cls.float(), patches], dim=0)
    if pos_dim_3:
        new_pos = new_pos.unsqueeze(0)

    new_pos = new_pos.to(dtype=visual.positional_embedding.dtype,
                          device=visual.positional_embedding.device)
    visual.positional_embedding = nn.Parameter(new_pos, requires_grad=False)
