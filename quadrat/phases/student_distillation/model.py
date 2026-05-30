"""Phase 2b Student ensemble — owned exclusively by student_distillation.

DO NOT modify this file for teacher or inference purposes.
Duplicate into the relevant phase's model.py instead.

Differences from TeacherEnsemble:
  - Default LoRA r=16 (lighter)
  - curriculum starts at 224px
  - includes warmup_head / phase1_head for legacy checkpoint compatibility
"""
from __future__ import annotations
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union


class StudentEnsemble(nn.Module):
    """Triple-backbone ensemble for student fine-tuning.

    Backbones are lazily loaded to keep VRAM budget flexible.
    LoRA is applied via apply_lora() before DeepSpeed initialisation.
    """

    BIOCLIP_DIM  = 768
    DINOV3_DIM   = 1024
    CONVNEXT_DIM = 1536
    PROJ_DIM     = 512

    def __init__(self, num_classes: int = 7808, input_res: int = 224,
                 bioclip_name:  str = "hf-hub:imageomics/bioclip-2",
                 dinov3_name:   str = "vit_large_patch16_dinov3.lvd1689m",
                 convnext_name: str = "convnextv2_large.fcmae_ft_in22k_in1k_384") -> None:
        super().__init__()
        self.num_classes   = num_classes
        self.input_res     = input_res
        self.bioclip_name  = bioclip_name
        self.dinov3_name   = dinov3_name
        self.convnext_name = convnext_name

        total = self.BIOCLIP_DIM + self.DINOV3_DIM + self.CONVNEXT_DIM  # 3328
        fused = self.PROJ_DIM * 3                                         # 1536

        self.bioclip:  nn.Module | None = None
        self.dinov3:   nn.Module | None = None
        self.convnext: nn.Module | None = None

        self.proj_linear   = nn.Linear(total, fused)
        self.proj_ln       = nn.LayerNorm(fused)
        self.proj_grouped  = nn.Sequential(self.proj_linear, self.proj_ln)
        self.gating_network = nn.Sequential(
            nn.Linear(total, 128), nn.LayerNorm(128), nn.GELU(), nn.Linear(128, 3)
        )
        self.agg_weights = nn.Parameter(torch.ones(fused))
        self.agg_bias    = nn.Parameter(torch.zeros(fused))

        from src.models.layers.gcn import EcologicalGCNHead
        self.species_classifier = EcologicalGCNHead(
            num_classes=num_classes, trait_dim=19, image_feat_dim=fused
        )
        # Warmup / phase-1 heads kept for checkpoint compatibility
        self.phase1_head = _ResidualMLP(1024, 2048, num_classes)
        self.warmup_head = _ResidualMLP(total, 2048, num_classes)

        self._lora_applied     = False
        self._backbones_frozen = False
        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        for m in [self.proj_linear, self.gating_network]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Sequential):
                for layer in m:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        if layer.bias is not None: nn.init.zeros_(layer.bias)

    # ------------------------------------------------------------------
    def ensure_backbones_loaded(self) -> None:
        if self.bioclip is not None:
            return
        from src.models.bioclip      import PlantBioCLIP
        from src.models.vit_backbone import PlantViTBackbone
        from src.models.convnext     import PlantConvNeXt
        rank = int(os.environ.get("RANK", 0))
        print(f"[Student] Rank {rank}: loading backbones...")
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bioclip  = PlantBioCLIP(checkpoint=self.bioclip_name,  input_res=self.input_res).to(device)
        self.dinov3   = PlantViTBackbone(model_name=self.dinov3_name,   input_res=self.input_res).to(device)
        self.convnext = PlantConvNeXt(model_name=self.convnext_name, input_res=self.input_res).to(device)
        for bb in (self.bioclip, self.dinov3, self.convnext):
            bb.train(self.training)

    # ------------------------------------------------------------------
    def set_resolution(self, res: int) -> None:
        self.input_res = res
        for bb in (self.bioclip, self.dinov3, self.convnext):
            if bb is not None and hasattr(bb, "set_resolution"):
                bb.set_resolution(res)

    # ------------------------------------------------------------------
    def apply_lora(self, r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05) -> None:
        self.ensure_backbones_loaded()
        if self._lora_applied:
            return
        from src.models.layers.fused_lora import FusedLoRALinear
        rank = int(os.environ.get("RANK", 0))

        def _swap(model: nn.Module) -> None:
            for name, module in list(model.named_modules()):
                if isinstance(module, nn.Linear) and (
                    module.in_features >= 768 or module.out_features >= 768
                ):
                    parent_name = ".".join(name.split(".")[:-1])
                    child_name  = name.split(".")[-1]
                    parent = model.get_submodule(parent_name) if parent_name else model
                    setattr(parent, child_name, FusedLoRALinear(module, r, lora_alpha, lora_dropout))

        if rank == 0:
            print(f"[Student] Applying LoRA r={r} to triple-backbone...")
        _swap(self.bioclip)
        _swap(self.dinov3)
        _swap(self.convnext)

        self.set_grad_checkpointing(False)
        self._lora_applied     = True
        self._backbones_frozen = False

    # ------------------------------------------------------------------
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.ensure_backbones_loaded()
        for bb in (self.bioclip, self.dinov3, self.convnext):
            if bb is not None and hasattr(bb, "set_grad_checkpointing"):
                bb.set_grad_checkpointing(enable)

    # ------------------------------------------------------------------
    def freeze_backbones(self) -> None:
        self.ensure_backbones_loaded()
        for bb in (self.bioclip, self.dinov3, self.convnext):
            for name, p in bb.named_parameters():
                p.requires_grad_(bool("lora_" in name))
        self._backbones_frozen = True

    def unfreeze_backbones(self) -> None:
        self.ensure_backbones_loaded()
        for bb in (self.bioclip, self.dinov3, self.convnext):
            for p in bb.parameters():
                p.requires_grad_(True)
        self._backbones_frozen = False

    def freeze_stem(self, num_layers: int = 6) -> None:
        self.ensure_backbones_loaded()
        for bb in (self.bioclip, self.dinov3, self.convnext):
            if hasattr(bb, "backbone"):
                layers = list(bb.backbone.children())
                for layer in layers[:num_layers]:
                    for p in layer.parameters():
                        p.requires_grad_(False)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def update_lora_fusion(self) -> None:
        from src.models.layers.fused_lora import FusedLoRALinear
        for m in self.modules():
            if isinstance(m, FusedLoRALinear):
                m.update_fusion()

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        return_gating_weights: bool = False,
        logit_adj: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # 2-D shortcut: cached-feature path
        if x.dim() == 2:
            return self.warmup_head(x) if x.shape[1] != 1024 else self.phase1_head(x)

        from src.training.cache import chunked_backbone_forward
        self.ensure_backbones_loaded()

        ctx       = torch.no_grad() if self._backbones_frozen else torch.enable_grad()
        dev_type  = "cuda" if x.is_cuda else "cpu"
        with ctx, torch.amp.autocast(dev_type, dtype=torch.bfloat16, enabled=x.is_cuda):
            f_bio  = chunked_backbone_forward(self.bioclip,  x, x.shape[0])
            f_dino = chunked_backbone_forward(self.dinov3,   x, x.shape[0])
            f_conv = chunked_backbone_forward(self.convnext, x, x.shape[0])

        f_bio  = F.normalize(f_bio.to(x.device).float(),  p=2, dim=1)
        f_dino = F.normalize(f_dino.to(x.device).float(), p=2, dim=1)
        f_conv = F.normalize(f_conv.to(x.device).float(), p=2, dim=1)

        fused_raw = torch.cat([f_bio, f_dino, f_conv], dim=1).contiguous()

        # Expert dropout (training stability)
        if self.training and torch.rand(1).item() < 0.1:
            drop = torch.randint(0, 3, (1,)).item()
            if drop == 0:   f_bio  = f_bio  * 0
            elif drop == 1: f_dino = f_dino * 0
            else:           f_conv = f_conv * 0
            fused_raw = torch.cat([f_bio, f_dino, f_conv], dim=1)

        g_logits      = self.gating_network(fused_raw)
        if g_logits.shape[1] == 8:          # FP8 padding fallback
            g_logits = g_logits[:, :3]
        gating_weights = F.softmax(g_logits, dim=1).clamp(min=1e-6)

        w = gating_weights
        f_weighted = torch.cat([f_bio * w[:, 0:1], f_dino * w[:, 1:2], f_conv * w[:, 2:3]], dim=1)
        fused_proj = self.proj_grouped(f_weighted)

        species_logits = self.species_classifier(F.normalize(fused_proj, p=2, dim=1))

        # LogitNorm
        with torch.no_grad():
            lmean = species_logits.mean(dim=1, keepdim=True)
            lstd  = species_logits.std(dim=1,  keepdim=True).clamp(min=1e-5)
        species_logits = (species_logits - lmean) / lstd

        if logit_adj is not None:
            species_logits = species_logits + logit_adj.to(species_logits.dtype)

        if self.training:
            return species_logits

        if return_gating_weights:
            return species_logits, gating_weights
        return species_logits


# ---------------------------------------------------------------------------
class _ResidualMLP(nn.Module):
    """Lightweight head used for warmup / Phase 1 checkpoint compatibility."""

    def __init__(self, in_features: int, hidden: int, out_features: int) -> None:
        super().__init__()
        self.fc1   = nn.Linear(in_features, hidden)
        self.final = nn.Linear(hidden, out_features)
        self.ln    = nn.LayerNorm(hidden)
        self.res   = nn.Sequential(
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden),
        )
        self.drop = nn.Dropout(0.2)
        self.act  = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=self.fc1.weight.dtype)
        x = self.act(self.ln(self.fc1(x)))
        x = x + self.res(x)
        return self.final(self.drop(x))
