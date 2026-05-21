"""Phase 2b Teacher ensemble — owned exclusively by expert_specialization.

DO NOT modify this file for student or inference purposes.
Duplicate into the relevant phase's model.py instead.

Differences from StudentEnsemble:
  - Designed for heavy LoRA (r=512 by default)
  - 512px resolution specialist
  - Includes SWA-compatible key remapping utilities
"""
from __future__ import annotations
import contextlib
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union


try:
    import plantclef_ext
    # ORACLE: Tell Torch Dynamo these C++ kernels are safe for CUDA Graphs
    if hasattr(torch, 'compiler'):
        torch.compiler.allow_in_graph(plantclef_ext.fused_gfam_projection)
except ImportError:
    plantclef_ext = None

try:
    import transformer_engine.pytorch as te
    HAS_TE = True
except ImportError:
    HAS_TE = False


class TeacherEnsemble(nn.Module):
    """Triple-backbone ensemble for high-resolution teacher fine-tuning.

    Architecture is intentionally identical to StudentEnsemble — isolation is
    achieved by file separation, not by diverging design.
    """

    BIOCLIP_DIM  = 768
    DINOV3_DIM   = 1024
    CONVNEXT_DIM = 1536
    PROJ_DIM     = 512

    def __init__(self, num_classes: int = 7808, input_res: int = 512,
                 bioclip_name:  str = "hf-hub:imageomics/bioclip-2",
                 dinov3_name:   str = "vit_large_patch16_dinov3.lvd1689m",
                 convnext_name: str = "convnextv2_large.fcmae_ft_in22k_in1k_384") -> None:
        super().__init__()
        self.num_classes   = num_classes
        self.input_res     = input_res
        self.bioclip_name  = bioclip_name
        self.dinov3_name   = dinov3_name
        self.convnext_name = convnext_name

        total = self.BIOCLIP_DIM + self.DINOV3_DIM + self.CONVNEXT_DIM
        fused = self.PROJ_DIM * 3

        self.bioclip:  nn.Module | None = None
        self.dinov3:   nn.Module | None = None
        self.convnext: nn.Module | None = None

        # ORACLE: Extreme Blackwell Optimization (FP8 + Fused Layers)
        # Check for FP8 enablement from global config
        from src import config as global_cfg
        use_fp8 = getattr(global_cfg, 'USE_FP8', False)

        if use_fp8 and HAS_TE and torch.cuda.is_available():
            self.proj_linear = te.Linear(total, fused, bias=True, params_dtype=torch.bfloat16)
        else:
            self.proj_linear = nn.Linear(total, fused)

        self.proj_ln       = nn.LayerNorm(fused)
        self.proj_grouped  = nn.Sequential(self.proj_linear, self.proj_ln)

        if use_fp8 and HAS_TE and torch.cuda.is_available():
            self.gating_network = nn.Sequential(
                te.Linear(total, 128, params_dtype=torch.bfloat16),
                nn.LayerNorm(128),
                nn.GELU(),
                te.Linear(128, 8, params_dtype=torch.bfloat16)
            )
        else:
            self.gating_network = nn.Sequential(
                nn.Linear(total, 128), nn.LayerNorm(128), nn.GELU(), nn.Linear(128, 3)
            )

        self.agg_weights = nn.Parameter(torch.ones(fused))
        self.agg_bias    = nn.Parameter(torch.zeros(fused))

        from src.models.layers.gcn import EcologicalGCNHead
        self.species_classifier = EcologicalGCNHead(
            num_classes=num_classes, trait_dim=19, image_feat_dim=fused
        )
        self.phase1_head = _ResidualMLP(1024, 2048, num_classes)
        self.warmup_head = _ResidualMLP(total, 2048, num_classes)

        self._lora_applied     = False
        self._backbones_frozen = False
        self.has_ext           = False
        self.orchestrator      = None
        self._register_surgery_hooks()
        self._init_weights()

    def _register_surgery_hooks(self):
        """ORACLE: Register pre-hooks for gating surgery (3 -> 8 outputs)."""
        def _surgery_hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            target_keys = ["gating_network.3.weight", "gating_network.3.bias"]
            model_sd = self.state_dict()
            
            for tk in target_keys:
                key = prefix + tk
                if key in state_dict and key in model_sd:
                    ckpt_v = state_dict[key]
                    model_v = model_sd[key]
                    if ckpt_v.shape != model_v.shape:
                        new_v = torch.zeros_like(model_v)
                        new_v[:ckpt_v.shape[0]] = ckpt_v
                        state_dict[key] = new_v
                        # Only master rank should print to avoid spam
                        if int(os.environ.get("RANK", 0)) == 0:
                            print(f"[Surgery] Pre-hook adapted {key} from {ckpt_v.shape} to {model_v.shape}")

        self._register_load_state_dict_pre_hook(_surgery_hook)

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

    def ensure_backbones_loaded(self) -> None:
        if self.bioclip is not None:
            return
        from src.models.bioclip      import PlantBioCLIP
        from src.models.vit_backbone import PlantViTBackbone
        from src.models.convnext     import PlantConvNeXt
        rank = int(os.environ.get("RANK", 0))
        print(f"[Teacher] Rank {rank}: loading backbones (res={self.input_res})...")
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bioclip  = PlantBioCLIP(checkpoint=self.bioclip_name,  input_res=self.input_res).to(device)
        self.dinov3   = PlantViTBackbone(model_name=self.dinov3_name,   input_res=self.input_res).to(device)
        self.convnext = PlantConvNeXt(model_name=self.convnext_name, input_res=self.input_res).to(device)
        for bb in (self.bioclip, self.dinov3, self.convnext):
            bb.train(self.training)

        # --- Blackwell Parallel Orchestration ---
        if device.type == 'cuda':
            try:
                import plantclef_ext
                print("[Teacher] CUDA Stream Orchestrator active.")
                self.orchestrator = plantclef_ext.StreamOrchestrator()
                self.has_ext = True
            except (ImportError, AttributeError):
                self.has_ext = False
                self.orchestrator = None

    def set_resolution(self, res: int) -> None:
        self.input_res = res
        for bb in (self.bioclip, self.dinov3, self.convnext):
            if bb is not None and hasattr(bb, "set_resolution"):
                bb.set_resolution(res)

    def apply_lora(self, r: int = 512, lora_alpha: int = 1024, lora_dropout: float = 0.05) -> None:
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
                    child_name  =  name.split(".")[-1]
                    parent = model.get_submodule(parent_name) if parent_name else model
                    setattr(parent, child_name, FusedLoRALinear(module, r, lora_alpha, lora_dropout))

        if rank == 0:
            print(f"[Teacher] Applying LoRA r={r} (heavy specialist)...")
        _swap(self.bioclip)
        _swap(self.dinov3)
        _swap(self.convnext)

        self.set_grad_checkpointing(False)
        self._lora_applied     = True
        self._backbones_frozen = False

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.ensure_backbones_loaded()
        for bb in (self.bioclip, self.dinov3, self.convnext):
            if bb is not None and hasattr(bb, "set_grad_checkpointing"):
                bb.set_grad_checkpointing(enable)

    def freeze_backbones(self) -> None:
        self.ensure_backbones_loaded()
        for bb in (self.bioclip, self.dinov3, self.convnext):
            for name, p in bb.named_parameters():
                p.requires_grad_(bool("lora_" in name))
        self._backbones_frozen = True

    def unfreeze_backbones(self) -> None:
        self.ensure_backbones_loaded()
        for bb in (self.bioclip, self.dinov3, self.convnext):
            for p in bb.parameters(): p.requires_grad_(True)
        self._backbones_frozen = False

    def freeze_stem(self, num_layers: int = 6) -> None:
        self.ensure_backbones_loaded()
        for bb in (self.bioclip, self.dinov3, self.convnext):
            if hasattr(bb, "backbone"):
                layers = list(bb.backbone.children())
                for layer in layers[:num_layers]:
                    for p in layer.parameters(): p.requires_grad_(False)

    @torch.no_grad()
    def update_lora_fusion(self) -> None:
        from src.models.layers.fused_lora import FusedLoRALinear
        for m in self.modules():
            if isinstance(m, FusedLoRALinear):
                m.update_fusion()

    def forward(
        self,
        x: torch.Tensor,
        return_gating_weights: bool = False,
        logit_adj: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if x.dim() == 2:
            return self.warmup_head(x) if x.shape[1] != 1024 else self.phase1_head(x)

        from src.training.cache import chunked_backbone_forward
        self.ensure_backbones_loaded()

        ctx      = torch.no_grad() if self._backbones_frozen else torch.enable_grad()
        dev_type = "cuda" if x.is_cuda else "cpu"
        
        # ORACLE: Blackwell-native FP8 autocast support
        try:
            import transformer_engine.pytorch as te
            fp8_ctx = te.fp8_autocast(enabled=True) if x.is_cuda else contextlib.nullcontext()
        except ImportError:
            fp8_ctx = contextlib.nullcontext()

        with ctx, fp8_ctx, torch.amp.autocast(dev_type, dtype=torch.bfloat16, enabled=x.is_cuda):
            # Parallel Stream Orchestration (Blackwell Optimized)
            use_parallel = self.has_ext and x.is_cuda and not torch.compiler.is_compiling()
            
            if use_parallel:
                current_stream = torch.cuda.current_stream()
                s0, s1, s2 = [torch.cuda.ExternalStream(self.orchestrator.get_stream(i)) for i in range(3)]
                
                with torch.cuda.stream(s0): f_bio  = chunked_backbone_forward(self.bioclip,  x, x.shape[0])
                with torch.cuda.stream(s1): f_dino = chunked_backbone_forward(self.dinov3,   x, x.shape[0])
                with torch.cuda.stream(s2): f_conv = chunked_backbone_forward(self.convnext, x, x.shape[0])
                
                current_stream.wait_stream(s0); current_stream.wait_stream(s1); current_stream.wait_stream(s2)
                self.orchestrator.synchronize()
            else:
                f_bio  = chunked_backbone_forward(self.bioclip,  x, x.shape[0])
                f_dino = chunked_backbone_forward(self.dinov3,   x, x.shape[0])
                f_conv = chunked_backbone_forward(self.convnext, x, x.shape[0])

        f_bio  = F.normalize(f_bio.to(x.device).float(),  p=2, dim=1)
        f_dino = F.normalize(f_dino.to(x.device).float(), p=2, dim=1)
        f_conv = F.normalize(f_conv.to(x.device).float(), p=2, dim=1)

        fused_raw = torch.cat([f_bio, f_dino, f_conv], dim=1).contiguous()

        if self.training:
            # ORACLE: Dynamo-safe expert dropout (no .item() calls)
            if torch.rand(1, device=x.device) < 0.1:
                drop_idx = torch.randint(0, 3, (1,), device=x.device)
                mask = torch.ones(3, device=x.device)
                mask.scatter_(0, drop_idx, 0.0)
                
                f_bio  = f_bio  * mask[0]
                f_dino = f_dino * mask[1]
                f_conv = f_conv * mask[2]
                fused_raw = torch.cat([f_bio, f_dino, f_conv], dim=1)

        g_logits = self.gating_network(fused_raw)
        # ORACLE: Handle FP8 padding for gating (8 outputs -> 3 weights)
        if g_logits.shape[1] == 8:
            g_logits = g_logits[:, :3]
        gating_weights = F.softmax(g_logits, dim=1).clamp(min=1e-6)

        w = gating_weights
        f_weighted = torch.cat([f_bio * w[:, 0:1], f_dino * w[:, 1:2], f_conv * w[:, 2:3]], dim=1)
        fused_proj = self.proj_grouped(f_weighted)

        species_logits = self.species_classifier(F.normalize(fused_proj, p=2, dim=1))

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

    def __getstate__(self) -> dict:
        """Excludes non-picklable C++ objects from the state for deepcopy/SWA."""
        state = self.__dict__.copy()
        if 'orchestrator' in state:
            state['orchestrator'] = None
        return state

    def __setstate__(self, state: dict) -> None:
        """Restores state and re-initializes the C++ orchestrator after deepcopy."""
        self.__dict__.update(state)
        if getattr(self, 'has_ext', False):
            try:
                import plantclef_ext
                self.orchestrator = plantclef_ext.StreamOrchestrator()
            except:
                self.has_ext = False
                self.orchestrator = None


class _ResidualMLP(nn.Module):
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
