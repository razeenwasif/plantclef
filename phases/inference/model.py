"""Inference ensemble — owned exclusively by the inference phase.

DO NOT import from other phases. DO NOT add training-specific code here.

Each InferenceEnsemble instance is bound to a single native resolution (the
resolution at which its checkpoint was trained). The pipeline instantiates
one ensemble per checkpoint so multi-resolution ensembling (e.g. student@224
+ teacher@512) can be done without pos_embed shape mismatches.
"""
from __future__ import annotations
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class InferenceEnsemble(nn.Module):
    """Triple-backbone ensemble for inference. No LoRA. No training methods."""

    BIOCLIP_DIM  = 768
    DINOV3_DIM   = 1024
    CONVNEXT_DIM = 1536
    PROJ_DIM     = 512

    def __init__(self, num_classes: int = 7806, input_res: int = 512, gating_dim: int = 3,
                 bioclip_name:  str = "hf-hub:imageomics/bioclip-2",
                 dinov3_name:   str = "vit_large_patch16_dinov3.lvd1689m",
                 convnext_name: str = "convnextv2_large.fcmae_ft_in22k_in1k_384") -> None:
        super().__init__()
        self.num_classes   = num_classes
        self.input_res     = input_res
        self.bioclip_name  = bioclip_name
        self.dinov3_name   = dinov3_name
        self.convnext_name = convnext_name

        # ORACLE: Normalization Buffers.
        # DINOv3 / ConvNeXt expect ImageNet stats; BioCLIP expects OpenAI-CLIP stats.
        # Applying ImageNet to the BioCLIP path silently degrades its features.
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.register_buffer("bioclip_mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer("bioclip_std",  torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))

        total = self.BIOCLIP_DIM + self.DINOV3_DIM + self.CONVNEXT_DIM
        fused = self.PROJ_DIM * 3

        self.bioclip:  nn.Module | None = None
        self.dinov3:   nn.Module | None = None
        self.convnext: nn.Module | None = None

        self.proj_linear   = nn.Linear(total, fused)
        self.proj_ln       = nn.LayerNorm(fused)
        self.proj_grouped  = nn.Sequential(self.proj_linear, self.proj_ln)
        # Toggle for the (logits - mean) / std post-classifier normalization.
        # Helpful when raw logits give better-calibrated sigmoid magnitudes.
        self.disable_logit_standardization = False
        self.gating_network = nn.Sequential(
            nn.Linear(total, 128), nn.LayerNorm(128), nn.GELU(), nn.Linear(128, gating_dim)
        )
        self.agg_weights = nn.Parameter(torch.ones(fused))
        self.agg_bias    = nn.Parameter(torch.zeros(fused))

        from src.models.layers.gcn import EcologicalGCNHead
        self.species_classifier = EcologicalGCNHead(
            num_classes=num_classes, trait_dim=19, image_feat_dim=fused
        )
        self.phase1_head = _ResidualMLP(1024, 2048, num_classes)
        self.warmup_head = _ResidualMLP(total, 2048, num_classes)

    # ------------------------------------------------------------------
    def load_backbones(self) -> None:
        if self.bioclip is not None:
            return
        from src.models.bioclip      import PlantBioCLIP
        from src.models.vit_backbone import PlantViTBackbone
        from src.models.convnext     import PlantConvNeXt
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bioclip  = PlantBioCLIP(checkpoint=self.bioclip_name,  input_res=self.input_res).to(device)
        self.dinov3   = PlantViTBackbone(model_name=self.dinov3_name,   input_res=self.input_res).to(device)
        self.convnext = PlantConvNeXt(model_name=self.convnext_name, input_res=self.input_res).to(device)

    # ------------------------------------------------------------------
    def set_resolution(self, res: int) -> None:
        """Switch backbone resolution post-load (interpolates pos_embed in-place)."""
        self.input_res = res
        for bb in (self.bioclip, self.dinov3, self.convnext):
            if bb is not None and hasattr(bb, "set_resolution"):
                bb.set_resolution(res)

    # ------------------------------------------------------------------
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        resolution: int,
        num_classes: int = 7806,
        bioclip_name: str = "hf-hub:imageomics/bioclip-2",
        dinov3_name:  str = "vit_large_patch16_dinov3.lvd1689m",
        convnext_name: str = "convnextv2_large.fcmae_ft_in22k_in1k_384",
        device: str | torch.device = "cuda",
    ) -> "InferenceEnsemble":
        """Build ensemble at `resolution` and load a single checkpoint trained at that resolution."""
        # ORACLE: Peek at checkpoint to detect architecture variants (e.g. FP8 gating)
        gating_dim = 3
        if os.path.exists(checkpoint_path):
            try:
                sd_peek = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                # Unwrap if nested
                for wrap_key in ["module", "model_state", "model_state_dict"]:
                    if isinstance(sd_peek, dict) and wrap_key in sd_peek:
                        sd_peek = sd_peek[wrap_key]
                
                for k, v in sd_peek.items():
                    if k.endswith("gating_network.3.weight"):
                        gating_dim = v.shape[0]
                        break
            except: pass

        model = cls(num_classes=num_classes, input_res=resolution, gating_dim=gating_dim,
                    bioclip_name=bioclip_name, dinov3_name=dinov3_name,
                    convnext_name=convnext_name)
        model.load_backbones()
        model = model.to(device)

        if not os.path.exists(checkpoint_path):
            print(f"[Inference] WARN: checkpoint missing → {checkpoint_path}; using pretrained backbones.")
            return model.eval()

        sd = torch.load(checkpoint_path, map_location=device, weights_only=False)
        # Unwrap if nested
        for wrap_key in ["module", "model_state", "model_state_dict"]:
            if isinstance(sd, dict) and wrap_key in sd:
                sd = sd[wrap_key]

        # Strip distributed/compile prefixes
        cleaned = {}
        for k, v in sd.items():
            nk = k.replace("_orig_mod.", "").replace("module.", "")
            cleaned[nk] = v

        model_sd   = model.state_dict()
        compatible = {k: v for k, v in cleaned.items() if k in model_sd}
        mismatched = [k for k, v in compatible.items() if v.shape != model_sd[k].shape]
        if mismatched:
            print(f"[Inference] WARN: {len(mismatched)} shape-mismatched keys at res={resolution} "
                  f"(e.g. {mismatched[0]}). Most likely the checkpoint was trained at a different resolution.")
            compatible = {k: v for k, v in compatible.items() if v.shape == model_sd[k].shape}

        missing = set(model_sd) - set(compatible)
        model.load_state_dict(compatible, strict=False)
        print(f"[Inference] Loaded {len(compatible)}/{len(model_sd)} keys from {os.path.basename(os.path.dirname(checkpoint_path))}/{os.path.basename(checkpoint_path)} (res={resolution}px); {len(missing)} missing.")
        return model.eval()

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, return_features: bool = False):
        """If return_features=True, returns (logits, raw_concat_features)
        where raw_concat_features = L2-normalized [bio | dino | conv]
        — the same form as the prototype index built from phase1 features.
        Used by the FAISS retrieval engine."""
        self.load_backbones()

        # ORACLE: Per-backbone normalization (BioCLIP uses CLIP stats, others use ImageNet).
        x_imagenet = (x - self.mean) / self.std
        x_bioclip  = (x - self.bioclip_mean) / self.bioclip_std

        dev_type = "cuda" if x.is_cuda else "cpu"
        with torch.amp.autocast(dev_type, dtype=torch.bfloat16, enabled=x.is_cuda):
            f_bio  = self.bioclip(x_bioclip)
            f_dino = self.dinov3(x_imagenet)
            f_conv = self.convnext(x_imagenet)

        f_bio  = F.normalize(f_bio.to(x.device).float(),  p=2, dim=1)
        f_dino = F.normalize(f_dino.to(x.device).float(), p=2, dim=1)
        f_conv = F.normalize(f_conv.to(x.device).float(), p=2, dim=1)

        fused_raw      = torch.cat([f_bio, f_dino, f_conv], dim=1)
        g_logits       = self.gating_network(fused_raw)
        if g_logits.shape[1] == 8:
            g_logits = g_logits[:, :3]
        gating_weights = F.softmax(g_logits, dim=1).clamp(min=1e-6)

        w = gating_weights
        f_weighted = torch.cat([f_bio * w[:, 0:1], f_dino * w[:, 1:2], f_conv * w[:, 2:3]], dim=1)
        fused_proj = self.proj_grouped(f_weighted)

        logits = self.species_classifier(F.normalize(fused_proj, p=2, dim=1))
        if not self.disable_logit_standardization:
            with torch.no_grad():
                lmean = logits.mean(dim=1, keepdim=True)
                lstd  = logits.std(dim=1,  keepdim=True).clamp(min=1e-5)
            logits = (logits - lmean) / lstd

        if return_features:
            # Re-normalize the concat features so they match the prototype index format.
            return logits, F.normalize(fused_raw, p=2, dim=1)
        return logits


class CRTInferenceEnsemble(nn.Module):
    """Dual-head cRT Ensemble (Head A + Head B) on Triple Backbone.
    Used for long-tail recognition without Destroying representation space.
    """
    BIOCLIP_DIM  = 768
    DINOV3_DIM   = 1024
    CONVNEXT_DIM = 1536

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

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.register_buffer("bioclip_mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer("bioclip_std",  torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))

        self.bioclip:  nn.Module | None = None
        self.dinov3:   nn.Module | None = None
        self.convnext: nn.Module | None = None

        total = self.BIOCLIP_DIM + self.DINOV3_DIM + self.CONVNEXT_DIM
        self.head_a = CRTHead(total, 2048, num_classes)
        self.head_b = CRTHead(total, 2048, num_classes)
        self.disable_logit_standardization = False

    def load_backbones(self) -> None:
        if self.bioclip is not None: return
        from src.models.bioclip import PlantBioCLIP
        from src.models.vit_backbone import PlantViTBackbone
        from src.models.convnext import PlantConvNeXt
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bioclip = PlantBioCLIP(checkpoint=self.bioclip_name, input_res=self.input_res).to(device)
        self.dinov3 = PlantViTBackbone(model_name=self.dinov3_name, input_res=self.input_res).to(device)
        self.convnext = PlantConvNeXt(model_name=self.convnext_name, input_res=self.input_res).to(device)

    def set_resolution(self, res: int) -> None:
        self.input_res = res
        for bb in (self.bioclip, self.dinov3, self.convnext):
            if bb is not None and hasattr(bb, "set_resolution"):
                bb.set_resolution(res)

    @classmethod
    def from_crt_dir(
        cls,
        crt_dir: str,
        resolution: int = 224,
        num_classes: int = 7808,
        bioclip_name: str = "hf-hub:imageomics/bioclip-2",
        dinov3_name:  str = "vit_large_patch16_dinov3.lvd1689m",
        convnext_name: str = "convnextv2_large.fcmae_ft_in22k_in1k_384",
        device: str | torch.device = "cuda",
    ) -> "CRTInferenceEnsemble":
        model = cls(num_classes=num_classes, input_res=resolution,
                    bioclip_name=bioclip_name, dinov3_name=dinov3_name,
                    convnext_name=convnext_name)
        model.load_backbones()
        model = model.to(device)

        path_a = os.path.join(crt_dir, "head_a_standard_best.pth")
        path_b = os.path.join(crt_dir, "head_b_balanced_best.pth")

        if os.path.exists(path_a):
            sd_a = torch.load(path_a, map_location=device)
            model.head_a.load_state_dict(sd_a.get("model_state", sd_a))
            print(f"[Inference] Loaded cRT Head A from {path_a}")
        
        if os.path.exists(path_b):
            sd_b = torch.load(path_b, map_location=device)
            model.head_b.load_state_dict(sd_b.get("model_state", sd_b))
            print(f"[Inference] Loaded cRT Head B from {path_b}")

        return model.eval()

    def forward(self, x: torch.Tensor):
        self.load_backbones()
        x_imagenet = (x - self.mean) / self.std
        x_bioclip  = (x - self.bioclip_mean) / self.bioclip_std

        dev_type = "cuda" if x.is_cuda else "cpu"
        with torch.amp.autocast(dev_type, dtype=torch.bfloat16, enabled=x.is_cuda):
            f_bio  = self.bioclip(x_bioclip)
            f_dino = self.dinov3(x_imagenet)
            f_conv = self.convnext(x_imagenet)

        f_bio  = F.normalize(f_bio.to(x.device).float(),  p=2, dim=1)
        f_dino = F.normalize(f_dino.to(x.device).float(), p=2, dim=1)
        f_conv = F.normalize(f_conv.to(x.device).float(), p=2, dim=1)

        fused_raw = torch.cat([f_bio, f_dino, f_conv], dim=1)
        
        logits_a = self.head_a(fused_raw)
        logits_b = self.head_b(fused_raw)

        if not self.disable_logit_standardization:
            for l in [logits_a, logits_b]:
                lmean = l.mean(dim=1, keepdim=True)
                lstd  = l.std(dim=1,  keepdim=True).clamp(min=1e-5)
                l.copy_((l - lmean) / lstd)

        # Ensemble probabilities (matching eval.py)
        probs_a = torch.softmax(logits_a, dim=-1)
        probs_b = torch.softmax(logits_b, dim=-1)
        avg_probs = (probs_a + probs_b) / 2.0
        
        # Return log-probs so they can be summed/ensembled with other models if needed
        return torch.log(avg_probs + 1e-9)


class BioCLIPMultiTaskEnsemble(nn.Module):
    """Support for i002 (per-head MLP) and 010 (shared MLP) BioCLIP 2.5 variants."""
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        super().__init__()
        from src.models.bioclip_multitask import load_checkpoint_model
        self.model, self.encoders, self.config = load_checkpoint_model(checkpoint_path, device=device)
        self.input_res = self.config.get("resolution", 224)
        self.device = device
        # Normalization (BioCLIP 2.5 uses CLIP stats)
        self.register_buffer("mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))
        self.disable_logit_standardization = False

    def forward(self, x: torch.Tensor, return_features: bool = False):
        x = (x - self.mean) / self.std
        with torch.inference_mode():
            # BioCLIP25MultiTask returns (species_logits, genus_logits, ...)
            outputs = self.model(x)
            logits = outputs[0] # species_logits
        
        if return_features:
            # Multi-task models don't easily expose fused features for FAISS 
            # in the same format as the triple-backbone ensemble.
            # Return normalized raw backbone features as fallback.
            feat = self.model._encode_raw(x)
            return logits, F.normalize(feat, p=2, dim=1)
        return logits


class CRTHead(nn.Module):
    """Matches the ClassificationHead architecture from crt_train/model.py exactly."""
    def __init__(self, in_features: int, hidden_features: int, out_features: int) -> None:
        super().__init__()
        self.fc1      = nn.Linear(in_features, hidden_features)
        self.fc_final = nn.Linear(hidden_features, out_features)
        self.ln1      = nn.LayerNorm(hidden_features)
        self.gelu     = nn.GELU()
        self.res_block = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.LayerNorm(hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, hidden_features),
            nn.LayerNorm(hidden_features),
        )
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=self.fc1.weight.dtype)
        x = self.gelu(self.ln1(self.fc1(x)))
        x = x + self.res_block(x)
        return self.fc_final(self.dropout(x))


class LegacyStandaloneModel(nn.Module):
    """Generic wrapper for single-backbone models (e.g. 0.38 LB giants)."""
    def __init__(self, checkpoint_path: str, resolution: int, device: str = "cuda"):
        super().__init__()
        self.input_res = resolution
        self.device = device

        # ORACLE: Normalization Buffers (ImageNet defaults)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        sd = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        # Unwrap
        orig_sd = sd
        for k in ["module", "model_state", "model_state_dict"]:
            if k in sd: sd = sd[k]
        
        # 1. Metadata detection
        config = orig_sd.get("config", {})
        model_name_meta = config.get("model_name", "")
        
        # 2. Head dimension detection
        head_in_dim = 1024
        target_prefix = ""
        if "species_head.weight" in sd: 
            target_prefix = "species_head"
            head_in_dim = sd["species_head.weight"].shape[1]
        elif "head.4.weight" in sd:
            target_prefix = "head.4"
            head_in_dim = sd["head.4.weight"].shape[1]
        elif "head.weight" in sd:
            target_prefix = "head"
            head_in_dim = sd["head.weight"].shape[1]
        elif "classifier.2.weight" in sd:
            # ORACLE: Support for lightweight specialists (classifier.0=LN, classifier.2=Linear)
            target_prefix = "classifier.2"
            head_in_dim = sd["classifier.2.weight"].shape[1]

        # 3. Backbone Detection & Init
        is_bioclip = any("bioclip" in k.lower() or "visual.conv1" in k for k in sd.keys()) or "bioclip" in model_name_meta.lower() or "bioclip" in checkpoint_path.lower()
        is_lightweight = "lightweight" in checkpoint_path.lower()

        # BioCLIP / OpenAI-CLIP normalization stats (training transforms used these,
        # not ImageNet's). Override the default ImageNet buffers when this is a BioCLIP backbone.
        if is_bioclip:
            self.mean.data.copy_(torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
            self.std.data.copy_(torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))
            print(f"  -> Using BioCLIP normalization stats (mean={self.mean.flatten().tolist()})")

        if is_bioclip and not is_lightweight:
            from src.models.bioclip import PlantBioCLIP
            # Use metadata name if available, fallback to standard or Huge
            load_name = model_name_meta if "bioclip" in model_name_meta.lower() else "hf-hub:imageomics/bioclip-2"
            # Special case: if we see 1024-dim head and it's a BioCLIP variant, it's likely ViT-L or ViT-H
            if head_in_dim == 1024 and "bioclip-2" in load_name:
                if "best.pt" in checkpoint_path or "vith" in model_name_meta.lower():
                    load_name = "hf-hub:imageomics/bioclip-2.5-vith14"

            self.backbone = PlantBioCLIP(checkpoint=load_name, input_res=resolution)
        elif is_lightweight:
            # ORACLE: Handle lightweight specialists trained with timm backbones
            from src.models.vit_backbone import PlantViTBackbone
            if "bioclip" in checkpoint_path.lower():
                model_name = "vit_tiny_patch16_224"
            else:
                model_name = "vit_small_patch14_dinov2"
            self.backbone = PlantViTBackbone(model_name=model_name, input_res=resolution)
        else:
            from src.models.vit_backbone import PlantViTBackbone
            self.backbone = PlantViTBackbone(model_name="vit_large_patch16_dinov3.lvd1689m", input_res=resolution)

        # 4. Optional shared MLP (LayerNorm + Linear) detected from checkpoint.
        # 009-style training puts an LN+Linear between the backbone and species_head.
        # Without this, species_head sees raw backbone features and produces near-uniform logits.
        self.shared_mlp: nn.Module | None = None
        if "shared_mlp.net.0.weight" in sd and "shared_mlp.net.1.weight" in sd:
            mlp_dim = sd["shared_mlp.net.1.weight"].shape[0]
            self.shared_mlp = _SharedMLP(mlp_dim)
            print(f"  -> Detected shared_mlp (dim={mlp_dim}); will load LN+Linear pre-head")

        # 5. Final Head Init
        # ORACLE: Adaptive Head Dimensioning
        # Check actual weight shape in checkpoint to prevent size mismatches
        actual_num_classes = 7806
        if target_prefix and f"{target_prefix}.weight" in sd:
            actual_num_classes = sd[f"{target_prefix}.weight"].shape[0]
        elif target_prefix == "classifier.2" and "classifier.2.weight" in sd:
             actual_num_classes = sd["classifier.2.weight"].shape[0]
        
        self.head = nn.Linear(head_in_dim, actual_num_classes)
        if target_prefix == "head.4":
            # Experiment 008 style sequential head
            self.head = nn.Sequential(
                nn.LayerNorm(2048) if head_in_dim == 2048 else nn.Identity(),
                nn.Linear(head_in_dim, 1024) if head_in_dim == 2048 else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(1024, actual_num_classes)
            )

        self.to(device)
        
        # 5. Load backbone weights.
        # The 009-style finetune saved keys as `backbone.visual.X` (visual encoder)
        # and `backbone.X` (text encoder). PlantBioCLIP wraps OpenCLIP as
        # `self.model = full_clip; self.backbone = self.model.visual`, so its
        # state_dict has every weight available under `model.X`. Map ckpt
        # `backbone.X` → `model.X` and load with strict=False to skip what doesn't
        # apply to a particular backbone variant.
        cleaned = {}
        for k, v in sd.items():
            nk = k.replace("_orig_mod.", "").replace("module.", "")
            if nk.startswith("backbone."):
                nk = "model." + nk[len("backbone."):]
            cleaned[nk] = v

        bb_sd = self.backbone.state_dict()
        compat = {k: v for k, v in cleaned.items() if k in bb_sd and v.shape == bb_sd[k].shape}
        self.backbone.load_state_dict(compat, strict=False)
        n_skipped = len(cleaned) - len(compat)
        print(f"  -> backbone: loaded {len(compat)}/{len(bb_sd)} keys "
              f"(skipped {n_skipped} non-matching like classifier heads / shape-mismatched pos_embed)")
        if len(compat) == 0:
            print(f"  -> WARNING: zero backbone keys matched — backbone is HF-pretrained, not fine-tuned!")

        # Load shared_mlp weights if present (matches state-dict keys exactly: shared_mlp.net.{0,1}.{weight,bias})
        if self.shared_mlp is not None:
            mlp_sd = {k: v for k, v in sd.items() if k.startswith("shared_mlp.")}
            mlp_sd = {k.replace("shared_mlp.", ""): v for k, v in mlp_sd.items()}
            missing, unexpected = self.shared_mlp.load_state_dict(mlp_sd, strict=False)
            print(f"  -> shared_mlp loaded ({len(mlp_sd)} keys; missing={list(missing)}, unexpected={list(unexpected)})")

        if self.head:
            head_sd = {}
            if target_prefix:
                print(f"  -> Selecting head weights with prefix: {target_prefix} (dim={head_in_dim})")
                for k, v in sd.items():
                    if k.startswith(target_prefix + "."):
                        hk = k.replace(target_prefix + ".", "")
                        head_sd[hk] = v
                    elif k == target_prefix:
                        head_sd["weight"] = v

            if head_sd:
                self.head.load_state_dict(head_sd, strict=False)
            
        self.eval()
        print(f"[Inference] Loaded Legacy Model from {os.path.basename(checkpoint_path)} ({self.backbone.__class__.__name__}, dim={head_in_dim})")

    def forward(self, x: torch.Tensor, return_features: bool = False):
        # ORACLE: Apply Normalization
        x = (x - self.mean) / self.std

        dev_type = "cuda" if x.is_cuda else "cpu"
        with torch.amp.autocast(dev_type, dtype=torch.bfloat16, enabled=x.is_cuda):
            # Get backbone features
            if hasattr(self.backbone, "forward_features"):
                # DINOv3-style: get all tokens
                all_tokens = self.backbone.forward_features(x)
                cls_token = all_tokens[:, 0]
                patch_tokens = all_tokens[:, 1:]
                
                # Check if head expects concatenated features (2048)
                head_input_dim = 0
                if hasattr(self.head, "weight"): head_input_dim = self.head.weight.shape[1]
                elif isinstance(self.head, nn.Sequential):
                    for m in self.head:
                        if hasattr(m, "weight") and not isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                            head_input_dim = m.weight.shape[1]
                            break
                
                if head_input_dim == 2048:
                    # CLS + Global Mean Pool
                    gem = patch_tokens.mean(dim=1)
                    feats = torch.cat([cls_token, gem], dim=1)
                else:
                    feats = cls_token
            else:
                feats = self.backbone(x)
        
        if feats.ndim > 2: feats = feats.mean(dim=(2, 3))

        feats = feats.float()
        if self.shared_mlp is not None:
            feats = self.shared_mlp(feats)

        if self.head:
            logits = self.head(feats)
        else:
            logits = torch.zeros((x.shape[0], 7806), device=x.device)

        if return_features:
            return logits, F.normalize(feats, p=2, dim=1)
        return logits


class _SharedMLP(nn.Module):
    """Mirrors the 009/010-style multitask `shared_mlp` module:
        net.0 = LayerNorm,  net.1 = Linear,  net.2 = GELU,  net.3 = Dropout.

    Older checkpoints (009) only had keys for net.0/net.1 — that still loads
    cleanly here because GELU and Dropout have no parameters. Newer checkpoints
    (010 multitask) trained their species_head AFTER a GELU non-linearity, so
    omitting GELU at inference silently produces blunted/incorrect logits.
    Dropout is a no-op in eval mode but kept in the chain for structural parity."""
    def __init__(self, in_dim: int, hidden_dim: int | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.to(dtype=self.net[1].weight.dtype))


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
