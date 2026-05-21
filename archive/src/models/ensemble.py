import torch
import torch.nn as nn
import torch.nn.functional as F
from .bioclip import PlantBioCLIP
from .dinov2 import PlantDINOv2
from .convnext import PlantConvNeXt


class ResidualMLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout=0.2, num_samples=5):
        super().__init__()
        self.num_samples = num_samples
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.ln1 = nn.LayerNorm(hidden_features)
        self.gelu = nn.GELU()
        
        self.res_block = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.LayerNorm(hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, hidden_features),
            nn.LayerNorm(hidden_features)
        )
        
        # Multi-Sample Dropout for the final classification layer
        self.dropout_samples = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_samples)])
        self.fc_final = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.gelu(self.ln1(self.fc1(x)))
        x = x + self.res_block(x)
        
        if self.training:
            # Multi-sample dropout: average logits from multiple masks
            logits = sum(self.fc_final(d(x)) for d in self.dropout_samples) / self.num_samples
        else:
            logits = self.fc_final(x)
        return logits


class PlantEnsemble(nn.Module):
    def __init__(self, num_classes=7800, input_res=384,
                 bioclip_name='hf-hub:imageomics/bioclip',
                 dinov2_name='vit_large_patch14_dinov2',
                 convnext_name='convnextv2_large.fcmae_ft_in22k_in1k_384'):
        """
        Triple Ensemble for PlantCLEF 2026.
        1. BioCLIP     (Taxonomic Foundation)     -- feature_dim: 512
        2. DINOv2-L    (Geometric/Structural)     -- feature_dim: 1024
        3. ConvNeXt-V2-L (Convolutional/Local)    -- feature_dim: 1536

        Phase 1: all backbones frozen, only projection heads + classifier train.
        Phase 2: LoRA adapters on DINOv2 + ConvNeXt; BioCLIP stays fully frozen.
                 ~5M trainable params vs ~800M for full fine-tuning.
        """
        super().__init__()
        print(f"Initializing Triple Ensemble ({input_res}px)...")

        self.bioclip  = PlantBioCLIP(checkpoint=bioclip_name, input_res=input_res)
        self.dinov2   = PlantDINOv2(model_name=dinov2_name,   input_res=input_res)
        self.convnext = PlantConvNeXt(model_name=convnext_name, input_res=input_res)

        # Grouped projection to shared 1536-d space (512 * 3)
        # Using a single Linear layer is much faster on Blackwell than 3 sequential ones
        # and allows torch.compile to generate a single fused kernel.
        self.PROJ_DIM = 512
        self.total_input_dim = self.bioclip.feature_dim + self.dinov2.feature_dim + self.convnext.feature_dim
        self.fusion_dim      = self.PROJ_DIM * 3
        
        self.proj_grouped = nn.Sequential(
            nn.Linear(self.total_input_dim, self.fusion_dim),
            nn.LayerNorm(self.fusion_dim)
        )

        print(f"Backbone dims: BioCLIP={self.bioclip.feature_dim}, "
              f"DINOv2={self.dinov2.feature_dim}, "
              f"ConvNeXt={self.convnext.feature_dim}")
        print(f"Grouped Projection: {self.total_input_dim} -> {self.fusion_dim}")

        # Deep fusion classifier
        # LayerNorm > BatchNorm for ViT outputs (batch stats unreliable at small B)
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes)
        )

        # Phase 1 linear probe on PCA-compressed features.
        # Upgraded to ResidualMLP with Multi-Sample Dropout for better PCA feature modeling.
        from config import PCA_COMPONENTS
        self.phase1_head = ResidualMLP(
            in_features=PCA_COMPONENTS, 
            hidden_features=2048, 
            out_features=num_classes,
            dropout=0.2
        )

        self._backbones_frozen = False
        self._lora_applied     = False

        # Blackwell Parallel Orchestration
        try:
            import plantclef_ext
            self.orchestrator = plantclef_ext.StreamOrchestrator()
            self.has_ext = True
        except ImportError:
            self.has_ext = False

    # -----------------------------------------------------------------------
    # Forward (used during validation; training calls components directly)
    # -----------------------------------------------------------------------

    def forward(self, x):
        if not self.has_ext:
            # Slow fallback path
            ctx = torch.no_grad() if self._backbones_frozen else torch.enable_grad()
            with ctx:
                feat_bio  = self.bioclip(x)
                feat_dino = self.dinov2(x)
                feat_conv = self.convnext(x)
            fused_raw = torch.cat([feat_bio, feat_dino, feat_conv], dim=1)
            fused_proj = self.proj_grouped(fused_raw)
            return self.classifier(F.normalize(fused_proj, dim=1))

        # --- OPTIMIZED BLACKWELL PATH (Parallel Streams + Fused Kernel) ---
        import plantclef_ext
        ctx = torch.no_grad() if self._backbones_frozen else torch.enable_grad()
        
        # 1. Launch all 3 backbones in parallel streams
        with ctx:
            # We use ExternalStream to wrap the C++ created streams
            s0 = torch.cuda.ExternalStream(self.orchestrator.get_stream(0))
            s1 = torch.cuda.ExternalStream(self.orchestrator.get_stream(1))
            s2 = torch.cuda.ExternalStream(self.orchestrator.get_stream(2))

            with torch.cuda.stream(s0):
                feat_bio = self.bioclip(x)
            with torch.cuda.stream(s1):
                feat_dino = self.dinov2(x)
            with torch.cuda.stream(s2):
                feat_conv = self.convnext(x)

            # Wait for all backbones to finish
            self.orchestrator.synchronize()

        # 2. Run Fused Slotted Projection (Eliminates 'cat' and sequential GEMM)
        linear = self.proj_grouped[0]
        ln     = self.proj_grouped[1]
        
        fused_proj = plantclef_ext.fused_projection(
            feat_bio.float(), feat_dino.float(), feat_conv.float(),
            linear.weight.float(), linear.bias.float(),
            ln.weight.float(), ln.bias.float()
        )
        fused_proj = fused_proj.to(feat_bio.dtype)

        # 3. Final Classifier
        return self.classifier(F.normalize(fused_proj, dim=1))

    # -----------------------------------------------------------------------
    # LoRA
    # -----------------------------------------------------------------------

    def apply_lora(self, r=16, lora_alpha=32, lora_dropout=0.05):
        """
        Apply LoRA adapters to DINOv2 and ConvNeXt.
        BioCLIP is intentionally left fully frozen -- its Tree-of-Life pretraining
        already provides taxonomically discriminative features.

        Target modules:
          DINOv2  (timm ViT)  : qkv, proj, fc1, fc2
          ConvNeXt (timm CNN) : fc1, fc2  (MLP layers inside each block)

        Reduces trainable params from ~800M -> ~5M while keeping most of the
        fine-tuning benefit, and enables batch=256 without OOM.
        """
        try:
            from peft import get_peft_model, LoraConfig
        except ImportError:
            raise ImportError(
                "PEFT not installed. Run: pip install peft\n"
                "LoRA is required for Phase 2 training."
            )

        if self._lora_applied:
            print("[LoRA] Already applied -- skipping.")
            return

        # Disable gradient checkpointing before applying LoRA to avoid hook conflicts
        self.set_grad_checkpointing(False)

        # DINOv2: LoRA on attention (QKV + output) and MLP (fc1, fc2)
        dino_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["qkv", "proj", "fc1", "fc2"],
            bias="none",
        )
        self.dinov2.backbone = get_peft_model(self.dinov2.backbone, dino_config)

        # ConvNeXt: LoRA on MLP linear layers
        conv_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["fc1", "fc2"],
            bias="none",
        )
        self.convnext.backbone = get_peft_model(self.convnext.backbone, conv_config)

        # Re-enable gradient checkpointing after LoRA application
        self.set_grad_checkpointing(True)

        self._lora_applied     = True
        self._backbones_frozen = False  # LoRA params are trainable

        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[LoRA] Applied: {trainable:,} trainable / {total:,} total "
              f"({100*trainable/total:.2f}%)")

    # -----------------------------------------------------------------------
    # Gradient checkpointing
    # -----------------------------------------------------------------------

    def set_grad_checkpointing(self, enable=True):
        for wrapper in [self.bioclip, self.dinov2, self.convnext]:
            # Try finding the method directly on the wrapper, its backbone, or its model
            for attr_name in ['', 'backbone', 'model', 'base_model']:
                obj = wrapper if not attr_name else getattr(wrapper, attr_name, None)
                if obj is None:
                    continue
                
                # Check for standard PyTorch/timm/HF method name
                if hasattr(obj, 'set_grad_checkpointing'):
                    obj.set_grad_checkpointing(enable)
                    break
                # Check for PEFT method name
                elif enable and hasattr(obj, 'gradient_checkpointing_enable'):
                    obj.gradient_checkpointing_enable()
                    break
                elif not enable and hasattr(obj, 'gradient_checkpointing_disable'):
                    obj.gradient_checkpointing_disable()
                    break
        
        status = 'enabled' if enable else 'disabled'
        print(f"Gradient checkpointing {status}.")

    # -----------------------------------------------------------------------
    # Freeze / unfreeze
    # -----------------------------------------------------------------------

    def freeze_backbones(self):
        """Freeze all backbone params for Phase 1 head warmup."""
        for backbone in [self.bioclip, self.dinov2, self.convnext]:
            for param in backbone.parameters():
                param.requires_grad = False
        self._backbones_frozen = True
        print("All 3 backbones frozen.")

    def unfreeze_backbones(self):
        """Unfreeze all backbone params (used for full fine-tuning without LoRA)."""
        for backbone in [self.bioclip, self.dinov2, self.convnext]:
            for param in backbone.parameters():
                param.requires_grad = True
        self._backbones_frozen = False
        print("All 3 backbones unfrozen.")
