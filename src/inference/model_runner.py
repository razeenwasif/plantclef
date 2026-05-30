"""Model loading and batched tile inference.

Architecture:
- :class:`BaseModelAdapter` - abstract interface; subclass to wrap any model.
- :class:`PlantEnsembleAdapter` - concrete adapter for the PlantEnsemble
  (BioCLIP + DINOv2 + ConvNeXt triple backbone).
- :func:`load_model_adapter` - factory that builds an adapter from a checkpoint.
- :class:`ModelRunner` - handles preprocessing, batching, and forward passes.
"""

from __future__ import annotations

import abc
import logging
import os
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# plantclef: Force static shape for inference stability & CUDA Graph capture
if hasattr(torch, '_inductor'):
    import torch._inductor.config as inductor_cfg
    try:
        # Standard settings for high-performance inference
        inductor_cfg.triton.cudagraphs = True
    except AttributeError:
        pass
    
    # Standard Dynamo settings for static shapes
    if hasattr(torch, '_dynamo'):
        torch._dynamo.config.assume_static_by_default = True

from .config import ModelConfig
from .types import TileSpec, TilePrediction

try:
    import plantclef_ext
except ImportError:
    plantclef_ext = None

logger = logging.getLogger(__name__)


def _retinex_normalize(arr: np.ndarray, sigmas: tuple = (15.0, 80.0, 250.0)) -> np.ndarray:
    """Multi-Scale Retinex illumination normalization.

    Separates surface reflectance from illumination by subtracting a
    multi-scale log-Gaussian estimate of the illuminant.  Removes the
    systematic brightness / colour-cast variation caused by different
    field capture conditions (overcast vs sunny, shadow vs direct light).

    Parameters
    ----------
    arr : np.ndarray
        (H, W, 3) float32 image in [0, 1].
    sigmas : tuple
        Gaussian kernel sigmas defining the illumination scales.

    Returns
    -------
    np.ndarray
        (H, W, 3) float32 reflectance image in [0, 1], per-channel normalized.
    """
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        return arr  # scipy unavailable — pass through unchanged

    log_arr = np.log1p(arr * 255.0)  # map [0,1] → [0, log(256)], avoids log(0)
    msr = np.zeros_like(log_arr)

    for sigma in sigmas:
        for c in range(3):
            blurred = gaussian_filter(log_arr[:, :, c], sigma=sigma)
            msr[:, :, c] += log_arr[:, :, c] - blurred

    msr /= len(sigmas)

    # Per-channel min-max normalization to [0, 1]
    for c in range(3):
        lo, hi = msr[:, :, c].min(), msr[:, :, c].max()
        if hi > lo:
            msr[:, :, c] = (msr[:, :, c] - lo) / (hi - lo)

    return np.clip(msr, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Model adapter interface
# ---------------------------------------------------------------------------

class BaseModelAdapter(abc.ABC):
    """
    Abstract wrapper around an ``nn.Module`` for inference.

    Subclass this to support any model architecture. The adapter is
    responsible only for instantiating and loading the model; preprocessing
    and batching are handled by :class:`ModelRunner`.
    """

    @abc.abstractmethod
    def get_model(self) -> nn.Module:
        """
        Return the underlying ``nn.Module`` in eval mode.

        Returns
        -------
        nn.Module
            The model in evaluation mode.
        """
        ...

    @property
    @abc.abstractmethod
    def num_classes(self) -> int:
        """
        Number of output classes (logit dimension).

        Returns
        -------
        int
            Number of classes.
        """
        ...


class PlantEnsembleAdapter(BaseModelAdapter):
    """
    Adapter for the PlantEnsemble triple-backbone model.

    Loads a checkpoint saved as a ``state_dict`` (``torch.save(model.state_dict(), ...)``).

    Parameters
    ----------
    checkpoint_path : Path
        Path to the ``.pth`` file.
    num_classes_ : int, optional
        Number of output classes (default is 7806).
    input_res : int, optional
        Input resolution the model was trained on (default is 448).
    device : str, optional
        PyTorch device string (default is "cuda").
    """

    def __init__(
        self,
        checkpoint_path: Path,
        num_classes_: int = 7806,
        input_res: int = 448,
        device: str = "cuda",
        bioclip_name: str = "hf-hub:imageomics/bioclip-2",
        dinov3_name: str = "vit_large_patch16_dinov3.lvd1689m",
        convnext_name: str = "convnextv2_large.fcmae_ft_in22k_in1k_384",
        use_region_features: bool = False,
        use_zero_shot: bool = False,
        zero_shot_path: Optional[str] = None,
        zero_shot_weight: float = 0.3,
        zero_shot_temp: float = 0.07,
    ) -> None:
        """Initialize the ensemble and load weights."""
        try:
            import sys
            # Ensure src/ is importable when running from project root.
            src_root = Path(__file__).resolve().parent.parent
            if str(src_root) not in sys.path:
                sys.path.insert(0, str(src_root))
            from models.ensemble import PlantEnsemble, ResidualMLP
        except ImportError as exc:
            raise ImportError(
                "Could not import PlantEnsemble from src/models/ensemble.py. "
                "Make sure you are running from the project root or that src/ "
                "is on your PYTHONPATH."
            ) from exc

        self._num_classes = num_classes_
        target_device = _resolve_device(device)
        
        # 1. Initialize architecture and move to GPU immediately
        logger.info(f"Initializing architecture on {target_device}...")
        self._model = PlantEnsemble(
            num_classes=num_classes_, 
            input_res=input_res,
            bioclip_name=bioclip_name,
            dinov3_name=dinov3_name,
            convnext_name=convnext_name,
            use_region_features=use_region_features,
            use_zero_shot=use_zero_shot,
            zero_shot_path=zero_shot_path,
            zero_shot_weight=zero_shot_weight,
            zero_shot_temp=zero_shot_temp
        ).to(target_device)

        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            logger.info("Loading checkpoint from %s...", checkpoint_path)
            # Load checkpoint to CPU first to save VRAM, then surgery moves to GPU
            ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # Extract state dict
            if isinstance(ckpt, dict):
                state = ckpt.get("module") or ckpt.get("state_dict") or ckpt.get("model_state") or ckpt.get("model_state_dict") or ckpt.get("model") or ckpt
            else:
                state = ckpt

            # plantclef: Detect if this is a teammate single-backbone model
            # Ensemble models use nested prefixes: 'bioclip.backbone.', 'dinov3.backbone.', etc.
            # Teammate models use top-level 'backbone.' or 'classifier.'
            ensemble_prefixes = ['bioclip.backbone', 'dinov3.backbone', 'convnext.backbone', 'proj_linear']
            self.is_teammate = not any(any(k.startswith(p) for p in ensemble_prefixes) for k in state.keys())
            
            if self.is_teammate:
                logger.info("[Surgery] Teammate single-backbone model detected. Re-routing weights...")
                
                # plantclef: Improved Expert Detection
                if any(k.endswith("text_projection") for k in state.keys()) or "bioclip" in str(checkpoint_path).lower():
                    target_expert = "bioclip"
                else:
                    target_expert = "dinov3"

                # plantclef: Find classification head FIRST to determine backbone requirements
                def find_head_key(s):
                    # priority candidates (looking for exactly _num_classes outputs)
                    candidates = ["species_head.weight", "fc_final.weight", "head.weight", "classifier.weight", "fc.weight"]
                    for cand in candidates:
                        key = next((k for k in s.keys() if k.endswith(cand) and "backbone" not in k), None)
                        if key and s[key].shape[0] == self._num_classes: return key
                    return None

                head_weight_key = find_head_key(state)
                
                # plantclef: Dynamic Backbone Adaptation
                # We use the head's input dimension to pick the correct BioCLIP variant
                if target_expert == "bioclip" and head_weight_key:
                    required_dim = state[head_weight_key].shape[1]
                    # Detect internal dimension if a projection layer exists
                    # We check for internal transformer width (e.g. resblocks.0.attn)
                    internal_dim = next((v.shape[1] for k, v in state.items() if "transformer.resblocks.0.attn.in_proj_weight" in k), required_dim)
                    
                    if internal_dim == 1280:
                        logger.info(f"[Surgery] Head expects features from {required_dim} (via 1280 internal). Upgrading to BioCLIP-H (ViT-H/14).")
                        bioclip_name = "hf-hub:imageomics/bioclip" 
                    elif internal_dim == 1024:
                        logger.info(f"[Surgery] Head expects features from 1024 internal. Upgrading to BioCLIP-L (ViT-L/14).")
                        bioclip_name = "hf-hub:imageomics/bioclip-vit-large-patch14"

                self._model = PlantEnsemble(
                    num_classes=num_classes_, 
                    input_res=input_res,
                    bioclip_name=bioclip_name,
                    dinov3_name=dinov3_name,
                    convnext_name=convnext_name,
                    use_region_features=use_region_features,
                    use_zero_shot=use_zero_shot,
                    zero_shot_path=zero_shot_path,
                    zero_shot_weight=zero_shot_weight,
                    zero_shot_temp=zero_shot_temp
                ).to(target_device)
                
                self._model.ensure_backbones_loaded()
                logger.info(f"[Surgery] Hijacking '{target_expert}' with teammate weights.")
                
                if head_weight_key:
                    feat_dim = state[head_weight_key].shape[1]
                    logger.info(f"[Surgery] Detected teammate head input_dim={feat_dim} from {head_weight_key}")
                    
                    # Single linear layer head
                    if "fc1.weight" not in head_weight_key and "classifier.0" not in head_weight_key:
                        logger.info(f"[Surgery] Single-layer head detected. Re-initializing phase1_head as SimpleLinearHead.")
                        self._model.phase1_head = SimpleLinearHead(feat_dim, self._num_classes).to(target_device)
                    else:
                        # MLP head
                        if feat_dim != self._model.phase1_head.fc1.in_features:
                            logger.info(f"[Surgery] Adjusting MLP phase1_head input_dim: {feat_dim}")
                            self._model.phase1_head = ResidualMLP(
                                in_features=feat_dim,
                                hidden_features=2048,
                                out_features=self._num_classes
                            ).to(target_device)
                
                teammate_state = {}
                head_mapped_keys = set()
                for k, v in state.items():
                    clean_k = k.replace("module.", "")
                    if clean_k.startswith("backbone."):
                        # Map internal backbone parts
                        new_k = clean_k.replace("backbone.", f"{target_expert}.backbone.")
                        teammate_state[new_k] = v
                    elif "visual.proj" in clean_k:
                        # plantclef: Handle the 1280 -> 1024 projection layer for ViT-H
                        teammate_state[f"{target_expert}.backbone.proj"] = v
                    elif head_weight_key and clean_k == head_weight_key:
                        teammate_state["phase1_head.fc_final.weight"] = v
                        head_mapped_keys.add("weight")
                    elif head_weight_key and clean_k == head_weight_key.replace("weight", "bias"):
                        teammate_state["phase1_head.fc_final.bias"] = v
                        head_mapped_keys.add("bias")
                    elif clean_k.startswith("classifier.") and not head_mapped_keys:
                        # Fallback for MLP teammates
                        new_k = f"phase1_head.{clean_k.split('.', 1)[1]}"
                        teammate_state[new_k] = v

                # Perform surgical load
                current_model_state = self._model.state_dict()
                final_teammate_state = {k: v.to(target_device) for k, v in teammate_state.items() 
                                       if k in current_model_state and v.shape == current_model_state[k].shape}
                
                self._model.load_state_dict(final_teammate_state, strict=False)
                
                def teammate_forward(inst, x, *args, **kwargs):
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        expert = getattr(inst, target_expert)
                        # plantclef: Use return_features=True to bypass CLIP 512-d bottleneck
                        if target_expert == "bioclip":
                            feat = expert(x, return_features=True)
                        else:
                            feat = expert(x)
                            
                        # plantclef: Final Safety Check
                        if feat.shape[1] != inst.phase1_head.fc_final.in_features:
                            # Apply internal projection if it exists in the checkpoint (1280 -> 1024)
                            if hasattr(expert.backbone, 'proj') and expert.backbone.proj is not None:
                                feat = feat @ expert.backbone.proj
                        
                        return inst.phase1_head(feat)
                
                import types
                self._model.forward = types.MethodType(teammate_forward, self._model)
                logger.info(f"[Surgery] Success. {target_expert} is now independent.")
            
            else:
                # Standard Ensemble Loading
                if any(k.startswith('_orig_mod.') for k in state.keys()):
                    state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
                
                model_state = self._model.state_dict()
                new_state = {}
                remap_rules = {
                    'bioclip.model.visual.': 'bioclip.backbone.',
                    'bioclip.backbone.':     'bioclip.backbone.',
                    'dinov2.backbone.':      'dinov3.backbone.',
                    'convnext.backbone.':    'convnext.backbone.',
                    'proj_grouped.0.':       'proj_linear.',
                    'proj_grouped.1.':       'proj_ln.',
                }

                for k, v in state.items():
                    remapped_k = k
                    for old_p, new_p in remap_rules.items():
                        if k.startswith(old_p):
                            remapped_k = k.replace(old_p, new_p)
                            break
                    if remapped_k in model_state and v.shape == model_state[remapped_k].shape:
                        new_state[remapped_k] = v.to(target_device)

                self._model.load_state_dict(new_state, strict=False)
                logger.info(f"RESTORED {len(new_state)} ensemble layers on {target_device}.")

        self._model.eval()

    def get_model(self) -> nn.Module:
        return self._model

    @property
    def num_classes(self) -> int:
        return self._num_classes


class GenericAdapter(BaseModelAdapter):
    """
    Adapter for any pre-loaded ``nn.Module``.

    Use this when you already have a model instance and just want to plug it
    into the inference pipeline.

    Parameters
    ----------
    model : nn.Module
        An ``nn.Module`` whose ``forward`` accepts a ``(B, 3, H, W)``
        float tensor and returns ``(B, C)`` logits.
    num_classes_ : int
        Number of output logits.
    device : str, optional
        Device to move the model to.
    """

    def __init__(self, model: nn.Module, num_classes_: int, device: str = "cpu") -> None:
        self._model = model.to(device).eval()
        self._num_classes = num_classes_

    def get_model(self) -> nn.Module:
        """
        Return the underlying ``nn.Module`` in eval mode.

        Returns
        -------
        nn.Module
            The model in evaluation mode.
        """
        return self._model

    @property
    def num_classes(self) -> int:
        """
        Number of output classes (logit dimension).

        Returns
        -------
        int
            Number of classes.
        """
        return self._num_classes


# ---------------------------------------------------------------------------
# Helpers for Teammate Surgery
# ---------------------------------------------------------------------------

class SimpleLinearHead(nn.Module):
    """Minimal linear head for single-backbone teammate models."""
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.fc_final = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Match input precision to model weights
        return self.fc_final(x.to(dtype=self.fc_final.weight.dtype))


def load_model_adapter(
    checkpoint_path: Path,
    cfg: ModelConfig,
    adapter_class: type[BaseModelAdapter] | None = None,
    input_res_override: Optional[int] = None,
) -> BaseModelAdapter:
    """
    Factory: load a model checkpoint and return a :class:`BaseModelAdapter`.

    If ``adapter_class`` is ``None``, :class:`PlantEnsembleAdapter` is used.

    Parameters
    ----------
    checkpoint_path : Path
        Path to the saved checkpoint.
    cfg : ModelConfig
        Model configuration (num_classes, device, etc.).
    adapter_class : type[BaseModelAdapter] | None, optional
        Optional custom adapter subclass.
    input_res_override : int | None, optional
        Override the input resolution for this specific model.

    Returns
    -------
    BaseModelAdapter
        Initialised adapter with the model in eval mode.
    """
    cls = adapter_class or PlantEnsembleAdapter
    res = input_res_override if input_res_override is not None else cfg.input_size
    return cls(  # type: ignore[call-arg]
        checkpoint_path=checkpoint_path,
        num_classes_=cfg.num_classes,
        input_res=res,
        device=cfg.device,
        bioclip_name=cfg.bioclip_name,
        dinov3_name=cfg.vit_name,
        convnext_name=cfg.convnext_name,
        use_region_features=cfg.use_region_features,
        use_zero_shot=cfg.use_zero_shot,
        zero_shot_path=cfg.zero_shot_path,
        zero_shot_weight=cfg.zero_shot_weight,
        zero_shot_temp=cfg.zero_shot_temp,
    )


# ---------------------------------------------------------------------------
# Tile dataset (for DataLoader-based batching)
# ---------------------------------------------------------------------------

class _TileDataset(Dataset):
    """
    Minimal PyTorch Dataset that preprocesses tile images into tensors.

    Parameters
    ----------
    tiles : list[tuple[TileSpec, Image.Image]]
        List of ``(TileSpec, PIL.Image)`` pairs.
    input_size : int
        Square resize target (px).
    mean : list[float]
        Normalisation mean (per channel).
    std : list[float]
        Normalisation std (per channel).
    """

    def __init__(
        self,
        tiles: list[tuple[TileSpec, Image.Image]],
        input_size: int,
        mean: list[float],
        std: list[float],
        interpolation: str = "bilinear",
        use_retinex: bool = False,
        retinex_sigmas: tuple = (15.0, 80.0, 250.0),
    ) -> None:
        self.tiles = tiles
        self.input_size = input_size
        self._mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self._std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        self._use_retinex = use_retinex
        self._retinex_sigmas = retinex_sigmas

        # Map string to PIL interpolation modes
        self._interp_map = {
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "lanczos": Image.LANCZOS
        }
        self._interp = self._interp_map.get(interpolation, Image.BILINEAR)

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """
        Return a preprocessed tile tensor and its index.

        The index is returned so the caller can match tensors back to specs.

        Parameters
        ----------
        idx : int
            Index of the tile to retrieve.

        Returns
        -------
        tuple[torch.Tensor, int]
            A tuple containing (preprocessed_tile_tensor, original_index).
        """
        _, img = self.tiles[idx]
        tensor = self._preprocess(img)
        return tensor, idx

    def _preprocess(self, img: Image.Image) -> torch.Tensor:
        """
        Preprocess a PIL image into a torch tensor.
        """
        if img.mode != "RGB":
            img = img.convert("RGB")
        # plantclef: Default to Lanczos for domain shift resilience
        img = img.resize((self.input_size, self.input_size), self._interp)
        arr = np.asarray(img, dtype=np.float32) / 255.0          # (H, W, 3)

        # Radiative transfer illumination normalization (Multi-Scale Retinex)
        if self._use_retinex:
            arr = _retinex_normalize(arr, sigmas=self._retinex_sigmas)

        tensor = torch.from_numpy(arr).permute(2, 0, 1)           # (3, H, W)
        tensor = (tensor - self._mean) / self._std
        return tensor


# ---------------------------------------------------------------------------
# Model runner
# ---------------------------------------------------------------------------

class ModelRunner:
    """
    Runs batched inference on tiles using a loaded model.
    Restored with EXTREME_MODE support for 96GB Blackwell nodes.

    Parameters
    ----------
    adapter : BaseModelAdapter
        A :class:`BaseModelAdapter` providing the model.
    cfg : ModelConfig
        Model configuration controlling device, batch size, etc.
    """

    def __init__(self, adapter: BaseModelAdapter, cfg: ModelConfig, input_res_override: Optional[int] = None) -> None:
        self.cfg = cfg
        self._device = _resolve_device(cfg.device)
        # Model is already on device and in eval mode via adapter
        self._model = adapter.get_model()
        self._num_classes = adapter.num_classes
        self.input_size = input_res_override if input_res_override is not None else cfg.input_size

        # Calibration Buffers
        self._logit_adj = None
        self._genus_ids = None

        # Extreme Mode Detection
        from src import config as _scfg
        self.extreme = getattr(_scfg, 'EXTREME_MODE', False)
        
        # plantclef: Ensure the model is in eval mode before any optimization
        # This handles submodules that might have been loaded lazily
        self._model.eval()
        
        # plantclef: SOTA Inference Optimization
        # Use torchao for Blackwell INT8/FP8 Quantization
        if self.cfg.device == "cuda" and not getattr(self, "is_teammate", False):
            try:
                import torchao
                from torchao.quantization import quantize_, int8_weight_only
                logger.info("[plantclef] Applying Blackwell-optimized INT8 Weight-Only Quantization via torchao...")
                quantize_(self._model, int8_weight_only())
            except ImportError:
                logger.warning("torchao not installed. Skipping INT8 Quantization.")

        # Use torch.compile to fuse kernels and enable automatic CUDA Graphs
        if self.cfg.use_compile:
            compile_mode = "default" if getattr(self, "is_teammate", False) else "max-autotune"
            logger.info(f"[plantclef] Compiling model (Mode: {compile_mode})...")
            
            try:
                if not getattr(self, "is_teammate", False):
                    import torch_tensorrt
                    self._model = torch.compile(self._model, mode=compile_mode, backend="tensorrt", dynamic=False)
                else:
                    self._model = torch.compile(self._model, mode=compile_mode, dynamic=False)
            except ImportError:
                self._model = torch.compile(self._model, mode="reduce-overhead" if compile_mode == "max-autotune" else "default", dynamic=False)

        if cfg.logit_adj_path:
            adj_path = Path(cfg.logit_adj_path)
            if adj_path.exists():
                self._logit_adj = np.load(adj_path).astype(np.float32)
                logger.info("Loaded logit adjustments from %s", adj_path)
                
                # If we have logit_adj, we likely need the taxonomic priors too
                genus_path = "/workspace/plantclef/processed/genus_ids.pt"
                if os.path.exists(genus_path):
                    # Load on CPU first
                    self._genus_ids = torch.load(genus_path, map_location='cpu').int().to(self._device)
                    logger.info("Loaded genus IDs for taxonomic calibration.")
            else:
                logger.warning("Logit adjustment file not found at %s", adj_path)

        # Extreme Override: Use maximum batch size for inference
        self.inference_batch = 256 if self.extreme else cfg.batch_size
        if self.extreme:
            logger.info("[Extreme] Overriding Inference Batch to 256 (Saturating 96GB VRAM)")

        logger.info(
            "ModelRunner: device=%s, batch_size=%d, extreme=%s, num_classes=%d",
            self._device,
            self.inference_batch,
            self.extreme,
            self._num_classes,
        )

    @torch.no_grad()
    def select_informative_tiles(
        self,
        tiles: list[tuple[TileSpec, Image.Image]],
        k: int = 20,
        scan_res: int = 224
    ) -> list[tuple[TileSpec, Image.Image]]:
        """
        Greedy Submodular Maximization for informative tile selection.
        Uses a low-res 'Quick-Scan' pass to select the best k tiles.
        """
        if len(tiles) <= k:
            return tiles

        # 1. Quick-Scan (Low Res)
        # Process all tiles at 224px to get rough species coverage
        dataset = _TileDataset(
            tiles=tiles,
            input_size=scan_res,
            mean=self.cfg.norm_mean,
            std=self.cfg.norm_std,
            interpolation=self.cfg.interpolation,
            use_retinex=getattr(self.cfg, 'use_retinex', False),
            retinex_sigmas=tuple(getattr(self.cfg, 'retinex_sigmas', (15.0, 80.0, 250.0))),
        )
        loader = DataLoader(dataset, batch_size=64, num_workers=0)
        
        all_probs = []
        for batch, _ in loader:
            batch = batch.to(self._device, dtype=torch.bfloat16)
            logits = self._model(batch)
            if isinstance(logits, tuple): logits = logits[0]
            all_probs.append(torch.sigmoid(logits).cpu())
        
        probs = torch.cat(all_probs).numpy() # [N, C]
        
        # 2. Greedy Selection
        # Objective: Maximize species coverage F(S) = sum_c max_{i in S} P(i, c)
        selected_indices = []
        current_max_probs = np.zeros(self._num_classes)
        remaining_indices = list(range(len(tiles)))
        
        for _ in range(k):
            # Calculate marginal gain for each remaining tile
            # G(i) = sum_c max(0, P(i, c) - current_max_probs(c))
            gains = np.sum(np.maximum(0, probs[remaining_indices] - current_max_probs), axis=1)
            
            best_idx_in_remaining = np.argmax(gains)
            best_global_idx = remaining_indices[best_idx_in_remaining]
            
            selected_indices.append(best_global_idx)
            current_max_probs = np.maximum(current_max_probs, probs[best_global_idx])
            remaining_indices.pop(best_idx_in_remaining)
            
            # Optimization: Stop early if gain is negligible
            if gains[best_idx_in_remaining] < 0.01:
                break
                
        return [tiles[i] for i in selected_indices]

    @torch.no_grad()
    def predict(
        self,
        tiles: list[tuple[TileSpec, Image.Image]],
        use_tta: bool = True,
        max_tiles: Optional[int] = None
    ) -> list[TilePrediction]:
        """
        Run inference with optional Vectorized TTA and Submodular Selection.
        """
        if not tiles:
            return []

        # plantclef: Submodular Optimization
        if max_tiles and len(tiles) > max_tiles:
            logger.info(f"[Optim] Submodular selection: Reducing {len(tiles)} -> {max_tiles} tiles.")
            tiles = self.select_informative_tiles(tiles, k=max_tiles)

        dataset = _TileDataset(
            tiles=tiles,
            input_size=self.input_size,
            mean=self.cfg.norm_mean,
            std=self.cfg.norm_std,
            interpolation=self.cfg.interpolation,
            use_retinex=getattr(self.cfg, 'use_retinex', False),
            retinex_sigmas=tuple(getattr(self.cfg, 'retinex_sigmas', (15.0, 80.0, 250.0))),
        )
        loader = DataLoader(
            dataset,
            batch_size=self.inference_batch,
            num_workers=0,  # Prevent nested multiprocessing deadlocks
            pin_memory=self._device.type == "cuda",
            drop_last=False,
        )

        all_logits: list[np.ndarray] = [None] * len(tiles)  # type: ignore[list-item]
        m_dtype = torch.bfloat16 if self.extreme else torch.float32
        amp_ctx = torch.amp.autocast(device_type=self._device.type, dtype=m_dtype)

        # plantclef: CUDA Graph Buffers (Persistent across batches)
        if not hasattr(self, "_cuda_graphs"):
            self._cuda_graphs = {} # Dict of {batch_size: graph}
            self._static_inputs = {}
            self._static_outputs = {}

        for batch_tensors, batch_indices in loader:
            batch_tensors = batch_tensors.to(self._device, non_blocking=True).to(m_dtype)
            curr_bs = batch_tensors.shape[0]
            
            with amp_ctx:
                # Base Forward Pass (Standard Path)
                # plantclef: Manual CUDA Graph capture is disabled as it is incompatible with 
                # DINOv3's dynamic RoPE generation. Use torch.compile for graph acceleration.
                use_manual_graph = False

                if use_manual_graph:
                    if curr_bs not in self._cuda_graphs:
                        logger.info(f"[Optim] Capturing CUDA Graph for Batch Size {curr_bs}...")
                        s = torch.cuda.Stream()
                        s.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(s):
                            for _ in range(3):
                                _ = self._model(batch_tensors)
                        torch.cuda.current_stream().wait_stream(s)
                        
                        g = torch.cuda.CUDAGraph()
                        self._static_inputs[curr_bs] = torch.empty_like(batch_tensors)
                        with torch.cuda.graph(g):
                            self._static_outputs[curr_bs] = self._model(self._static_inputs[curr_bs])
                            if isinstance(self._static_outputs[curr_bs], tuple):
                                self._static_outputs[curr_bs] = self._static_outputs[curr_bs][0]
                        self._cuda_graphs[curr_bs] = g
                    
                    self._static_inputs[curr_bs].copy_(batch_tensors)
                    self._cuda_graphs[curr_bs].replay()
                    base_logits = self._static_outputs[curr_bs]
                else:
                    base_logits = self._model(batch_tensors)
                    if isinstance(base_logits, tuple): base_logits = base_logits[0]
                
                logits = base_logits.clone()
                
                # Entropy-Gated TTA
                if use_tta:
                    probs = torch.sigmoid(base_logits)
                    max_probs = probs.max(dim=1)[0]
                    uncertain_mask = max_probs < 0.9
                    
                    if uncertain_mask.any():
                        uncertain_indices = torch.nonzero(uncertain_mask, as_tuple=True)[0]
                        uncertain_tensors = batch_tensors[uncertain_indices]
                        b_u, c, h, w = uncertain_tensors.shape
                        
                        tta_batch = torch.cat([
                            torch.flip(uncertain_tensors, [3]), 
                            torch.flip(uncertain_tensors, [2]), 
                            torch.rot90(uncertain_tensors, 2, [2, 3])
                        ], dim=0)
                        
                        tta_logits = self._model(tta_batch)
                        if isinstance(tta_logits, tuple): tta_logits = tta_logits[0]
                        
                        tta_logits = tta_logits.view(3, b_u, -1).mean(0)
                        
                        # Average with base logits
                        fused_logits = (base_logits[uncertain_indices] + tta_logits) / 2.0
                        logits[uncertain_indices] = fused_logits
            
            logits_np = logits.detach().float().cpu().numpy()
            for local_i, global_i in enumerate(batch_indices.tolist()):
                all_logits[global_i] = logits_np[local_i]

        # Explicitly clear loader to release file descriptors immediately
        del loader

        results: list[TilePrediction] = []
        all_logits_np = np.stack(all_logits)
        
        # High-Speed Vectorized Probability Calculation (p = sigmoid(logits + log_prior))
        adj_logits = all_logits_np
        if self._logit_adj is not None:
            adj_logits = all_logits_np + self._logit_adj
        
        # Log-stable Sigmoid
        all_probs = 1.0 / (1.0 + np.exp(-np.clip(adj_logits, -15, 15)))
        
        for i, (spec, _) in enumerate(tiles):
            results.append(
                TilePrediction(
                    image_id=spec.image_id,
                    tile_id=spec.tile_id,
                    tile_spec=spec,
                    logits=all_logits[i],
                    probs=all_probs[i],
                    veg_weight=spec.veg_weight,
                )
            )

        return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_device(device_str: str) -> torch.device:
    """
    Resolve a device string, falling back to CPU if CUDA is unavailable.

    Parameters
    ----------
    device_str : str
        The requested device string (e.g., "cuda", "cpu").

    Returns
    -------
    torch.device
        The resolved PyTorch device.
    """
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA not available - falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_str)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable sigmoid.

    Parameters
    ----------
    x : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Array after applying sigmoid activation.
    """
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


class _nullcontext:
    """
    Minimal no-op context manager (backport of contextlib.nullcontext).
    """

    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: Any) -> bool:
        return False
