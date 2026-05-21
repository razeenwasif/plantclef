import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
from typing import Optional, Tuple, Union, List, Dict
from .hotspot import HotspotDetector

try:
    import plantclef_ext
    HAS_EXT = True
except ImportError:
    HAS_EXT = False

class CUDASahiEngine:
    """
    Blackwell-optimized inference engine for SAHI.

    Handles CUDA-native tiling, batch processing, fused max-pooling, and
    phenological filtering. Includes Hybrid Hotspot Inference to skip empty tiles.

    Parameters
    ----------
    model : nn.Module
        The trained ensemble model for inference.
    resolution : int, optional
        Target resolution for input tiles (default is 384).
    device : str, optional
        Device to run inference on (default is 'cuda').
    use_hotspots : bool, optional
        Whether to use HotspotDetector to skip empty tiles (default is True).
    bloom_calendar_path : str, optional
        Path to the JSON bloom calendar for filtering (default is None).
    """

    def __init__(self, model: nn.Module, resolution: int = 384, device: str = 'cuda', use_hotspots: bool = True, bloom_calendar_path: Optional[str] = None) -> None:
        self.model = model
        self.resolution = resolution
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        self.use_hotspots = use_hotspots
        if use_hotspots:
            self.hotspot_detector = HotspotDetector().to(device).eval()
        
        # Normalization constants (BioCLIP/DINOv2 standard)
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
        self.std  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)

        # Optional Phenological Filter
        self.pheno_filter = None
        if bloom_calendar_path and os.path.exists(bloom_calendar_path) and HAS_EXT:
            import json
            with open(bloom_calendar_path, 'r') as f:
                calendar = json.load(f)
            self._temp_calendar = calendar
            print("Phenological Filter initialized (Pending index mapping).")

    def set_pheno_mapping(self, species_to_idx: Dict[int, int]) -> None:
        """
        Maps species_id from bloom calendar to model indices.

        Parameters
        ----------
        species_to_idx : Dict[int, int]
            Dictionary mapping species IDs to model output indices.

        Returns
        -------
        None
        """
        if not hasattr(self, '_temp_calendar') or not HAS_EXT: return
        num_classes = len(species_to_idx)
        masks = [4095] * num_classes 
        for s_id, mask in self._temp_calendar.items():
            if int(s_id) in species_to_idx:
                masks[species_to_idx[int(s_id)]] = mask
        self.pheno_filter = plantclef_ext.PhenologicalFilter(masks)
        del self._temp_calendar
        print(f"Phenological Filter mapped for {num_classes} classes.")

    def _predict_batch(self, tiles: torch.Tensor, return_uncertainty: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Internal method to run inference on a batch of tiles.

        Parameters
        ----------
        tiles : torch.Tensor
            Tensor of image tiles of shape (batch_size, 3, H, W).
        return_uncertainty : bool, optional
            Whether to return uncertainty estimation (default is False).

        Returns
        -------
        output : Tuple[torch.Tensor, torch.Tensor]
            A tuple of (probabilities, uncertainty).
        """
        if tiles.shape[2] != self.resolution:
            tiles = F.interpolate(tiles, size=(self.resolution, self.resolution), mode='bilinear')
        
        tiles = (tiles - self.mean) / self.std
        
        # Determine precision based on hardware and config
        from src import config as _cfg
        mixed_precision_dtype = torch.bfloat16
        if getattr(_cfg, 'USE_FP8', False) and hasattr(torch, 'float8_e4m3fn'):
            mixed_precision_dtype = torch.float8_e4m3fn

        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=mixed_precision_dtype):
                if return_uncertainty and hasattr(self.model, 'forward_with_uncertainty'):
                    logits, uncertainty = self.model.forward_with_uncertainty(tiles)
                else:
                    logits = self.model(tiles)
                    uncertainty = torch.zeros(tiles.size(0), device=tiles.device)
            probs = torch.softmax(logits, dim=1)
        return probs, uncertainty

    def predict_image(self, image_path: str, slice_h: int = 512, slice_w: int = 512, overlap: float = 0.2, chunk_size: int = 16, return_uncertainty: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
        """
        Predicts species in a local image file.

        Parameters
        ----------
        image_path : str
            Path to the image file.
        slice_h : int, optional
            Height of each SAHI slice (default is 512).
        slice_w : int, optional
            Width of each SAHI slice (default is 512).
        overlap : float, optional
            Overlap ratio between slices (default is 0.2).
        chunk_size : int, optional
            Number of tiles to process in parallel (default is 16).
        return_uncertainty : bool, optional
            Whether to return uncertainty estimation (default is False).

        Returns
        -------
        output : Union[np.ndarray, Tuple[np.ndarray, float]]
            Final probability distribution or (probs, uncertainty) tuple.
        """
        # Fallback for code that still uses the old method signature
        img_pil = Image.open(image_path).convert('RGB')
        img_tensor = torch.from_numpy(np.array(img_pil)).permute(2, 0, 1).float().to(self.device) / 255.0
        return self.predict_tensor(img_tensor, slice_h, slice_w, overlap, chunk_size, return_uncertainty)

    def predict_tensor(self, img_tensor: torch.Tensor, slice_h: int = 512, slice_w: int = 512, overlap: float = 0.2, chunk_size: int = 16, 
                       return_uncertainty: bool = False, gated_threshold: float = 0.0, month: Optional[int] = None) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
        """
        Runs the full optimized SAHI pipeline on an image tensor.

        Parameters
        ----------
        img_tensor : torch.Tensor
            Input image tensor (C, H, W).
        slice_h : int, optional
            Height of each SAHI slice (default is 512).
        slice_w : int, optional
            Width of each SAHI slice (default is 512).
        overlap : float, optional
            Overlap ratio between slices (default is 0.2).
        chunk_size : int, optional
            Number of tiles to process in parallel (default is 16).
        return_uncertainty : bool, optional
            Whether to return uncertainty estimation (default is False).
        gated_threshold : float, optional
            If global prediction > this, skip tiling (default is 0.0).
        month : int, optional
            Optional month (1-12) for phenological filtering (default is None).

        Returns
        -------
        output : Union[np.ndarray, Tuple[np.ndarray, float]]
            Final probability distribution or (probs, uncertainty) tuple.
        """
        # 1. Global View First (If Gated)
        if gated_threshold > 0.0:
            global_probs, global_uncertainty = self._predict_batch(img_tensor.unsqueeze(0), return_uncertainty=True)
            if global_probs.max() >= gated_threshold:
                u_val = global_uncertainty.mean().item() if isinstance(global_uncertainty, torch.Tensor) else global_uncertainty
                if return_uncertainty:
                    return global_probs.squeeze(0).float().cpu().numpy(), u_val
                else:
                    return global_probs.squeeze(0).float().cpu().numpy()

        # 2. Hotspot Selection (Phase 4 Implementation)
        hotspot_indices = None
        if self.use_hotspots:
            # Identifies tiles containing vegetation to avoid processing bare soil
            hotspot_indices = self.hotspot_detector.get_hotspot_indices(
                img_tensor.unsqueeze(0), slice_h, slice_w, overlap
            )
            if not hotspot_indices:
                # If no hotspots found, process the global view at least
                return self._predict_batch(img_tensor.unsqueeze(0), return_uncertainty=return_uncertainty)[0].squeeze(0).float().cpu().numpy()

        # 3. CUDA-Native Tiling
        if HAS_EXT:
            tiles = plantclef_ext.extract_tiles(img_tensor, slice_h, slice_w, overlap)
            
            # Hybrid Filtering: Only keep tiles with hotspots
            if hotspot_indices is not None:
                tiles = tiles[hotspot_indices]
                
            # Always add global view to the tiles for better context (index 0)
            global_view = F.interpolate(img_tensor.unsqueeze(0), size=(slice_h, slice_w), mode='bilinear')
            tiles = torch.cat([global_view, tiles], dim=0)
        else:
            # Fallback for CPU/Standard PyTorch
            tiles = img_tensor.unsqueeze(0)

        # 4. Process tiles in chunks to manage VRAM
        all_probs = []
        all_uncertainties = []
        for i in range(0, len(tiles), chunk_size):
            p, u = self._predict_batch(tiles[i:i+chunk_size], return_uncertainty=return_uncertainty)
            all_probs.append(p)
            all_uncertainties.append(u)
        
        tile_probs = torch.cat(all_probs, dim=0)
        tile_uncertainties = torch.cat(all_uncertainties, dim=0)

        # 5. Fused CUDA Max Pooling
        if HAS_EXT:
            final_probs_tensor = plantclef_ext.fused_max_pool(tile_probs)
        else:
            final_probs_tensor = torch.max(tile_probs, dim=0)[0]

        # 6. Apply Phenological Filter (Seasonal Suppression)
        if month is not None and self.pheno_filter is not None:
            self.pheno_filter.apply_filter(final_probs_tensor, month)

        final_uncertainty = tile_uncertainties.mean().item()
        final_probs = final_probs_tensor.float().cpu().numpy()

        if return_uncertainty:
            return final_probs, final_uncertainty
        else:
            return final_probs
