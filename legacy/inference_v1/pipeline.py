"""Inference pipeline — owned exclusively by the inference phase.

Orchestrates: image loading → tiling → batched multi-resolution inference
→ logit-level ensemble → postprocessing → submission CSV.

Each checkpoint is loaded into its own InferenceEnsemble at its native
training resolution. Predictions from all available checkpoints are summed
at the logit level, so student@224 and teacher@512 contribute correctly
without any pos_embed shape gymnastics.
"""
from __future__ import annotations
import csv
import os
from pathlib import Path
from typing import Dict, Iterator, List, Tuple, Counter
from collections import Counter as _Counter

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from .config import InferenceConfig
from .model  import InferenceEnsemble, LegacyStandaloneModel
from .ttt    import TestTimeAdaptor


# ── Tile generation ──────────────────────────────────────────────────────────


# ── Vegetation Filter (Winner Recipe) ──────────────────────────────────────────

def _exg_vegetation_mask(arr: np.ndarray, exg_thresh: float = 20.0) -> np.ndarray:
    """ExG = 2G - R - B. Works on [0, 1] float32 as well (scaled internally)."""
    # Scale to [0, 255] for standard ExG tuning
    r = arr[0] * 255.0
    g = arr[1] * 255.0
    b = arr[2] * 255.0
    exg = 2 * g - r - b
    return (exg > exg_thresh) & (g > r) & (g > b)

def _filter_tiles(tiles: List[torch.Tensor], min_frac: float = 0.15) -> List[torch.Tensor]:
    if min_frac <= 0 or not tiles:
        return tiles
    
    fracs = []
    for t in tiles:
        mask = _exg_vegetation_mask(t.numpy())
        fracs.append(mask.mean())
    
    kept = [t for t, f in zip(tiles, fracs) if f >= min_frac]
    if not kept:
        # Fallback: take top 25% most green tiles
        idx = np.argsort(fracs)[-max(1, len(tiles)//4):]
        kept = [tiles[i] for i in idx]
    return kept

def _generate_tiles(
    img: torch.Tensor,          # [3, H, W] float32 in [0,1]
    tile_size: int,
    overlap: float,
    scales: List[float],
) -> List[torch.Tensor]:
    """Returns a flat list of [3, tile_size, tile_size] crops."""
    tiles: List[torch.Tensor] = []
    stride = max(1, int(tile_size * (1.0 - overlap)))

    for scale in scales:
        h_orig, w_orig = img.shape[1], img.shape[2]
        if scale != 1.0:
            new_h = max(tile_size, int(h_orig * scale))
            new_w = max(tile_size, int(w_orig * scale))
            scaled = F.interpolate(img.unsqueeze(0), size=(new_h, new_w),
                                   mode="bilinear", align_corners=False).squeeze(0)
        else:
            scaled = img

        H, W = scaled.shape[1], scaled.shape[2]

        if H <= tile_size and W <= tile_size:
            tile = F.interpolate(scaled.unsqueeze(0), size=(tile_size, tile_size),
                                 mode="bilinear", align_corners=False).squeeze(0)
            tiles.append(tile)
            continue

        for y in range(0, H - tile_size + 1, stride):
            for x in range(0, W - tile_size + 1, stride):
                tiles.append(scaled[:, y: y + tile_size, x: x + tile_size])
        tiles.append(scaled[:, H - tile_size:, W - tile_size:])

    return tiles


# ── Image loading ─────────────────────────────────────────────────────────────

def _retinex_normalize(arr: np.ndarray, sigmas=(15.0, 80.0, 250.0)) -> np.ndarray:
    """Multi-Scale Retinex: subtract Gaussian-blurred log-illumination at multiple
    scales to flatten shadow/highlight gradients. Gives the model a more
    illumination-invariant view, which helps for forest-canopy quadrats with
    strong dappled light. arr is HxWx3 float32 in [0,1].

    Source: src/inference/model_runner.py:_retinex_normalize."""
    from scipy.ndimage import gaussian_filter
    log_arr = np.log1p(arr * 255.0)
    msr = np.zeros_like(log_arr)
    for s in sigmas:
        # Per-channel gaussian blur of log-image, subtract from log-image.
        blurred = np.stack([gaussian_filter(log_arr[..., c], sigma=s) for c in range(3)], axis=-1)
        msr += (log_arr - blurred)
    msr /= len(sigmas)
    # Stretch to [0, 1] per-image
    lo, hi = msr.min(), msr.max()
    if hi - lo < 1e-6:
        return arr
    return ((msr - lo) / (hi - lo)).astype(np.float32)


def _load_image(path: str, use_retinex: bool = False) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    if use_retinex:
        arr = _retinex_normalize(arr)
    return torch.from_numpy(arr).permute(2, 0, 1)


def _iter_test_images(cfg: InferenceConfig) -> Iterator[Tuple[str, str]]:
    if os.path.exists(cfg.test_csv):
        import csv as _csv
        with open(cfg.test_csv) as f:
            reader = _csv.DictReader(f, delimiter=";")
            for row in reader:
                # PlantCLEF 2024-2026 CSVs use a single id column (quadrat_id, plot_id,
                # observation_id, or image_id); image filename is usually the id + .jpg.
                img_id = (row.get("quadrat_id") or row.get("plot_id")
                          or row.get("observation_id") or row.get("image_id") or "")
                filename = (row.get("filename") or row.get("file_name")
                            or row.get("image_name") or (f"{img_id}.jpg" if img_id else ""))
                if not filename:
                    continue
                path = os.path.join(cfg.img_dir, filename)
                if os.path.exists(path):
                    yield img_id, path
    else:
        for p in sorted(Path(cfg.img_dir).rglob("*")):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                yield p.stem, str(p)


# ── Aggregation (within-image, across tiles) ─────────────────────────────────

def _aggregate(tile_logits: torch.Tensor, method: str) -> torch.Tensor:
    if method == "mean":
        return tile_logits.mean(0)
    elif method == "max":
        return tile_logits.max(0).values
    elif method == "logsumexp":
        # log(sum(exp(z))) / N is a common robust estimator
        return torch.logsumexp(tile_logits, dim=0) - torch.log(torch.tensor(tile_logits.shape[0]))
    else:                                        # conf_weighted
        probs   = torch.softmax(tile_logits, dim=1)
        weights = probs.max(1).values.unsqueeze(1)
        return (tile_logits * weights).sum(0) / weights.sum(0).clamp(min=1e-8)


# ── Postprocessing ────────────────────────────────────────────────────────────

def _load_pav_solver(cfg: InferenceConfig):
    """Hierarchical isotonic regression: smooths probs so P(species)≤P(genus)≤P(family).
    Uses src/inference/pav_solver.py + data/pav_taxonomy_tree.json."""
    if not cfg.use_pav_calibration:
        return None
    tree_path = "data/pav_taxonomy_tree.json"
    if not os.path.exists(tree_path):
        print(f"[Inference] PAV tree missing at {tree_path}; PAV calibration disabled.")
        return None
    try:
        from src.inference.pav_solver import PAVTreeSolver
        solver = PAVTreeSolver(tree_path)
        print(f"[Inference] PAV-tree calibration loaded.")
        return solver
    except Exception as e:
        print(f"[Inference] WARN: PAV solver init failed ({e}); skipping.")
        return None


def _load_taxon_filter(cfg: InferenceConfig):
    """Rust-backed AC-3 taxonomic consistency filter. Zeroes scores for species
    that violate genus/family/bioclim constraints.
    Uses data_auditor.TaxonomicFilter + the four data/*.json files."""
    if not cfg.use_taxon_filter:
        return None
    paths = {
        "graph":   "data/taxonomic_graph.json",
        "s2g":     "data/species_to_genus.json",
        "g2f":     "data/genus_to_family.json",
        "bioclim": "data/bioclim_data.json",
    }
    missing = [k for k, p in paths.items() if not os.path.exists(p)]
    if missing:
        print(f"[Inference] Taxon filter missing {missing}; AC-3 disabled.")
        return None
    try:
        import json as _json
        import data_auditor
        with open(paths["graph"])   as f: adj     = _json.load(f)
        with open(paths["s2g"])     as f: s2g     = _json.load(f)
        with open(paths["g2f"])     as f: g2f     = _json.load(f)
        with open(paths["bioclim"]) as f: bioclim = _json.load(f)

        # ORACLE: Align to model's num_classes (e.g. 7806)
        if len(adj) < cfg.num_classes:
            for i in range(len(adj), cfg.num_classes):
                adj.append([i])
        if len(s2g) < cfg.num_classes:
            # Fallback to a "Misc" genus (0) for new species
            s2g.extend([0] * (cfg.num_classes - len(s2g)))

        tf = data_auditor.TaxonomicFilter(adj, s2g, g2f, bioclim)
        print(f"[Inference] Rust TaxonomicFilter (AC-3) loaded.")
        return tf
    except Exception as e:
        print(f"[Inference] WARN: TaxonomicFilter init failed ({e}); skipping.")
        return None


def _load_sam_generator(cfg: InferenceConfig):
    """Lazily load SAM-ViT-H. Returns the SamAutomaticMaskGenerator or None
    on any failure (missing weights, missing package). Only called when
    cfg.use_sam_tiling=True."""
    if not cfg.use_sam_tiling:
        return None
    if not os.path.exists(cfg.sam_weights_path):
        print(f"[Inference] SAM weights missing at {cfg.sam_weights_path}; SAM tiling disabled.")
        return None
    try:
        import torch as _torch
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
        sam = sam_model_registry[cfg.sam_model_type](checkpoint=cfg.sam_weights_path).to(device)
        # Lower-overhead defaults: reduce point grid for speed on quadrats
        gen = SamAutomaticMaskGenerator(
            sam,
            points_per_side=16,         # default 32; halving cuts time ~4×
            pred_iou_thresh=0.85,
            stability_score_thresh=0.90,
            min_mask_region_area=cfg.sam_min_area,
        )
        print(f"[Inference] SAM {cfg.sam_model_type} loaded for instance-mask tiling.")
        return gen
    except Exception as e:
        print(f"[Inference] WARN: SAM init failed ({e}); SAM tiling disabled.")
        return None


def _load_faiss_engine(cfg: InferenceConfig):
    """Lazily build the FAISS retrieval index from species prototypes.
    Returns (faiss_index, prototypes_torch) or (None, None)."""
    if not cfg.use_faiss_retrieval:
        return None, None
    if not os.path.exists(cfg.faiss_prototypes_path):
        print(f"[Inference] FAISS prototypes missing at {cfg.faiss_prototypes_path}; retrieval disabled.")
        return None, None
    try:
        import faiss
        protos = torch.load(cfg.faiss_prototypes_path, map_location="cpu", weights_only=False)
        protos_np = protos.numpy().astype(np.float32)
        # Already L2-normalized by the build script, but re-normalize defensively
        faiss.normalize_L2(protos_np)
        idx = faiss.IndexFlatIP(protos_np.shape[1])
        idx.add(protos_np)
        # ORACLE: Explicitly run FAISS on CPU to avoid kernel image errors on B200 (Blackwell)
        placement = "CPU"
        # if hasattr(faiss, "StandardGpuResources"):
        #     try:
        #         res = faiss.StandardGpuResources()
        #         idx = faiss.index_cpu_to_gpu(res, 0, idx)
        #         placement = "GPU"
        #     except Exception:
        #         placement = "CPU (GPU transfer failed)"
        # else:
        #     placement = "CPU"
        print(f"[Inference] FAISS retrieval index built on {placement} "
              f"(α={cfg.faiss_alpha}, fires if max(prob)<{cfg.faiss_low_conf_threshold}). "
              f"Prototypes: {protos_np.shape}")
        return idx, protos_np.shape[1]
    except Exception as e:
        print(f"[Inference] WARN: FAISS init failed ({e}); retrieval disabled.")
        return None, None


def _sam_generate_tiles(img_pil, mask_generator, cfg: InferenceConfig) -> List[torch.Tensor]:
    """SAM-based tiling: detects plant instances and crops each one.
    Falls back to a single full-image tile if SAM finds nothing useful.
    Returns a flat list of [3, H, W] float crops in [0,1]."""
    img_np = np.array(img_pil)
    H_full, W_full = img_np.shape[:2]
    masks = mask_generator.generate(img_np)
    # Sort by area descending; cap at sam_max_instances
    masks = sorted(masks, key=lambda m: m.get("area", 0), reverse=True)[:cfg.sam_max_instances]

    tiles: List[torch.Tensor] = []
    for m in masks:
        x, y, w, h = (int(v) for v in m["bbox"])
        x = max(0, x); y = max(0, y)
        w = max(1, min(w, W_full - x))
        h = max(1, min(h, H_full - y))
        if w * h < cfg.sam_min_area:
            continue
        crop = img_np[y:y+h, x:x+w].astype(np.float32) / 255.0
        tiles.append(torch.from_numpy(crop).permute(2, 0, 1))

    if not tiles:
        # Whole image as a single tile fallback
        arr = img_np.astype(np.float32) / 255.0
        tiles.append(torch.from_numpy(arr).permute(2, 0, 1))
    return tiles


def _load_fw_solver(cfg: InferenceConfig):
    """Frank-Wolfe constrained solver with Island Biogeography prior over the
    species adjacency graph. Returns a sparse selection (top-k species)."""
    if not cfg.use_frank_wolfe:
        return None
    adj_path = "data/taxonomic_graph.json"
    if not os.path.exists(adj_path):
        print(f"[Inference] FW adjacency missing at {adj_path}; FW disabled.")
        return None
    try:
        from src.inference.frank_wolfe import FrankWolfeSolver
        solver = FrankWolfeSolver(adj_path)
        print(f"[Inference] Frank-Wolfe / Island Biogeography solver loaded (top_k={cfg.fw_top_k}).")
        return solver
    except Exception as e:
        print(f"[Inference] WARN: FW solver init failed ({e}); skipping.")
        return None


def _load_postprocess_state(cfg: InferenceConfig, species_ids: List[str] | None = None):
    """Load Brent's per-class thresholds and conformal q_hat if their files exist.
    Returns (per_class_thresholds: Tensor[num_classes] or None, q_hat: float or None).

    The Brent JSON may be keyed by:
      - "thresholds": flat list (already class-index aligned)
      - flat list                (already class-index aligned)
      - {"<class_idx>": thr}     (string-int keys)
      - {"<species_id>": thr}    (real species IDs — needs species_ids mapping)
    Quietly falls back to None if files missing or load fails."""
    per_class_thr = None
    q_hat = None

    if cfg.brent_thresholds_path and os.path.exists(cfg.brent_thresholds_path):
        try:
            import json
            with open(cfg.brent_thresholds_path) as f:
                data = json.load(f)
            arr = None
            if isinstance(data, dict) and "thresholds" in data:
                arr = data["thresholds"]
            elif isinstance(data, list):
                arr = data
            elif isinstance(data, dict):
                # Could be class-idx keys or species-id keys. Detect by checking
                # whether keys map cleanly to class indices.
                sample_keys = [k for k in list(data)[:10] if isinstance(k, str)]
                all_small_ints = all(k.isdigit() and int(k) < cfg.num_classes for k in sample_keys)
                if all_small_ints:
                    # Class-index keyed
                    arr = [float(data.get(str(i), cfg.threshold)) for i in range(cfg.num_classes)]
                elif species_ids and len(species_ids) == cfg.num_classes:
                    # Species-ID keyed → look up via species_ids list
                    arr = [float(data.get(species_ids[i], cfg.threshold))
                           for i in range(cfg.num_classes)]
                else:
                    print(f"[Inference] WARN: Brent JSON keyed by species IDs but no "
                          f"species_ids mapping available; skipping.")
            if arr is not None and len(arr) == cfg.num_classes:
                per_class_thr = torch.tensor(arr, dtype=torch.float32)
                n_real = sum(1 for v in arr if v != cfg.threshold)
                print(f"[Inference] Loaded Brent's per-class thresholds: "
                      f"{n_real}/{cfg.num_classes} class-specific, rest = global ({cfg.threshold}).")
        except Exception as e:
            print(f"[Inference] WARN: failed to load Brent thresholds ({e}); skipping.")

    if cfg.conformal_calibration_path and os.path.exists(cfg.conformal_calibration_path):
        try:
            state = torch.load(cfg.conformal_calibration_path, map_location="cpu", weights_only=False)
            if isinstance(state, dict):
                q_hat = float(state.get("q_hat") or state.get("threshold") or 0.0) or None
            elif isinstance(state, (int, float)):
                q_hat = float(state)
            if q_hat is not None:
                print(f"[Inference] Loaded conformal q_hat = {q_hat:.4f} (alpha={cfg.conformal_alpha}).")
        except Exception as e:
            print(f"[Inference] WARN: failed to load conformal calibration ({e}); skipping.")

    return per_class_thr, q_hat



def _logit_adjustment(logits: torch.Tensor, counts_path: str, mapping_path: str, tau: float = 1.0) -> torch.Tensor:
    import pandas as pd
    try:
        counts_df = pd.read_csv(counts_path)
        mapping_df = pd.read_csv(mapping_path, header=None)
        
        mapping_ids = mapping_df[0].tolist()
        counts_dict = dict(zip(counts_df['species_id'], counts_df['count']))
        
        # Build prior vector in mapping order
        priors = []
        for sid in mapping_ids:
            # Missing IDs get count 1 (neutral)
            count = counts_dict.get(int(sid), 1)
            priors.append(float(count))
            
        priors = torch.tensor(priors, device=logits.device, dtype=torch.float32)
        priors = priors / priors.sum()
        
        # Logit adjustment: z - tau * log(pi)
        # Note: we use log(pi + epsilon)
        adjusted = logits - tau * torch.log(priors + 1e-9)
        return adjusted
    except Exception as e:
        print(f"[Inference] Logit Adjustment failed: {e}")
        return logits

def _postprocess(logits: torch.Tensor, cfg: InferenceConfig, species_ids: List[str],
                 return_stats=False,
                 per_class_thr: torch.Tensor | None = None,
                 q_hat: float | None = None,
                 pav_solver=None,
                 taxon_filter=None,
                 fw_solver=None) -> List[str]:
    """Postprocessing chain:
       sigmoid → PAV-tree smoothing → AC-3 taxon filter →
       [Frank-Wolfe | conformal | Brent | global threshold] → top-k → species IDs"""
    stats = {"pav": 0, "ac3": 0, "fw": 0, "f1": 0.0}
    # ORACLE Day 1: Logit Adjustment
    # Counts path from config
    counts_path = "/workspace/plantclef/processed/species_train_counts.csv"
    mapping_path = cfg.species_mapping
    logits = _logit_adjustment(logits, counts_path, mapping_path, tau=cfg.logit_adj_tau)
    
    logits = torch.nan_to_num(logits, nan=0.0)
    probs = torch.sigmoid(logits)
    stats["f1"] = float(probs.max().item())

    # ── Hierarchical PAV-tree smoothing ────────────────────────────────────
    # Enforces P(species) ≤ P(genus) ≤ P(family). Stabilizes uncertain
    # tail species toward their genus/family-level prior.
    if pav_solver is not None:
        try:
            old_p = probs.clone()
            probs = pav_solver.solve(probs)
            stats["pav"] = int((probs != old_p).sum().item())
        except Exception as e:
            print(f"[Inference] WARN: PAV solver failed on this image ({e}); skipping.")

    # ── AC-3 taxonomic / bioclim consistency filter ────────────────────────
    # Zeroes out species that violate genus/family/bioclim constraints
    # given the candidate set (Rust SIMD bitset-based).
    if taxon_filter is not None:
        try:
            scores_np = probs.detach().cpu().numpy().astype(np.float32)
            filtered  = taxon_filter.filter_predictions(scores_np)
            new_p     = torch.from_numpy(np.asarray(filtered)).to(probs.device).float()
            stats["ac3"] = int((probs > 0).sum().item() - (new_p > 0).sum().item())
            probs     = new_p
        except Exception as e:
            print(f"[Inference] WARN: TaxonomicFilter failed on this image ({e}); skipping.")

    # ── Selection ──────────────────────────────────────────────────────────
    # Path A: Frank-Wolfe + Island Biogeography (richness-aware sparse selection)
    if fw_solver is not None:
        try:
            sel = fw_solver.solve(probs.float(), max_iters=5, sparsity_k=cfg.fw_top_k)
            stats["fw"] = int((sel > 1e-3).sum().item())
            above = torch.nonzero(sel > 1e-3).squeeze(-1).tolist()
            if isinstance(above, int): above = [above]
            if not above:
                above = [int(torch.argmax(probs))]
        except Exception as e:
            print(f"[Inference] WARN: FW solver failed on this image ({e}); falling back.")
            fw_solver = None  # disable for this iteration's fallback chain

    if fw_solver is None:
        # Path B: Conformal Adaptive Prediction Sets
        if q_hat is not None and 0 < q_hat < 1:
            sorted_idx = torch.argsort(probs, descending=True)
            sorted_p   = probs[sorted_idx]
            cum        = torch.cumsum(sorted_p / sorted_p.sum().clamp(min=1e-9), dim=0)
            cutoff     = (cum >= q_hat).nonzero(as_tuple=True)[0]
            n_keep     = (cutoff[0].item() + 1) if len(cutoff) else cfg.top_k
            n_keep     = max(cfg.min_predictions, min(n_keep, cfg.top_k))
            above      = sorted_idx[:n_keep].tolist()

        # Path C: Brent's per-class thresholds
        elif per_class_thr is not None:
            above = (probs >= per_class_thr.to(probs.device)).nonzero(as_tuple=True)[0].tolist()
            if len(above) < cfg.min_predictions:
                above = torch.topk(probs, k=cfg.top_k).indices.tolist()

        # Path D: Global threshold fallback
        else:
            above = (probs >= cfg.threshold).nonzero(as_tuple=True)[0].tolist()
            if len(above) < cfg.min_predictions:
                above = torch.topk(probs, k=cfg.top_k).indices.tolist()

    above = above[: cfg.top_k]
    above = sorted(above, key=lambda i: float(probs[i]), reverse=True)
    final_preds = [species_ids[i] for i in above if i < len(species_ids)]

    if return_stats:
        return final_preds, stats
    return final_preds


# ── Checkpoint discovery + auto-fallback ─────────────────────────────────────

def _resolve_models(cfg: InferenceConfig, device: torch.device) -> List[Tuple[InferenceEnsemble, int]]:
    """Returns list of (model, native_resolution). Honors auto-fallback to teachers."""
    entries  = list(cfg.checkpoints)
    present  = [e for e in entries if os.path.exists(e["path"])]
    missing  = [e for e in entries if not os.path.exists(e["path"])]

    student_entries = [e for e in entries if "student" in e["path"]]
    student_present = [e for e in student_entries if os.path.exists(e["path"])]
    teacher_present = [e for e in present if "student" not in e["path"]]

    if not student_present and student_entries:
        msg = f"[Inference] Student checkpoint(s) missing → falling back to {len(teacher_present)} teacher model(s)."
        if cfg.require_student:
            raise FileNotFoundError(msg.replace("falling back", "ABORT (require_student=true); could fall back"))
        print(msg)
    if missing:
        for e in missing:
            print(f"[Inference]   skip (not found): {e['path']}")
    if not present:
        raise FileNotFoundError("[Inference] No checkpoints exist on disk; nothing to load.")

    models: List[Tuple[nn.Module, int]] = []
    for e in present:
        # ORACLE: Detect God-Tier / Legacy Standalone models
        is_legacy = any(x in e["path"] for x in ["best.pt", "plantnet_v2", "Exp008", "phase_b_best"])
        
        if is_legacy:
            m = LegacyStandaloneModel(
                checkpoint_path = e["path"],
                resolution      = e["resolution"],
                device          = device,
            )
        else:
            m = InferenceEnsemble.from_checkpoint(
                checkpoint_path = e["path"],
                resolution      = e["resolution"],
                num_classes     = cfg.num_classes,
                bioclip_name    = cfg.bioclip,
                dinov3_name     = cfg.dinov3,
                convnext_name   = cfg.convnext,
                device          = device,
            )
            m.disable_logit_standardization = bool(cfg.disable_logit_standardization)
            
        models.append((m, e["resolution"]))
    if cfg.disable_logit_standardization:
        print(f"[Inference] Logit standardization DISABLED for all {len(models)} model(s).")
    return models


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_inference(cfg: InferenceConfig) -> None:
    device = torch.device(f"cuda:{cfg.local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    if cfg.world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", device_id=device if device.type == "cuda" else None)

    # Species mapping
    species_ids: List[str] = []
    if os.path.exists(cfg.species_mapping):
        import csv as _csv
        with open(cfg.species_mapping) as f:
            reader = _csv.reader(f)
            for row in reader:
                if row:
                    species_ids.append(row[0])
    if not species_ids:
        species_ids = [str(i) for i in range(cfg.num_classes)]

    # Models (one per checkpoint, each at its native resolution)
    models = _resolve_models(cfg, device)
    print(f"[Inference] Active ensemble: {len(models)} model(s) "
          f"at resolutions {[r for _, r in models]}")

    # Advanced postprocessing chain — each component falls back gracefully if
    # its required data file is missing.
    per_class_thr, q_hat   = _load_postprocess_state(cfg, species_ids)
    pav_solver             = _load_pav_solver(cfg)
    taxon_filter           = _load_taxon_filter(cfg)
    fw_solver              = _load_fw_solver(cfg)
    sam_generator          = _load_sam_generator(cfg)
    faiss_index, _faiss_dim = _load_faiss_engine(cfg)

    if cfg.use_retinex:
        print(f"[Inference] Retinex illumination normalization ENABLED (sigmas=15/80/250).")

    results: Dict[str, List[str]] = {}
    species_freq: Counter = _Counter()

    # ORACLE: Test-Time Training (TTT) adaptor. Only uses the first model for speed.
    ttt_adaptor = None
    if cfg.use_ttt and models:
        ttt_adaptor = TestTimeAdaptor(models[0][0], steps=cfg.ttt_steps, lr=cfg.ttt_lr)
        print(f"[Inference] TTT Enabled ({cfg.ttt_steps} steps, lr={cfg.ttt_lr}).")

    def _infer_one_image(img: torch.Tensor, img_pil=None, img_id: str = "") -> torch.Tensor:
        """Runs all models on one image; returns aggregated logits [C].
        If FAISS retrieval is active, also boosts logits with prototype similarity.
        If SAM tiling is active, replaces grid tiling with instance-mask crops."""
        # --- TTT Adaptation ---
        ttt_loss = 0.0
        if ttt_adaptor is not None and img_pil is not None:
            ttt_loss = ttt_adaptor.adapt(img_pil, image_id=img_id)

        # Tile generation: SAM-based if enabled, else multi-scale grid.
        if cfg.use_sam_tiling and sam_generator is not None and img_pil is not None:
            tiles = _sam_generate_tiles(img_pil, sam_generator, cfg)
        elif cfg.tiling_enabled:
            tiles = _generate_tiles(img, cfg.tile_size, cfg.tile_overlap, cfg.scales)
        else:
            tiles = [img]
        
        # ORACLE: Apply Vegetation Filter
        if cfg.min_vegetation_frac > 0:
            tiles = _filter_tiles(tiles, cfg.min_vegetation_frac)

        per_model_logits: List[torch.Tensor] = []
        all_query_features: List[torch.Tensor] = []   # for FAISS

        for model, native_res in models:
            # SAM tiles have heterogeneous shapes; resize each individually then stack.
            # ORACLE: B200 Optimized Tile Batching
            all_logits_T = []
            all_feats_T  = []
            for i in range(0, len(tiles), cfg.batch_size):
                batch_tiles = tiles[i : i + cfg.batch_size]
                resized = torch.stack([
                    F.interpolate(t.unsqueeze(0), size=(native_res, native_res),
                                  mode="bilinear", align_corners=False).squeeze(0)
                    for t in batch_tiles
                ]).to(device)
                
                with torch.inference_mode():
                    if faiss_index is not None:
                        l_t, f_t = model(resized, return_features=True)
                        if cfg.use_hflip_tta:
                            l_flipped = model(torch.flip(resized, dims=[3]))
                            l_t = (l_t + l_flipped) / 2.0
                        all_logits_T.append(l_t.float().cpu())
                        all_feats_T.append(f_t.float().cpu())
                    else:
                        l_t = model(resized)
                        if cfg.use_hflip_tta:
                            l_flipped = model(torch.flip(resized, dims=[3]))
                            l_t = (l_t + l_flipped) / 2.0
                        all_logits_T.append(l_t.float().cpu())
            
            logits_T = torch.cat(all_logits_T, dim=0)
            if faiss_index is not None:
                all_query_features.append(torch.cat(all_feats_T, dim=0).mean(dim=0).to(device))
            
            # ORACLE: Slice to 7806 if model has padding (7808)
            aggregated = _aggregate(logits_T.to(device), cfg.aggregation)
            if aggregated.shape[0] > 7806:
                aggregated = aggregated[:7806]
            per_model_logits.append(aggregated)

        # Cross-model logit ensemble
        if cfg.ensemble_weights and len(cfg.ensemble_weights) == len(per_model_logits):
            # Weighted average
            weights = torch.tensor(cfg.ensemble_weights, device=device, dtype=torch.float32)
            weights = weights / weights.sum()
            agg_logits = torch.stack(per_model_logits, dim=0) * weights.unsqueeze(1)
            agg_logits = agg_logits.sum(0)
        else:
            # Simple mean
            agg_logits = torch.stack(per_model_logits, dim=0).mean(0)    # [C]

        # FAISS retrieval boost — only fires for low-confidence predictions.
        if faiss_index is not None and all_query_features:
            probs_for_check = torch.sigmoid(agg_logits)
            if probs_for_check.max().item() < cfg.faiss_low_conf_threshold:
                qf = torch.stack(all_query_features, dim=0).mean(0).cpu().numpy().astype(np.float32)
                qf = qf.reshape(1, -1)
                import faiss as _faiss
                _faiss.normalize_L2(qf)
                k_search = min(cfg.num_classes, faiss_index.ntotal)
                D, I = faiss_index.search(qf, k_search)             # [1, k_search]
                retrieval_scores = torch.zeros_like(agg_logits)
                retrieval_scores[I[0]] = torch.from_numpy(D[0]).to(agg_logits.device).float()
                # Mix: α·logit + (1-α)·retrieval
                agg_logits = cfg.faiss_alpha * agg_logits + (1.0 - cfg.faiss_alpha) * retrieval_scores

        return agg_logits, ttt_loss, 0.0

    n_done = 0
    n_sam_used = 0
    n_faiss_fired = 0

    # ORACLE: Distributed sharding of test images
    all_test_images = list(_iter_test_images(cfg))
    if cfg.limit > 0:
        all_test_images = all_test_images[:cfg.limit]
        if cfg.rank == 0:
            print(f"[Inference] Limit active: processing only first {cfg.limit} images.")

    if cfg.world_size > 1:
        my_images = all_test_images[cfg.rank::cfg.world_size]
        if cfg.rank == 0:
            print(f"[Inference] Sharding {len(all_test_images)} total images -> ~{len(my_images)} per GPU.")
    else:
        my_images = all_test_images

    pbar = None
    if cfg.rank == 0:
        pbar = tqdm(total=len(my_images), desc=f"[Inference] Rank {cfg.rank}")

    for img_id, img_path in my_images:
        try:
            img_pil = Image.open(img_path).convert("RGB")
            arr = np.array(img_pil, dtype=np.float32) / 255.0
            retinex_shift = 0.0
            if cfg.use_retinex:
                old_arr = arr.copy()
                arr = _retinex_normalize(arr)
                retinex_shift = float(np.abs(arr - old_arr).mean())
            img = torch.from_numpy(arr).permute(2, 0, 1)
        except Exception as e:
            print(f"[Inference] Skipping {img_path}: {e}")
            continue
        agg_logits, current_loss, retinex_shift = _infer_one_image(img, img_pil=img_pil, img_id=img_id)
        preds, pp_stats = _postprocess(
            agg_logits.cpu(), cfg, species_ids,
            per_class_thr=per_class_thr, q_hat=q_hat,
            pav_solver=pav_solver, taxon_filter=taxon_filter, fw_solver=fw_solver,
            return_stats=True
        )
        results[img_id] = preds
        species_freq.update(preds)
        n_done += 1
        if pbar is not None:
            # Diversity check: Top-5 Species Share
            top5_share = 0.0
            if n_done > 10 and species_freq:
                top5_share = sum(c for _, c in species_freq.most_common(5)) / max(1, sum(species_freq.values()))

            pbar.set_postfix(
                loss=f"{current_loss:.2f}",
                f1=f"{pp_stats['f1']:.3f}",
                div=f"{top5_share*100:.1f}%",
                pav=pp_stats["pav"],
                ac3=pp_stats["ac3"],
                fw=pp_stats["fw"]
            )
            pbar.update(1)
        elif n_done % 100 == 0:
            print(f"[Inference] GPU {cfg.rank}: {n_done} images processed")
    
    if pbar is not None:
        pbar.close()

    # ORACLE: Gather all results across GPUs
    if cfg.world_size > 1:
        dist.barrier()
        gathered_list = [None] * cfg.world_size
        dist.all_gather_object(gathered_list, results)
        
        # Merge all dictionaries into one on Rank 0
        if cfg.rank == 0:
            merged_results = {}
            for r_dict in gathered_list:
                merged_results.update(r_dict)
            results = merged_results

    # Submission CSV — only Rank 0 writes
    if cfg.rank == 0:
        os.makedirs(os.path.dirname(os.path.abspath(cfg.submission_csv)) or ".", exist_ok=True)
        with open(cfg.submission_csv, "w", newline="") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL, lineterminator="\n")
            writer.writerow(["quadrat_id", "species_ids"])
            for img_id, preds in results.items():
                writer.writerow([img_id, f"[{', '.join(preds)}]"])

        print(f"[Inference] Written {len(results)} predictions → {cfg.submission_csv}")
        
        # ORACLE: Automated Submission Validation
        print(f"\n[Inference] Running automated validation on {cfg.submission_csv}...")
        import subprocess
        try:
            val_cmd = [sys.executable, "scripts/validate_submission.py", cfg.submission_csv]
            subprocess.run(val_cmd, check=True)
        except subprocess.CalledProcessError:
            print(f"[Inference] ⚠ Validation failed for {cfg.submission_csv}. Check the errors above.")
        except Exception as e:
            print(f"[Inference] WARN: Could not run validation script ({e}).")
    
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
