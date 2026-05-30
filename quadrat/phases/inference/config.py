"""Inference config — completely standalone, no src.config dependency.

Supports multi-resolution ensemble:
  checkpoints can be a list of strings (legacy, all share `resolution` field)
  OR a list of dicts {path, resolution} for per-checkpoint native resolution.

Auto-fallback: if no student is present at runtime, only the teacher entries
are loaded; pipeline still functions correctly.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any
import os, yaml


@dataclass
class InferenceConfig:
    # Model — list of {path, resolution} dicts. Each entry runs at its native res.
    checkpoints: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"path": "models/cuda_deep_sat/student/swa_model_final.pth",            "resolution": 224},
        {"path": "models/cuda_deep_sat/expert_bioclip_512/swa_model_final.pth", "resolution": 512},
        {"path": "models/cuda_deep_sat/expert_dinov3_512/swa_model_final.pth",  "resolution": 512},
    ])
    num_classes:  int = 7806
    resolution:   int = 224    # fallback resolution for legacy string-only checkpoint entries
    bioclip:      str = "hf-hub:imageomics/bioclip-2"
    dinov3:       str = "vit_large_patch16_dinov3.lvd1689m"
    convnext:     str = "convnextv2_large.fcmae_ft_in22k_in1k_384"
    require_student: bool = False  # error out if no student checkpoint exists; default: warn + continue
    # Dataset
    test_csv:  str = "/workspace/plantclef/processed/PlantCLEF2024_test.csv"
    img_dir:   str = "/workspace/plantclef/test_images"
    # Tiling
    tiling_enabled:  bool       = True
    tile_size:        int        = 512
    tile_overlap:     float      = 0.2
    scales:           List[float] = field(default_factory=lambda: [1.0, 0.7, 0.5])
    # Inference
    batch_size:   int = 16
    num_workers:  int = 8
    aggregation:  str = "logsumexp"
    disable_logit_standardization: bool = False  # if True, skip (logits-mean)/std in model forward
    ensemble_weights: List[float] = field(default_factory=lambda: [])
    use_hflip_tta:     bool  = False             # Winner Recipe: average each tile with its mirror
    # Test-Time Training (TTT)
    use_ttt:           bool  = False
    ttt_steps:         int   = 10
    ttt_lr:            float = 1e-5
    # Postprocess
    threshold:         float = 0.5
    top_k:             int   = 5
    min_predictions:   int   = 1
    use_pav_isotonic:  bool  = True
    brent_thresholds_path:      str  = ""    # JSON of per-class thresholds (Brent's method)
    conformal_calibration_path: str  = ""    # .pt with conformal q_hat
    conformal_alpha:            float = 0.1  # 1 - target coverage
    logit_adj_tau:              float = 0.05 # plantclef: logit adjustment tau (favor rare if > 0)
    # Hierarchical / ecological postprocessing (all auto-disable if data missing)
    use_pav_calibration:        bool = True   # P(species)≤P(genus)≤P(family) via pav_taxonomy_tree.json
    use_taxon_filter:           bool = True   # AC-3 over data/taxonomic_graph.json (Rust)
    use_frank_wolfe:            bool = False  # Frank-Wolfe + Island Biogeography prior over ecological_adj
    fw_top_k:                   int  = 5      # FW sparsity target (only when use_frank_wolfe)
    use_ising_model:            bool = False  # Spin Glass (Statistical Mechanics) via Simulated Annealing
    use_phenology:              bool = False  # Thermodynamic Phenology (GDD) constraint
    phenology_beta:             float = 1.0   # Inverse-temperature for seasonal prior
    use_geochem:                bool = False  # Biogeochemistry & Edaphic mask
    # Image preprocessing
    use_retinex:                bool = False  # Multi-Scale Retinex illumination normalization
    min_vegetation_frac:        float = 0.0   # Winner Recipe: drop tiles with ExG < this (e.g. 0.15)
    # SAM instance-mask tiling
    use_sam_tiling:             bool = False
    sam_weights_path:           str  = "models/sam_vit_h_4b8939.pth"
    sam_model_type:             str  = "vit_h"
    sam_min_area:               int  = 1024     # px² floor for an instance crop
    sam_max_instances:          int  = 24       # cap per image (bounded inference time)
    # FAISS retrieval-augmented classification
    use_faiss_retrieval:        bool = False
    faiss_prototypes_path:      str  = "models/species_prototypes.pt"
    faiss_alpha:                float = 0.7      # logit weight: final = α·logit + (1-α)·retrieval
    faiss_low_conf_threshold:   float = 0.3      # only fire if max(probs) < this
    # Agentic CV / Visual Chain-of-Thought (VCoT)
    use_llm_arbiter:            bool = False
    use_gemma_tiebreaker:       bool = True   # Fine-grained visual differentiation
    use_nemotron_veto:          bool = True   # Logic/math conflict resolution
    
    # Self-Healing Test-Time Adaptation
    use_async_healing:          bool = False  # Real-time model drift correction via ReflexiveLayer
    submission_csv:  str = "submission.csv"
    # Distributed (for multi-GPU inference)
    local_rank: int = field(default_factory=lambda: int(os.environ.get("LOCAL_RANK", 0)))
    rank:       int = field(default_factory=lambda: int(os.environ.get("RANK", 0)))
    world_size: int = field(default_factory=lambda: int(os.environ.get("WORLD_SIZE", 1)))
    limit:      int = 0  # Process only the first N images (0 = all)


def _normalize_checkpoints(raw_list, default_res: int) -> List[Dict[str, Any]]:
    """Accepts list of strings OR list of {path, resolution} dicts. Returns dicts."""
    out: List[Dict[str, Any]] = []
    for entry in raw_list:
        if isinstance(entry, str):
            out.append({"path": entry, "resolution": default_res})
        elif isinstance(entry, dict) and "path" in entry:
            out.append({
                "path":       entry["path"],
                "resolution": int(entry.get("resolution", default_res)),
            })
    return out


def load(yaml_path: str | None = None) -> InferenceConfig:
    cfg = InferenceConfig()
    if not yaml_path or not os.path.exists(yaml_path):
        return cfg
    with open(yaml_path) as f:
        raw = yaml.safe_load(f) or {}

    m = raw.get("model", {})
    if "num_classes" in m: cfg.num_classes = int(m["num_classes"])
    if "resolution"  in m: cfg.resolution  = int(m["resolution"])
    if "checkpoints" in m:
        cfg.checkpoints = _normalize_checkpoints(m["checkpoints"], cfg.resolution)
    if "require_student" in m: cfg.require_student = bool(m["require_student"])
    for k in ("bioclip", "dinov3", "convnext"):
        if k in m: setattr(cfg, k, m[k])

    ds = raw.get("dataset", {})
    if "test_csv" in ds: cfg.test_csv = ds["test_csv"]
    if "img_dir"  in ds: cfg.img_dir  = ds["img_dir"]

    tl = raw.get("tiling", {})
    if "enabled"   in tl: cfg.tiling_enabled = bool(tl["enabled"])
    if "tile_size" in tl: cfg.tile_size      = int(tl["tile_size"])
    if "overlap"   in tl: cfg.tile_overlap   = float(tl["overlap"])
    if "scales"    in tl: cfg.scales         = list(tl["scales"])

    inf = raw.get("inference", {})
    if "batch_size"  in inf: cfg.batch_size  = int(inf["batch_size"])
    if "num_workers" in inf: cfg.num_workers = int(inf["num_workers"])
    if "aggregation" in inf: cfg.aggregation = inf["aggregation"]
    if "disable_logit_standardization" in inf:
        cfg.disable_logit_standardization = bool(inf["disable_logit_standardization"])
    if "ensemble_weights" in inf: cfg.ensemble_weights = list(inf["ensemble_weights"])
    if "use_hflip_tta" in inf: cfg.use_hflip_tta = bool(inf["use_hflip_tta"])
    if "use_ttt" in inf: cfg.use_ttt = bool(inf["use_ttt"])
    if "ttt_steps" in inf: cfg.ttt_steps = int(inf["ttt_steps"])
    if "ttt_lr" in inf: cfg.ttt_lr = float(inf["ttt_lr"])

    pp = raw.get("postprocess", {})
    if "threshold"        in pp: cfg.threshold        = float(pp["threshold"])
    if "top_k"            in pp: cfg.top_k            = int(pp["top_k"])
    if "min_predictions"  in pp: cfg.min_predictions  = int(pp["min_predictions"])
    if "use_pav_isotonic" in pp: cfg.use_pav_isotonic = bool(pp["use_pav_isotonic"])
    if "brent_thresholds_path"      in pp: cfg.brent_thresholds_path      = pp["brent_thresholds_path"]
    if "conformal_calibration_path" in pp: cfg.conformal_calibration_path = pp["conformal_calibration_path"]
    if "conformal_alpha"            in pp: cfg.conformal_alpha            = float(pp["conformal_alpha"])
    if "logit_adj_tau"              in pp: cfg.logit_adj_tau              = float(pp["logit_adj_tau"])
    if "use_pav_calibration"        in pp: cfg.use_pav_calibration        = bool(pp["use_pav_calibration"])
    if "use_taxon_filter"           in pp: cfg.use_taxon_filter           = bool(pp["use_taxon_filter"])
    if "use_frank_wolfe"            in pp: cfg.use_frank_wolfe            = bool(pp["use_frank_wolfe"])
    if "fw_top_k"                   in pp: cfg.fw_top_k                   = int(pp["fw_top_k"])
    if "use_ising_model"            in pp: cfg.use_ising_model            = bool(pp["use_ising_model"])
    if "use_phenology"              in pp: cfg.use_phenology              = bool(pp["use_phenology"])
    if "phenology_beta"             in pp: cfg.phenology_beta             = float(pp["phenology_beta"])
    if "use_geochem"                in pp: cfg.use_geochem                = bool(pp["use_geochem"])
    if "use_retinex"                in pp: cfg.use_retinex                = bool(pp["use_retinex"])
    if "min_vegetation_frac"        in pp: cfg.min_vegetation_frac        = float(pp["min_vegetation_frac"])
    if "use_sam_tiling"             in pp: cfg.use_sam_tiling              = bool(pp["use_sam_tiling"])
    if "sam_weights_path"           in pp: cfg.sam_weights_path            = pp["sam_weights_path"]
    if "sam_model_type"             in pp: cfg.sam_model_type              = pp["sam_model_type"]
    if "sam_min_area"               in pp: cfg.sam_min_area                = int(pp["sam_min_area"])
    if "sam_max_instances"          in pp: cfg.sam_max_instances           = int(pp["sam_max_instances"])
    if "use_faiss_retrieval"        in pp: cfg.use_faiss_retrieval         = bool(pp["use_faiss_retrieval"])
    if "faiss_prototypes_path"      in pp: cfg.faiss_prototypes_path       = pp["faiss_prototypes_path"]
    if "faiss_alpha"                in pp: cfg.faiss_alpha                 = float(pp["faiss_alpha"])
    if "faiss_low_conf_threshold"   in pp: cfg.faiss_low_conf_threshold    = float(pp["faiss_low_conf_threshold"])
    if "use_llm_arbiter"            in pp: cfg.use_llm_arbiter             = bool(pp["use_llm_arbiter"])
    if "use_gemma_tiebreaker"       in pp: cfg.use_gemma_tiebreaker        = bool(pp["use_gemma_tiebreaker"])
    if "use_nemotron_veto"          in pp: cfg.use_nemotron_veto           = bool(pp["use_nemotron_veto"])
    if "use_async_healing"          in pp: cfg.use_async_healing           = bool(pp["use_async_healing"])

    out = raw.get("output", {})
    if "species_mapping" in out: cfg.species_mapping = out["species_mapping"]
    if "submission_csv"  in out: cfg.submission_csv  = out["submission_csv"]
    if "limit" in out: cfg.limit = int(out["limit"])

    cfg.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    cfg.rank       = int(os.environ.get("RANK", 0))
    cfg.world_size = int(os.environ.get("WORLD_SIZE", 1))
    return cfg
