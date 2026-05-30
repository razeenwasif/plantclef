"""Phase 1 config — completely standalone, no src.config dependency."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import os, yaml


@dataclass
class P1Config:
    # Dataset
    csv_path:    str = "/workspace/plantclef/processed/student_train_final.csv"
    img_dir:     str = "/workspace/plantclef/raw/train/images_max_side_800/"
    # Model
    bioclip:     str = "hf-hub:imageomics/bioclip-2"
    dinov3:      str = "vit_large_patch16_dinov3.lvd1689m"
    convnext:    str = "convnextv2_large.fcmae_ft_in22k_in1k_384"
    resolution:  int = 224
    # Extraction
    batch_size:  int = 512
    num_threads: int = 16
    output_dir:  str = "models/cuda_deep_sat"
    cache_file:  str = "phase1_feature_cache.pt"
    use_compile: bool = False
    # Distributed (populated from env at runtime)
    local_rank:  int = field(default_factory=lambda: int(os.environ.get("LOCAL_RANK", 0)))
    rank:        int = field(default_factory=lambda: int(os.environ.get("RANK", 0)))
    world_size:  int = field(default_factory=lambda: int(os.environ.get("WORLD_SIZE", 1)))

    @property
    def cache_path(self) -> str:
        return os.path.join(self.output_dir, self.cache_file)


def load(yaml_path: str | None = None) -> P1Config:
    cfg = P1Config()
    if not yaml_path or not os.path.exists(yaml_path):
        return cfg
    with open(yaml_path) as f:
        raw = yaml.safe_load(f) or {}

    ds = raw.get("dataset", {})
    if "csv_path" in ds:    cfg.csv_path    = ds["csv_path"]
    if "img_dir"  in ds:    cfg.img_dir     = ds["img_dir"]

    m = raw.get("model", {})
    if "bioclip"    in m:   cfg.bioclip     = m["bioclip"]
    if "dinov3"     in m:   cfg.dinov3      = m["dinov3"]
    if "convnext"   in m:   cfg.convnext    = m["convnext"]
    if "resolution" in m:   cfg.resolution  = int(m["resolution"])

    ex = raw.get("extraction", {})
    if "batch_size"  in ex: cfg.batch_size  = int(ex["batch_size"])
    if "num_threads" in ex: cfg.num_threads = int(ex["num_threads"])
    if "output_dir"  in ex: cfg.output_dir  = ex["output_dir"]
    if "cache_file"  in ex: cfg.cache_file  = ex["cache_file"]

    hw = raw.get("hardware", {})
    if "use_compile" in hw: cfg.use_compile = bool(hw["use_compile"])

    # Populate distributed fields from env
    cfg.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    cfg.rank       = int(os.environ.get("RANK", 0))
    cfg.world_size = int(os.environ.get("WORLD_SIZE", 1))
    return cfg
