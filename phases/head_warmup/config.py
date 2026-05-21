"""Phase 2a config — completely standalone, no src.config dependency."""
from __future__ import annotations
from dataclasses import dataclass, field
import os, yaml


@dataclass
class P2aConfig:
    # Dataset
    feature_cache: str = "models/cuda_deep_sat/phase1_feature_cache.pt"
    genus_ids:     str = "/workspace/plantclef/processed/genus_ids.pt"
    # Model
    in_features:      int = 3328
    hidden_features:  int = 2048
    num_classes:      int = 7808
    dropout:          float = 0.2
    # Training
    batch_size:    int   = 2048
    lr:            float = 1e-4
    weight_decay:  float = 0.01
    epochs:        int   = 20
    # Output
    checkpoint: str = "models/warmup_hardened.pth"
    # Distributed
    local_rank: int = field(default_factory=lambda: int(os.environ.get("LOCAL_RANK", 0)))
    rank:       int = field(default_factory=lambda: int(os.environ.get("RANK", 0)))
    world_size: int = field(default_factory=lambda: int(os.environ.get("WORLD_SIZE", 1)))


def load(yaml_path: str | None = None) -> P2aConfig:
    cfg = P2aConfig()
    if not yaml_path or not os.path.exists(yaml_path):
        return cfg
    with open(yaml_path) as f:
        raw = yaml.safe_load(f) or {}

    ds = raw.get("dataset", {})
    if "feature_cache" in ds: cfg.feature_cache = ds["feature_cache"]
    if "genus_ids"     in ds: cfg.genus_ids     = ds["genus_ids"]

    m = raw.get("model", {})
    if "in_features"     in m: cfg.in_features     = int(m["in_features"])
    if "hidden_features" in m: cfg.hidden_features = int(m["hidden_features"])
    if "num_classes"     in m: cfg.num_classes     = int(m["num_classes"])
    if "dropout"         in m: cfg.dropout         = float(m["dropout"])

    t = raw.get("training", {})
    if "batch_size"   in t: cfg.batch_size   = int(t["batch_size"])
    if "lr"           in t: cfg.lr           = float(t["lr"])
    if "weight_decay" in t: cfg.weight_decay = float(t["weight_decay"])
    if "epochs"       in t: cfg.epochs       = int(t["epochs"])

    out = raw.get("output", {})
    if "checkpoint" in out: cfg.checkpoint = out["checkpoint"]

    cfg.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    cfg.rank       = int(os.environ.get("RANK", 0))
    cfg.world_size = int(os.environ.get("WORLD_SIZE", 1))
    return cfg
