"""Phase 2b Student config — completely standalone, no src.config dependency."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import os, yaml


@dataclass
class P2bStudentConfig:
    # Project Root Discovery
    project_root: str = str(Path(__file__).parents[2].resolve())

    # Dataset
    csv_path:    str = "/workspace/plantclef/processed/student_train_final.csv"
    cleaned_csv: str = "/workspace/plantclef/processed/resolved_build_df.csv"
    img_dir:     str = "/workspace/plantclef/raw/train/images_max_side_800/"
    genus_ids:   str = "/workspace/plantclef/processed/genus_ids.pt"
    # Model
    bioclip:     str = "hf-hub:imageomics/bioclip-2"
    dinov3:      str = "vit_large_patch16_dinov3.lvd1689m"
    convnext:    str = "convnextv2_large.fcmae_ft_in22k_in1k_384"
    num_classes: int = 7808
    # LoRA
    lora_r:       int   = 16
    lora_alpha:   int   = 32
    lora_dropout: float = 0.05
    # Curriculum
    warmup_epochs: int = 3   # warmup_res px
    middle_epochs: int = 5   # 448px
    high_epochs:   int = 2   # 512px
    warmup_res:    int = 224  # resolution for warmup phase
    # Training
    batch_size:         int   = 128
    accumulation_steps: int   = 4
    lr:                 float = 2e-5
    weight_decay:       float = 0.05
    val_every_n_epochs: int   = 2
    max_val_batches:    int   = 200
    num_threads:        int   = 16
    # Teacher KD
    teacher_cache:  str   = "models/cuda_deep_sat/teacher_logit_cache.npy"
    teacher_paths:  List[str] = field(default_factory=lambda: [
        "models/cuda_deep_sat/expert_bioclip_512/swa_model_final.pth",
        "models/cuda_deep_sat/expert_dinov3_512/swa_model_final.pth",
    ])
    kd_alpha:  float = 0.3
    kd_beta:   float = 0.7
    kd_temp:   float = 2.0
    # SWA
    swa_start_epoch: int   = 7
    swa_lr:          float = 5e-6
    # Checkpoints
    warmup_baseline: str = "models/warmup_hardened.pth"
    output_dir:      str = "models/cuda_deep_sat/student"
    # Hardware
    use_compile: bool = True
    use_fp8:     bool = True
    preset:      str  = "auto"
    # Low-RAM Data Management
    low_ram_mode:      bool = False
    shard_window_size: int  = 5
    # Distributed
    local_rank:  int = field(default_factory=lambda: int(os.environ.get("LOCAL_RANK", 0)))
    rank:        int = field(default_factory=lambda: int(os.environ.get("RANK", 0)))
    world_size:  int = field(default_factory=lambda: int(os.environ.get("WORLD_SIZE", 1)))

    def __post_init__(self):
        # Resolve to absolute
        root = Path(self.project_root)
        if not self.warmup_baseline.startswith("/"):
            self.warmup_baseline = str(root / self.warmup_baseline)
        if not self.output_dir.startswith("/"):
            self.output_dir = str(root / self.output_dir)

    @property
    def total_epochs(self) -> int:
        return self.warmup_epochs + self.middle_epochs + self.high_epochs

    @property
    def swa_ckpt(self) -> str:
        return os.path.join(self.output_dir, "swa_model_final.pth")

def _probe_gpu_name() -> str:
    import subprocess
    try:
        res = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True)
        return res.strip().split('\n')[0]
    except:
        return "Unknown GPU"

def load(yaml_path: str | None = None) -> P2bStudentConfig:
    cfg = P2bStudentConfig()
    
    # 1. Probing
    probed = _probe_gpu_name()
    if "Blackwell" in probed or "6000" in probed:
        if cfg.world_size == 1:
            cfg.preset = "rtx_6000_blackwell_1x"
        else:
            cfg.preset = "rtx_6000_blackwell"
    elif "B200" in probed:
        cfg.preset = "rtx_b200"
    elif "B300" in probed:
        cfg.preset = "rtx_b300"
        
    if not yaml_path or not os.path.exists(yaml_path):
        # Even without YAML, allow env override
        name = os.environ.get("ORACLE_NAME")
        if name:
            cfg.output_dir = os.path.join(str(Path(cfg.project_root) / "models/cuda_deep_sat"), name)
        return cfg

    with open(yaml_path) as f:
        raw = yaml.safe_load(f) or {}

    ds = raw.get("dataset", {})
    for k in ("csv_path", "cleaned_csv", "img_dir", "genus_ids"):
        if k in ds: setattr(cfg, k, ds[k])

    m = raw.get("model", {})
    for k in ("bioclip", "dinov3", "convnext"):
        if k in m: setattr(cfg, k, m[k])
    if "num_classes" in m: cfg.num_classes = int(m["num_classes"])

    lora = raw.get("lora", {})
    if "r"       in lora: cfg.lora_r       = int(lora["r"])
    if "alpha"   in lora: cfg.lora_alpha   = int(lora["alpha"])
    if "dropout" in lora: cfg.lora_dropout = float(lora["dropout"])

    cur = raw.get("curriculum", {})
    if "warmup_epochs" in cur: cfg.warmup_epochs = int(cur["warmup_epochs"])
    if "middle_epochs" in cur: cfg.middle_epochs = int(cur["middle_epochs"])
    if "high_epochs"   in cur: cfg.high_epochs   = int(cur["high_epochs"])
    if "warmup_res"    in cur: cfg.warmup_res    = int(cur["warmup_res"])

    teacher = raw.get("teacher", {})
    if "cache_path"  in teacher: cfg.teacher_cache = teacher["cache_path"]
    if "live_paths"  in teacher: cfg.teacher_paths = list(teacher["live_paths"])
    if "kd_alpha"    in teacher: cfg.kd_alpha      = float(teacher["kd_alpha"])
    if "kd_beta"     in teacher: cfg.kd_beta       = float(teacher["kd_beta"])
    if "kd_temp"     in teacher: cfg.kd_temp       = float(teacher["kd_temp"])

    def apply_training(t_dict):
        if "batch_size"         in t_dict: cfg.batch_size         = int(t_dict["batch_size"])
        if "accumulation_steps" in t_dict: cfg.accumulation_steps = int(t_dict["accumulation_steps"])
        if "lr"                 in t_dict: cfg.lr                 = float(t_dict["lr"])
        if "weight_decay"       in t_dict: cfg.weight_decay       = float(t_dict["weight_decay"])
        if "val_every_n_epochs" in t_dict: cfg.val_every_n_epochs = int(t_dict["val_every_n_epochs"])
        if "max_val_batches"    in t_dict: cfg.max_val_batches    = int(t_dict["max_val_batches"])
        if "num_threads"        in t_dict: cfg.num_threads        = int(t_dict["num_threads"])

    # Base training
    apply_training(raw.get("training", {}))
    apply_training(raw.get("hyperparameters", {}))

    swa = raw.get("swa", {})
    if "start_epoch" in swa: cfg.swa_start_epoch = int(swa["start_epoch"])
    if "lr"          in swa: cfg.swa_lr          = float(swa["lr"])

    ckpt = raw.get("checkpoints", {})
    if "warmup_baseline" in ckpt: cfg.warmup_baseline = ckpt["warmup_baseline"]
    if "output_dir"      in ckpt: cfg.output_dir      = ckpt["output_dir"]

    hw = raw.get("hardware", {})
    if "use_compile" in hw: cfg.use_compile = bool(hw["use_compile"])
    if "use_fp8"     in hw: cfg.use_fp8     = bool(hw["use_fp8"])
    if "low_ram_mode" in hw: cfg.low_ram_mode = bool(hw["low_ram_mode"])
    if "shard_window_size" in hw: cfg.shard_window_size = int(hw["shard_window_size"])

    # 3. Apply Presets
    if "presets" in raw and cfg.preset in raw["presets"]:
        pre = raw["presets"][cfg.preset]
        apply_training(pre.get("training", {}))
        apply_training(pre.get("hyperparameters", {}))
        
        p_lora = pre.get("lora", {})
        if "r"     in p_lora: cfg.lora_r     = int(p_lora["r"])
        if "alpha" in p_lora: cfg.lora_alpha = int(p_lora["alpha"])

    # ORACLE: Auto-enable low_ram_mode for consumer cards if not explicitly set
    if "4090" in probed or "3090" in probed:
        if "low_ram_mode" not in hw:
            cfg.low_ram_mode = True
            print("[Config] 4090/3090 detected: Auto-enabling low_ram_mode.")

    # ORACLE: Dynamic output pathing for multi-seed/orchestrator support
    # Applies AFTER YAML so env-var instructions take final precedence.
    name = os.environ.get("ORACLE_NAME")
    if name:
        root = Path(cfg.project_root)
        cfg.output_dir = str(root / "models/cuda_deep_sat" / name)

    # Global Blackwell optimizations
    if "Blackwell" in probed or "6000" in probed:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
        os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
        os.environ["TORCH_CUDA_MATMUL_TF32"] = "1"

    cfg.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    cfg.rank       = int(os.environ.get("RANK", 0))
    cfg.world_size = int(os.environ.get("WORLD_SIZE", 1))
    return cfg
