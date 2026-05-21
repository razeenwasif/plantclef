from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Literal
from pathlib import Path

@dataclass
class HardwareConfig:
    """Hardware detection and Blackwell-specific optimization flags."""
    probed_gpu: str = "Unknown GPU"
    preset: str = "auto"
    use_fp8: bool = True
    use_compile: bool = True
    compile_mode: str = "max-autotune"
    num_gpus: int = 1
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    extreme_mode: bool = False

@dataclass
class DatasetConfig:
    """Dataset paths and local shard management."""
    mode: str = "full"
    raw_csv: str = "/workspace/plantclef/raw/PlantCLEF2024_single_plant_training_metadata.csv"
    cleaned_csv: str = "/workspace/plantclef/processed/train_metadata_cleaned_verified_stratified_deep_audit.csv"
    img_dir: str = "/workspace/plantclef/raw/train/images_max_side_800/"
    shard_dir: str = "/workspace/plantclef/shards/"

@dataclass
class ModelConfig:
    """Core architecture settings (Backbones, LoRA, Resolution)."""
    num_classes: int = 7808
    resolution: int = 448
    bioclip_name: str = "hf-hub:imageomics/bioclip-2"
    dinov3_name: str = "vit_large_patch16_dinov3.lvd1689m"
    convnext_name: str = "convnextv2_large.fcmae_ft_in22k_in1k_384"

    lora_r: int = 512
    lora_alpha: int = 1024
    lora_dropout: float = 0.05

    use_flash_attn: bool = True
    use_zero_shot: bool = True
    zero_shot_weight: float = 0.1
    zero_shot_temp: float = 0.1
    zero_shot_path: str = "models/zero_shot_anchors.pt"

    # Inference runtime settings
    input_size: int = 448
    vit_name: str = "vit_large_patch16_dinov3.lvd1689m"
    device: str = "cuda"
    batch_size: int = 32
    use_compile: bool = True
    logit_adj_path: str = ""
    interpolation: str = "bilinear"
    norm_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    norm_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    use_region_features: bool = False

    # Retinex illumination normalization
    use_retinex: bool = False
    retinex_sigmas: List[float] = field(default_factory=lambda: [15.0, 80.0, 250.0])

@dataclass
class TrainingConfig:
    """Hyperparameters and distributed training schedules."""
    batch_size: int = 128
    p2_batch_size: int = 1280
    accumulation_steps: int = 3
    epochs_p1: int = 5
    epochs_p2: int = 20
    p2_samples_per_epoch: int = 1150000
    learning_rate: float = 5e-6
    weight_decay: float = 0.01
    
    val_every_n_epochs: int = 5
    start_val_epoch: int = 5
    max_val_batches: int = 500
    patience: int = 10
    
    use_swa: bool = True
    swa_start_epoch: int = 15
    swa_lr: float = 1e-5
    
    use_trait_distillation: bool = True
    kd_alpha: float = 0.3
    curriculum: Dict[str, int] = field(default_factory=dict)

@dataclass
class MemoryConfig:
    """Per-backbone forward chunk sizes for memory-efficient inference/training."""
    extract_chunk_size: int = 32
    chunk_size: int = 32
    p2_chunk_size: int = 8
    prefetch_depth: int = 5

@dataclass
class TilingConfig:
    """Inference-specific tiling strategy."""
    method: Literal["sliding", "grid"] = "sliding"
    tile_size: int = 518
    stride: int = 172
    scales: List[float] = field(default_factory=lambda: [1.0, 0.707])

@dataclass
class FilterConfig:
    """Runtime tile filtering to drop noise."""
    enabled: bool = True
    min_variance: float = 10.0
    min_vegetation_ratio: float = 0.01

@dataclass
class AggregationConfig:
    """Pooling logic for multi-tile consensus."""
    method: Union[str, List[str]] = field(default_factory=lambda: ["max", "bayesian_veg", "mean"])
    temperature: float = 1.0
    top_p_percentile: float = 90.0

@dataclass
class PostprocessConfig:
    """Final refinement and consistency checking."""
    global_threshold: float = 0.10
    per_class_thresholds: Optional[Dict[int, float]] = None
    top_k: int = 3
    min_predictions: int = 1
    use_few_shot_retrieval: bool = True
    solve_quadrat_consistency: bool = True
    consistency_method: Literal["ac3", "loopy_bp"] = "loopy_bp"
    bp_iterations: int = 10
    noise_mask_penalty: float = 0.35

@dataclass
class OracleConfig:
    """Root configuration object for the ORACLE engine."""
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    tiling: TilingConfig = field(default_factory=TilingConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    
    # Global Paths
    base_model_dir: str = "models/cuda_deep_sat/cluster_final"
    output_dir: str = "/workspace/PlantCLEF2026/submissions/"
    checkpoint_paths: List[str] = field(default_factory=list)
    
    # Utility Shortcuts (Legacy Compatibility)
    def __post_init__(self):
        # Ensure Paths are resolved
        self.output_dir = str(Path(self.output_dir).resolve())
