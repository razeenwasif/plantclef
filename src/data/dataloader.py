import os
import glob
import torch
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Iterator
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

try:
    from nvidia.dali import ops
    from nvidia.dali import types
    from nvidia.dali import fn
    from nvidia.dali.pipeline import Pipeline
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
    HAS_DALI = True
except ImportError:
    HAS_DALI = False

from src import config as _cfg

from .shard_manager import DynamicShardManager
# WebDataset loader for the TPU path. Imported lazily inside get_loaders
# so a CUDA-only host with no webdataset package still loads this module.
from .wds_loader import get_webdataset_loaders  # noqa: F401  (re-exported)

class DynamicDALIIterator:
    """
    Wrapper that rotates DALI pipelines over shard windows to save RAM.
    """
    def __init__(self, manager: DynamicShardManager, pipeline_factory):
        self.manager = manager
        self.pipeline_factory = pipeline_factory
        self.current_iterator = None
        self._prepare_next_loader()

    def _prepare_next_loader(self):
        local_paths = self.manager.prepare_next_window()
        if not local_paths:
            self.current_iterator = None
            return False
        
        # Create a new pipeline for this specific batch of local shards
        pipeline = self.pipeline_factory(local_paths)
        pipeline.build()
        
        from nvidia.dali.plugin.pytorch import DALIGenericIterator
        self.current_iterator = DALIGenericIterator(
            [pipeline], ["data", "label"], 
            size=pipeline.epoch_size("Reader"),
            auto_reset=True,
            last_batch_policy=types.LAST_BATCH_FILL
        )
        return True

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_iterator is None:
            raise StopIteration
            
        try:
            return next(self.current_iterator)
        except StopIteration:
            # Current window exhausted, try next
            if self._prepare_next_loader():
                return next(self.current_iterator)
            else:
                self.current_iterator = None
                raise StopIteration

    def __len__(self):
        # Rough estimate based on all shards
        return len(self.manager.all_shards) * 1000 // 16 # Assuming 1000 samples/shard, batch 16

    def reset(self):
        self.manager.reset()
        self._prepare_next_loader()


# ---------------------------------------------------------------------------
# High-Speed DALI Pipeline
# ---------------------------------------------------------------------------
if HAS_DALI:
    class PlantDALIPipeline(Pipeline):
        def __init__(self, batch_size: int, num_threads: int, device_id: int, 
                     file_paths: List[str], labels: List[int], training: bool = True, 
                     resolution: int = 384, num_shards: int = 1, shard_id: int = 0, seed: int = 42,
                     shard_path: Optional[str] = None, use_index: bool = False,
                     manual_paths: Optional[List[str]] = None):
            
            # Outer pipeline buffer — was hardcoded 3 which starved B200/B300.
            # Honor the config-driven PREFETCH_DEPTH (5 default, 8 on B200, 10 on B300).
            super(PlantDALIPipeline, self).__init__(
                batch_size, num_threads, device_id,
                seed=seed + shard_id,
                prefetch_queue_depth=getattr(_cfg, "PREFETCH_DEPTH", 5),
            )
            
            self.is_webdataset = False
            if manual_paths:
                self.is_webdataset = True
                tar_files = manual_paths
                index_paths = None # We skip index for dynamic loading to keep it simple/fast
                
                self.input = ops.readers.Webdataset(
                    paths=tar_files,
                    index_paths=index_paths,
                    ext=["jpg", "cls"],
                    missing_component_behavior="skip",
                    random_shuffle=training,
                    num_shards=1, # Already sharded by ShardManager
                    shard_id=0,
                    pad_last_batch=True,
                    prefetch_queue_depth=_cfg.PREFETCH_DEPTH,
                    name="Reader"
                )
            elif shard_path and os.path.exists(shard_path):
                self.is_webdataset = True
                prefix = "train_" if training else "val_"
                tar_files = sorted(glob.glob(os.path.join(shard_path, f"{prefix}*.tar")))
                
                # plantclef: Filter out garbage files (e.g. 1KB placeholders/truncated shards)
                tar_files = [f for f in tar_files if os.path.getsize(f) > 1024 * 1024]
                
                index_paths = None
                if use_index:
                    index_dir = os.path.join(shard_path, ".index")
                    idx_files = [os.path.join(index_dir, Path(f).stem + ".idx") for f in tar_files]
                    missing = [idx for idx in idx_files if not os.path.exists(idx)]
                    if len(missing) == 0:
                        index_paths = idx_files
                        print(f"[Loader] Found {len(index_paths)} pre-generated indices.")
                    else:
                        print(f"[Loader] Missing {len(missing)} index files (e.g., {os.path.basename(missing[0])}).")
                        print("[Loader] Auto-generating missing indices now...")
                        from tools.data.generate_dali_index import generate_index
                        from tqdm import tqdm
                        
                        # Only generate on local_rank 0 to avoid race conditions where 
                        # multiple processes truncate or corrupt the binary index files.
                        local_rank = int(os.environ.get("LOCAL_RANK", 0))
                        if local_rank == 0:
                            iterator = tqdm(zip(tar_files, idx_files), total=len(tar_files), desc="Generating Indices")
                            for tar_f, idx_f in iterator:
                                if not os.path.exists(idx_f):
                                    generate_index(tar_f, idx_f)
                        
                        # plantclef: Fast-Epoch Transition
                        # Sync all processes to ensure indices are visible before DALI reader init
                        import torch.distributed as dist
                        if dist.is_initialized():
                            dist.barrier()
                        
                        index_paths = idx_files
                # plantclef: Conditional RAM-Disk Indexing
                # DALI will map the shards in memory if index_paths is None.
                self.input = ops.readers.Webdataset(
                    paths=tar_files,
                    index_paths=index_paths,
                    ext=["jpg", "cls"],
                    missing_component_behavior="skip",
                    random_shuffle=training,
                    num_shards=num_shards,
                    shard_id=shard_id,
                    pad_last_batch=True,
                    prefetch_queue_depth=_cfg.PREFETCH_DEPTH,
                    name="Reader"
                )
            else:
                self.input = ops.readers.File(
                    files=file_paths,
                    labels=list(labels),
                    random_shuffle=training,
                    num_shards=num_shards,
                    shard_id=shard_id,
                    pad_last_batch=True,
                    prefetch_queue_depth=_cfg.PREFETCH_DEPTH,
                    name="Reader" 
                )

            self.is_training = training
            self.decode = ops.decoders.Image(
                device="mixed", output_type=types.RGB,
                hw_decoder_load=1.0, affine=True
            )
            
            # plantclef: Force fixed shape for batch stacking (IsDenseTensor fix)
            self.resizer = ops.Resize(
                device="gpu",
                size=[resolution, resolution], 
                interp_type=types.INTERP_LANCZOS3
            )
            
            if training:
                self.flip = ops.Flip(device="gpu")

            self.normalize = ops.CropMirrorNormalize(
                device="gpu",
                dtype=types.FLOAT,
                output_layout=types.NCHW,
                mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                std=[0.229 * 255, 0.224 * 255, 0.225 * 255]
            )

        def define_graph(self) -> Tuple[Any, Any]:
            jpegs, labels = self.input()
            
            # plantclef: Binary Reinterpretation (Zero-Copy)
            # If using WebDataset, labels are 4-byte raw binary blocks from Rust
            if self.is_webdataset:
                labels = fn.reinterpret(labels, dtype=types.INT32)

            decoded = self.decode(jpegs)
            images = self.resizer(decoded)
            if self.is_training:
                images = self.flip(images, horizontal=fn.random.coin_flip(probability=0.5))
            return self.normalize(images).gpu(), labels.gpu()

# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
import yaml

def _load_dataset_config(dataset_name: str) -> dict:
    config_path = "configs/datasets.yaml"
    if not os.path.exists(config_path):
        return {}
    with open(config_path, 'r') as f:
        full_cfg = yaml.safe_load(f)
    return full_cfg.get(dataset_name, {})

def get_dali_loaders(batch_size: int, resolution: int, device_id: int = 0,
                     num_threads: int = 16, num_shards: int = 1, shard_id: int = 0,
                     seed: int = 42, training: bool = True,
                     img_dir: Optional[str] = None, val_ratio: float = 0.1,
                     low_ram_mode: bool = False, window_size: int = 5,
                     dataset: Optional[str] = None):
    """
    Dynamically discovers the dataset and returns DALI iterators.
    Supports named configurations (e.g. plantnet300k) from configs/datasets.yaml.

    NOTE: this is the CUDA-only entry point. The TPU path uses
    `get_webdataset_loaders` (see `wds_loader.py`); callers that want
    accelerator-agnostic dispatch should use `get_loaders(...)` below.
    """
    if not HAS_DALI:
        # We delay this fail-fast until someone actually calls the function so
        # that a TPU host importing this module (e.g. for the type alias) doesn't
        # blow up at import time. The error mentions the WebDataset alternative.
        try:
            from src.training.accelerator import accelerator as _accel
            mode = _accel().mode
        except Exception:
            mode = "unknown"
        raise ImportError(
            f"NVIDIA DALI is not installed (accelerator mode: {mode!r}). "
            "The DALI data pipeline is CUDA-only. On TPU hosts, use "
            "`get_loaders(...)` with `shard_path=...` — it routes to the "
            "WebDataset backend automatically. On a CUDA host, run "
            "`pip install --extra-index-url https://pypi.nvidia.com nvidia-dali-cuda120`."
        )
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from .hash_indexer import resolve_external_path, HashPathIndexer

    # 1. Resolve Dataset Logic
    ds_cfg = _load_dataset_config(dataset) if dataset else {}
    ds_type = ds_cfg.get("type", "flat")
    root = resolve_external_path(ds_cfg.get("root", img_dir or _cfg.IMG_DIR))
    
    if not os.path.exists(root) and dataset:
        logger.warning(f"[Loader] Dataset path '{root}' invalid or missing.")
        logger.info("[Loader] Handing over to Agentic Auto-Discovery (Gemma 4)...")
        from .auto_discover import AgenticDatasetFinder
        finder = AgenticDatasetFinder()
        new_cfg = finder.locate_and_save(dataset)
        
        if new_cfg and os.path.exists(new_cfg.get("root", "")):
            ds_cfg = new_cfg
            root = new_cfg["root"]
            ds_type = new_cfg["type"]
            logger.info(f"[Loader] Auto-discovery successful. Resuming pipeline with {root}")
        else:
            raise FileNotFoundError(f"Agentic Auto-Discovery failed to locate dataset '{dataset}'.")

    logger.info(f"[Loader] Dataset: {dataset or 'unnamed'} (Type: {ds_type}) at {root}")

    if ds_type == "pre_split":
        # Handle PlantNet-300K style: train, val, test folders
        train_path = os.path.join(root, ds_cfg.get("train_dir", "train"))
        val_path   = os.path.join(root, ds_cfg.get("val_dir", "val"))
        
        # Discover classes from train (assumes val has subset/same)
        classes = sorted([d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))])
        class_to_idx = {cls: i for i, cls in enumerate(classes)}
        num_classes = len(classes)

        def scan_dir(p):
            res = []
            for cls in classes:
                c_dir = os.path.join(p, cls)
                if not os.path.exists(c_dir): continue
                for img in os.listdir(c_dir):
                    if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                        res.append({'path': os.path.join(c_dir, img), 'label': class_to_idx[cls]})
            return res

        t_data = scan_dir(train_path)
        v_data = scan_dir(val_path)
        
        t_file_paths = [x['path'] for x in t_data]
        t_labels     = [x['label'] for x in t_data]
        v_file_paths = [x['path'] for x in v_data]
        v_labels     = [x['label'] for x in v_data]

    else:
        # Default: Flat folder discovery with stratified split
        data = []
        classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        class_to_idx = {cls: i for i, cls in enumerate(classes)}
        num_classes = len(classes)

        for cls in classes:
            cls_dir = os.path.join(root, cls)
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    data.append({'path': os.path.join(cls, img_name), 'label': class_to_idx[cls]})

        df = pd.DataFrame(data)
        train_df, val_df = train_test_split(
            df, test_size=ds_cfg.get("val_ratio", val_ratio), stratify=df['label'], random_state=seed
        )
        
        t_file_paths = (root + "/" + train_df['path']).tolist()
        t_labels     = train_df['label'].tolist()
        v_file_paths = (root + "/" + val_df['path']).tolist()
        v_labels     = val_df['label'].tolist()

    logger.info(f"[Loader] Final counts: {len(t_file_paths)} train, {len(v_file_paths)} val across {num_classes} classes.")

    # 2. Path Hashing (WSL Memory Optimization)
    indexer = HashPathIndexer()
    indexer.register_paths(t_file_paths)
    indexer.register_paths(v_file_paths)

    # 3. Pipeline Initialization
    t_pipe = PlantDALIPipeline(
        batch_size, num_threads, device_id,
        file_paths=t_file_paths, labels=t_labels,
        training=True, resolution=resolution,
        num_shards=num_shards, shard_id=shard_id, seed=seed
    )
    t_pipe.build()

    v_pipe = PlantDALIPipeline(
        batch_size, num_threads, device_id,
        file_paths=v_file_paths, labels=v_labels,
        training=False, resolution=resolution,
        num_shards=1, shard_id=0, seed=seed
    )
    v_pipe.build()

    from nvidia.dali.plugin.pytorch import DALIGenericIterator
    train_loader = DALIGenericIterator([t_pipe], ['data', 'label'], reader_name="Reader")
    val_loader = DALIGenericIterator([v_pipe], ['data', 'label'], reader_name="Reader")

    return train_loader, val_loader, num_classes, None


# ───────────────────────────────────────────────────────────────────────────
# Accelerator-aware selector
# ───────────────────────────────────────────────────────────────────────────
def get_loaders(batch_size: int, resolution: int, *,
                # DALI-only knobs (ignored on TPU path):
                device_id: int = 0, num_threads: int = 16,
                img_dir: Optional[str] = None, val_ratio: float = 0.1,
                low_ram_mode: bool = False, window_size: int = 5,
                dataset: Optional[str] = None,
                # WebDataset knobs (ignored on CUDA path):
                shard_path: Optional[str] = None,
                num_workers: int = 8,
                # Shared knobs:
                num_shards: int = 1, shard_id: int = 0, seed: int = 42,
                training: bool = True):
    """Single accelerator-agnostic loader factory.

    Routes to:
      * `get_dali_loaders` when `accelerator().is_cuda` (or CPU dev fallback)
      * `get_webdataset_loaders` when `accelerator().is_tpu`

    Both branches return `(train_loader, val_loader, num_classes, _)` where
    each loader yields `[{'data': Tensor[B,3,H,W], 'label': Tensor[B]}]`
    per the DALI iterator convention, so consumer code (trainer.py,
    loops.py) is mode-agnostic.

    TPU-only requirements
    ---------------------
    * `shard_path` MUST be provided (the directory containing `train_*.tar`
      and `val_*.tar` written by `src/data/shard_manager.py`).
    * `num_classes` is returned as ``None`` — fix the taxonomy size from
      config (`_cfg.NUM_CLASSES` or your dataset config).
    """
    # Lazy import — circular dep avoidance + don't force accelerator init on
    # callers that import this module purely for side-effects (type hints, etc.)
    from src.training.accelerator import accelerator as _accel
    accel = _accel()

    if accel.is_tpu:
        if not shard_path:
            shard_path = getattr(_cfg, "SHARD_DIR", None) or getattr(_cfg, "SHARDS", None)
        if not shard_path:
            raise ValueError(
                "TPU data path requires a shard_path (or config.SHARD_DIR). "
                "Build shards via src/data/shard_manager.py before training."
            )
        return get_webdataset_loaders(
            batch_size=batch_size, resolution=resolution,
            shard_path=shard_path,
            num_shards=num_shards, shard_id=shard_id, seed=seed,
            num_workers=num_workers,
        )

    # CUDA / CPU-dev fallback path
    return get_dali_loaders(
        batch_size=batch_size, resolution=resolution,
        device_id=device_id, num_threads=num_threads,
        num_shards=num_shards, shard_id=shard_id, seed=seed,
        training=training, img_dir=img_dir, val_ratio=val_ratio,
        low_ram_mode=low_ram_mode, window_size=window_size, dataset=dataset,
    )
