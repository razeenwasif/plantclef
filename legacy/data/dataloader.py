import os
# Bypass NVML errors on WSL
os.environ["DALI_DONT_USE_NVML"] = "1"
os.environ["DALI_NVML_FORCE_NO_PCIE_GEN"] = "1"
import cudf
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import numpy as np


class PlantDALIPipeline(Pipeline):
    """
    NVIDIA DALI Pipeline for GPU-accelerated image loading and augmentation.
    Decodes JPEGs on the GPU and performs all augmentations (resize, flip, normalize)
    directly in VRAM to eliminate CPU bottlenecks.
    """
    def __init__(self, batch_size, num_threads, device_id, file_paths, labels,
                 training=True, resolution=384):
        super(PlantDALIPipeline, self).__init__(
            batch_size,
            num_threads,
            device_id,
            seed=42,
            prefetch_queue_depth=4
        )

        self.file_paths = file_paths
        self.labels     = np.array(labels, dtype=np.int32)

        self.input = ops.readers.File(
            files=self.file_paths,
            labels=list(self.labels),
            random_shuffle=training,
            name="Reader",
            num_shards=1,
            shard_id=0,
            pad_last_batch=True,
            dont_use_mmap=True  # Stabilises extraction on network drives (MFS/WSL)
        )

        self.decode = ops.decoders.Image(
            device="mixed",
            output_type=types.RGB,
            device_memory_padding=25165824, # Increased for larger batches
            host_memory_padding=10485760,
        )

        self.training = training
        if self.training:
            # Inception-style random crop to target resolution
            self.resizer = ops.RandomResizedCrop(
                device="gpu", size=resolution, random_area=[0.08, 1.0]
            )
            self.coin = ops.random.CoinFlip(probability=0.5)
        else:
            # Fixed center resize for validation
            self.resizer = ops.Resize(device="gpu", size=(resolution, resolution))

        self.flip = ops.Flip(device="gpu", vertical=0)

        # BioCLIP / DINOv2 normalisation constants
        self.normalize = ops.CropMirrorNormalize(
            device="gpu",
            dtype=types.FLOAT,
            output_layout=types.NCHW,
            mean=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255],
            std=[0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255]
        )

    def define_graph(self):
        jpegs, labels = self.input()
        images = self.decode(jpegs)
        images = self.resizer(images)

        if self.training:
            images = self.flip(images, horizontal=self.coin())
        else:
            images = self.flip(images, horizontal=0)

        output = self.normalize(images)
        return output.gpu(), labels.gpu()


def get_dali_loaders(csv_path, img_dir, batch_size=128, resolution=384,
                     val_split=0.1, num_threads=4, device_id=0,
                     sampling_mode='natural', samples_per_epoch=None):
    """
    Constructs training and validation DALI iterators.

    Args:
        resolution (int): Image size passed to the DALI resize ops.
                          Must match the resolution used during feature extraction.
        sampling_mode (str):
            'natural'  - Direct distribution from CSV.
            'sqrt'     - Square-root resampling (P ∝ 1/√N). Best for Phase 2.
            'balanced' - Full class-aware balancing (P ∝ 1/N).
        samples_per_epoch (int): Optional sub-sampling limit. If set, each epoch
                                 will only use this many images (drawn randomly).
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df          = cudf.read_csv(csv_path, sep=';')
    img_names   = df['image_name'].to_arrow().to_pylist()
    species_ids = df['species_id'].to_arrow().to_pylist()

    unique_species = sorted(set(species_ids))
    species_to_idx = {s: i for i, s in enumerate(unique_species)}
    all_labels     = [species_to_idx[s] for s in species_ids]
    all_paths      = [
        os.path.join(img_dir, str(sid), fname)
        for fname, sid in zip(img_names, species_ids)
    ]

    indices = np.arange(len(all_paths))
    np.random.seed(42)
    np.random.shuffle(indices)

    val_size      = int(len(all_paths) * val_split)
    train_indices = indices[val_size:]
    val_indices   = indices[:val_size]

    train_paths  = [all_paths[i]  for i in train_indices]
    train_labels = [all_labels[i] for i in train_indices]
    val_paths    = [all_paths[i]  for i in val_indices]
    val_labels   = [all_labels[i] for i in val_indices]

    if sampling_mode in ['sqrt', 'balanced']:
        print(f"[Dataloader] Applying {sampling_mode} resampling...")
        label_counts = {}
        for l in train_labels:
            label_counts[l] = label_counts.get(l, 0) + 1

        if sampling_mode == 'sqrt':
            class_weights = {l: 1.0 / np.sqrt(c) for l, c in label_counts.items()}
        else:
            class_weights = {l: 1.0 / c for l, c in label_counts.items()}

        sample_weights  = np.array([class_weights[l] for l in train_labels])
        sample_weights /= sample_weights.sum()

        target_size = samples_per_epoch if samples_per_epoch else len(train_paths)
        resampled = np.random.choice(
            len(train_paths), size=target_size, replace=True, p=sample_weights
        )
        train_paths  = [train_paths[i]  for i in resampled]
        train_labels = [train_labels[i] for i in resampled]
        print(f"[Dataloader] Resampling complete. Effective samples: {len(train_paths):,}")
    elif samples_per_epoch and samples_per_epoch < len(train_paths):
        # Random sub-sampling for 'natural' mode
        resampled = np.random.choice(
            len(train_paths), size=samples_per_epoch, replace=False
        )
        train_paths  = [train_paths[i]  for i in resampled]
        train_labels = [train_labels[i] for i in resampled]
        print(f"[Dataloader] Sub-sampling complete. Effective samples: {len(train_paths):,}")

    train_pipe = PlantDALIPipeline(
        batch_size, num_threads, device_id, train_paths, train_labels,
        training=True, resolution=resolution
    )
    val_pipe = PlantDALIPipeline(
        batch_size, num_threads, device_id, val_paths, val_labels,
        training=False, resolution=resolution
    )

    train_pipe.build()
    val_pipe.build()

    train_loader = DALIGenericIterator(
        [train_pipe], ['data', 'label'], reader_name="Reader", auto_reset=False
    )
    val_loader = DALIGenericIterator(
        [val_pipe], ['data', 'label'], reader_name="Reader", auto_reset=False
    )

    return train_loader, val_loader, len(unique_species)
