# Rust data-I/O engines

SIMD-parallel Rust preprocessing layer used to prepare the PlantCLEF
training and test corpora for fine-tuning runs. The paper's i002
model can be trained directly from a Python `DataLoader` reading the
raw `images_max_side_800/` tree (`experiments/i002_bioclip25_cap_image/
dataset.py`), but in practice the project ran the bulk
resize and shard-packing steps through this Rust layer because Rayon
parallelism with `fast_image_resize` beats `PIL.Image.open` by a
sizeable margin on a many-core host. The crates here are how the
2.65 M-image manifest got pre-processed in finite wall time.

## Crates

| Crate                   | Role                                                                                                       |
|-------------------------|------------------------------------------------------------------------------------------------------------|
| `train_resizer`         | Bulk-resize the 1.41 M-image PlantCLEF 2024 training corpus to a 700-px max side, JPEG q=85. Also exposes a C-FFI (`rust_resize_image` / `rust_free_image`) via `lib.rs` so a C++ loader can call into the SIMD resize path. |
| `test_resizer`          | Resize the 2,105 high-resolution test quadrats to 700 px with Lanczos3 + JPEG q=90 for inference-grade fidelity. |
| `train_packer`          | Pack pre-resized images into 5,000-sample WebDataset `.tar` shards keyed by species ID.                    |
| `val_packer`            | Same packer logic but for the 5 % validation split (`val_*.tar`).                                          |
| `fused_resizer_packer`  | Fused variant of resize + pack in a single pass (used when disk space for the intermediate 700-px tree was tight). |
| `train_indexer`         | Generate NVIDIA DALI v1.2 `.idx` files from `train_*.tar` shards for instant DataLoader startup.           |
| `val_indexer`           | Same indexer logic for `val_*.tar`.                                                                        |

All crates share the same Rayon-based parallelism, `indicatif`
progress bars, `fast_image_resize` for SIMD-optimised resampling, and
the `tar` crate for streaming shard writes. Source/destination paths
are hardcoded to the original `/workspace/plantclef/...` layout at
the top of each `main.rs`; edit those before re-running on a
different host.

## Relationship to the paper pipeline

The i002 training code does not *require* these crates - if you point
`--train-image-root` at any directory tree of `species_id/image_name`
JPEGs, the Python `DataLoader` reads them directly. The Rust layer
was used to:

1. **Resize 1.41 M training images to 700 px once**, so subsequent
   epochs of training do not pay the full-resolution decode cost. The
   `--img-size 224` training transform then upsamples via
   `RandomResizedCrop` at minimal cost.
2. **Shard the resized images into WebDataset `.tar` files**, which
   is the form the (now-retired) DALI-based loader used during
   development of the abandoned ensemble pipeline. Even outside that
   pipeline, the shards are useful for any future loader that prefers
   sequential reads.
3. **Generate DALI `.idx` indices** so a DALI-based loader can start
   in under one second on a millions-of-files dataset.

## Building

Standard Cargo projects. From inside any crate:

```bash
cargo build --release
./target/release/<crate_name>
```

The release builds are what you want; debug builds of these are
roughly 10x slower because `fast_image_resize` relies on inlined
SIMD intrinsics.
