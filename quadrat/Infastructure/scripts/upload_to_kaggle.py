"""
PlantCLEF 2026 - Upload Datasets to Kaggle

Kaggle dataset size limit: ~100GB per dataset.
The training set (~160GB) needs to be split into multiple datasets.

Usage:
    python3 scripts/upload_to_kaggle.py --what all
    python3 scripts/upload_to_kaggle.py --what train
    python3 scripts/upload_to_kaggle.py --what pseudo
    python3 scripts/upload_to_kaggle.py --what models
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path("/workspace/plantclef/raw")
KAGGLE_USERNAME = None  # Auto-detected from kaggle.json


def get_kaggle_username():
    global KAGGLE_USERNAME
    if KAGGLE_USERNAME:
        return KAGGLE_USERNAME
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("ERROR: ~/.kaggle/kaggle.json not found.")
        print("Run setup_environment.sh first.")
        sys.exit(1)
    with open(kaggle_json) as f:
        KAGGLE_USERNAME = json.load(f)["username"]
    return KAGGLE_USERNAME


def create_dataset_metadata(dataset_dir, slug, title, subtitle=""):
    """Create the dataset-metadata.json required by Kaggle."""
    username = get_kaggle_username()
    metadata = {
        "title": title,
        "id": f"{username}/{slug}",
        "licenses": [{"name": "CC-BY-4.0"}],
        "subtitle": subtitle,
    }
    meta_path = dataset_dir / "dataset-metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Created metadata: {meta_path}")
    return meta_path


def kaggle_dataset_create(dataset_dir, is_update=False):
    """Create or update a Kaggle dataset."""
    cmd = ["kaggle", "datasets"]
    cmd.append("version" if is_update else "create")
    cmd.extend(["-p", str(dataset_dir)])
    if is_update:
        cmd.extend(["-m", "Updated dataset"])
    cmd.append("--dir-mode=tar")

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  ✓ Upload successful")
        print(f"    {result.stdout.strip()}")
    else:
        stderr = result.stderr.strip()
        if "already exists" in stderr.lower():
            print("  Dataset already exists, trying update...")
            kaggle_dataset_create(dataset_dir, is_update=True)
        else:
            print(f"  ✗ Upload failed: {stderr}")
            return False
    return True


def get_subdirectory_sizes(root):
    """Get sizes of immediate subdirectories in GB."""
    sizes = {}
    for d in sorted(root.iterdir()):
        if d.is_dir():
            total = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            sizes[d.name] = total / (1024 ** 3)
    return sizes


def split_into_chunks(items, max_size_gb=90):
    """Split a list of (name, size_gb) tuples into chunks under max_size_gb."""
    chunks = []
    current_chunk = []
    current_size = 0

    for name, size in sorted(items, key=lambda x: x[1], reverse=True):
        if current_size + size > max_size_gb and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_size = 0
        current_chunk.append(name)
        current_size += size

    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def upload_training_data():
    """Upload training data, split into chunks if >100GB."""
    print("\n══ Uploading Training Data ══")
    train_dir = BASE_DIR / "train"

    if not train_dir.exists() or not any(train_dir.iterdir()):
        print("  ✗ Training directory is empty. Download data first.")
        return

    print("  Calculating directory sizes...")
    sizes = get_subdirectory_sizes(train_dir)
    total_gb = sum(sizes.values())
    print(f"  Total size: {total_gb:.1f} GB across {len(sizes)} directories")

    if total_gb <= 95:
        print("  Fits in a single dataset, uploading directly...")
        create_dataset_metadata(
            train_dir,
            "plantclef2026-training-data",
            "PlantCLEF 2026 Training Data",
            "Single-plant images for 7.8k species (max-side 800px)",
        )
        kaggle_dataset_create(train_dir)
    else:
        items = [(name, size) for name, size in sizes.items()]
        chunks = split_into_chunks(items, max_size_gb=90)
        print(f"  Splitting into {len(chunks)} datasets (Kaggle ~100GB limit)")

        for i, chunk_dirs in enumerate(chunks, 1):
            chunk_size = sum(sizes[d] for d in chunk_dirs)
            print(f"\n  ── Chunk {i}/{len(chunks)}: {len(chunk_dirs)} dirs, {chunk_size:.1f} GB ──")

            chunk_staging = BASE_DIR / f"_train_chunk_{i}"
            chunk_staging.mkdir(exist_ok=True)

            for dirname in chunk_dirs:
                src = train_dir / dirname
                dst = chunk_staging / dirname
                if not dst.exists():
                    dst.symlink_to(src)

            create_dataset_metadata(
                chunk_staging,
                f"plantclef2026-training-part{i}",
                f"PlantCLEF 2026 Training Data (Part {i}/{len(chunks)})",
                f"Single-plant images chunk {i}",
            )
            kaggle_dataset_create(chunk_staging)


def upload_pseudo_quadrats():
    """Upload pseudo-quadrat images."""
    print("\n══ Uploading Pseudo-Quadrat Images ══")
    pq_dir = BASE_DIR / "pseudo_quadrats"

    if not pq_dir.exists() or not any(pq_dir.iterdir()):
        print("  ✗ Pseudo-quadrat directory is empty. Download data first.")
        return

    total_size = sum(f.stat().st_size for f in pq_dir.rglob("*") if f.is_file())
    total_gb = total_size / (1024 ** 3)
    print(f"  Total size: {total_gb:.1f} GB")

    if total_gb <= 95:
        create_dataset_metadata(
            pq_dir,
            "plantclef2026-pseudo-quadrats",
            "PlantCLEF 2026 Pseudo-Quadrat Images",
            "212K unlabeled pseudo-quadrat images for domain adaptation",
        )
        kaggle_dataset_create(pq_dir)
    else:
        all_files = sorted([f for f in pq_dir.rglob("*") if f.is_file()])
        midpoint = len(all_files) // 2

        for i, file_slice in enumerate([all_files[:midpoint], all_files[midpoint:]], 1):
            chunk_dir = BASE_DIR / f"_pq_chunk_{i}"
            chunk_dir.mkdir(exist_ok=True)

            for f in file_slice:
                rel = f.relative_to(pq_dir)
                dst = chunk_dir / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    dst.symlink_to(f)

            create_dataset_metadata(
                chunk_dir,
                f"plantclef2026-pseudo-quadrats-part{i}",
                f"PlantCLEF 2026 Pseudo-Quadrats (Part {i}/2)",
            )
            kaggle_dataset_create(chunk_dir)


def upload_models():
    """Upload pre-trained models."""
    print("\n══ Uploading Pre-trained Models ══")
    model_dir = BASE_DIR / "models"

    if not model_dir.exists() or not any(model_dir.iterdir()):
        print("  ✗ Models directory is empty. Download models first.")
        return

    create_dataset_metadata(
        model_dir,
        "plantclef2026-pretrained-models",
        "PlantCLEF 2026 Pre-trained DINOv2 Models",
        "ViT-base DINOv2 models fine-tuned on PlantCLEF 2024 data",
    )
    kaggle_dataset_create(model_dir)


def main():
    parser = argparse.ArgumentParser(description="Upload PlantCLEF data to Kaggle")
    parser.add_argument(
        "--what",
        choices=["all", "train", "pseudo", "models"],
        default="all",
        help="Which dataset to upload",
    )
    args = parser.parse_args()

    username = get_kaggle_username()
    print(f"Kaggle username: {username}")
    print(f"Base directory: {BASE_DIR}")

    if args.what in ("all", "train"):
        upload_training_data()
    if args.what in ("all", "pseudo"):
        upload_pseudo_quadrats()
    if args.what in ("all", "models"):
        upload_models()

    print("\n============================================")
    print("  Upload complete!")
    print("  Check your datasets at:")
    print(f"    https://www.kaggle.com/{username}/datasets")
    print("============================================")


if __name__ == "__main__":
    main()
