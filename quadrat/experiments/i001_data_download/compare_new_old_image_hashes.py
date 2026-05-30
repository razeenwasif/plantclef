#!/usr/bin/env python3

import os
import argparse
import hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from PIL import Image, ImageOps
from tqdm.auto import tqdm

from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm
import os
import pandas as pd


def parallel_exists(paths, workers=32, desc="Checking files"):
    """
    Parallel file existence check.
    Hashes/checks unique paths once, then maps results back to original order.
    """
    paths_series = pd.Series(paths).astype(str)
    unique_paths = paths_series.drop_duplicates().tolist()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        exists_list = list(
            tqdm(
                ex.map(os.path.exists, unique_paths),
                total=len(unique_paths),
                desc=desc,
            )
        )

    exists_map = dict(zip(unique_paths, exists_list))

    return paths_series.map(exists_map).astype(bool)

# ---------------------------------------------------------------------
# Hash functions
# ---------------------------------------------------------------------

def average_hash(path, hash_size=16):
    """
    Perceptual average hash.
    Good for catching same/similar images even if resized/recompressed.

    hash_size=16 gives 256-bit hash, safer than the common 8x8 hash.
    """
    try:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)
        img = img.convert("L")
        img = img.resize((hash_size, hash_size), Image.Resampling.LANCZOS)

        pixels = list(img.getdata())
        avg = sum(pixels) / len(pixels)

        bits = "".join("1" if p > avg else "0" for p in pixels)
        digest = hex(int(bits, 2))[2:].zfill((hash_size * hash_size) // 4)

        return path, digest, True, ""

    except Exception as e:
        return path, None, False, repr(e)


def sha256_hash(path):
    """
    Exact file hash.
    Only matches if the image bytes are identical.
    Resized/recompressed copies will not match.
    """
    try:
        h = hashlib.sha256()

        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)

        return path, h.hexdigest(), True, ""

    except Exception as e:
        return path, None, False, repr(e)


def hash_worker(args):
    path, hash_type, hash_size = args

    if hash_type == "ahash":
        return average_hash(path, hash_size=hash_size)

    if hash_type == "sha256":
        return sha256_hash(path)

    raise ValueError(f"Unknown hash_type: {hash_type}")

# ---------------------------------------------------------------------
# Parallel hashing with cache
# ---------------------------------------------------------------------

def compute_hash_cache(
    image_paths,
    cache_csv,
    hash_type="ahash",
    hash_size=16,
    workers=16,
    chunksize=128,
):
    """
    Computes hashes only for paths not already present in cache_csv.
    Saves/updates cache_csv.
    """

    image_paths = pd.Series(image_paths).dropna().astype(str).drop_duplicates()
    image_paths = image_paths.tolist()

    cache_csv = Path(cache_csv)
    cache_csv.parent.mkdir(parents=True, exist_ok=True)

    if cache_csv.exists():
        cache = pd.read_csv(cache_csv)
        cache["image_path"] = cache["image_path"].astype(str)

        already_done = set(cache["image_path"])
        todo_paths = [p for p in image_paths if p not in already_done]

        print(f"Loaded existing cache: {cache_csv}")
        print(f"Cached paths: {len(already_done):,}")
        print(f"Remaining paths to hash: {len(todo_paths):,}")

    else:
        cache = pd.DataFrame(columns=["image_path", "img_hash", "ok", "error"])
        todo_paths = image_paths

        print(f"No existing cache found: {cache_csv}")
        print(f"Paths to hash: {len(todo_paths):,}")

    if len(todo_paths) == 0:
        return cache

    tasks = [(p, hash_type, hash_size) for p in todo_paths]

    new_rows = []

    with ProcessPoolExecutor(max_workers=workers) as ex:
        for path, digest, ok, error in tqdm(
            ex.map(hash_worker, tasks, chunksize=chunksize),
            total=len(tasks),
            desc=f"Hashing {hash_type}",
        ):
            new_rows.append(
                {
                    "image_path": path,
                    "img_hash": digest,
                    "ok": ok,
                    "error": error,
                }
            )

    new_cache = pd.DataFrame(new_rows)
    cache = pd.concat([cache, new_cache], ignore_index=True)

    cache = cache.drop_duplicates("image_path", keep="last")
    cache.to_csv(cache_csv, index=False)

    print(f"Saved cache: {cache_csv}")
    print(f"Cache rows: {len(cache):,}")
    print(f"Readable images: {cache['ok'].sum():,}")
    print(f"Failed images: {(~cache['ok']).sum():,}")

    return cache


# ---------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--old-csv",
        default="/root/workspace/PlantCLEF2026/src_experiments/002_bioclip_tile_zero_shot_v2/data/PlantCLEF2024_single_plant_training_metadata.csv",
    )

    parser.add_argument(
        "--new-csv",
        default="/workspace/plantclef/processed/inat_research_grade_manifest_clean.csv",
    )

    parser.add_argument(
        "--old-img-root",
        default="/workspace/plantclef/raw/train/images_max_side_800",
    )

    parser.add_argument(
        "--out-dir",
        default="/workspace/plantclef/processed/old_new_image_overlap",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(16, os.cpu_count() or 1)),
    )

    parser.add_argument(
        "--chunksize",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--hash-type",
        choices=["ahash", "sha256"],
        default="ahash",
        help="ahash = perceptual hash; sha256 = exact byte hash",
    )

    parser.add_argument(
        "--hash-size",
        type=int,
        default=16,
        help="Only used for ahash. 16 gives a 256-bit perceptual hash.",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("========== CONFIG ==========")
    print("old_csv:", args.old_csv)
    print("new_csv:", args.new_csv)
    print("old_img_root:", args.old_img_root)
    print("out_dir:", out_dir)
    print("workers:", args.workers)
    print("chunksize:", args.chunksize)
    print("hash_type:", args.hash_type)
    print("hash_size:", args.hash_size)
    print("============================")

    # -----------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------

    old_data = pd.read_csv(args.old_csv, sep=";", low_memory=False)
    new_data = pd.read_csv(args.new_csv, low_memory=False)

    print("\nLoaded data")
    print("Old rows:", len(old_data))
    print("New rows:", len(new_data))

    # -----------------------------------------------------------------
    # Build old image paths
    # -----------------------------------------------------------------

    old_data["image_path"] = old_data.apply(
        lambda row: str(
            Path(args.old_img_root)
            / str(row["species_id"])
            / str(row["image_name"])
        ),
        axis=1,
    )

    # For safety, dedupe new by image_path again
    new_data = new_data.drop_duplicates("image_path").copy()

    # Old image_name was already unique, but dedupe just in case
    old_data = old_data.drop_duplicates("image_path").copy()

    print("\nAfter image-path dedupe")
    print("Old unique image paths:", old_data["image_path"].nunique())
    print("New unique image paths:", new_data["image_path"].nunique())

    # -----------------------------------------------------------------
    # Check missing files
    # -----------------------------------------------------------------

    print("\nChecking file existence...")

    old_exists = parallel_exists(
        old_data["image_path"],
        workers=args.workers,
        desc="Checking old files",
    )

    new_exists = parallel_exists(
        new_data["image_path"],
        workers=args.workers,
        desc="Checking new files",
    )

    print("Old existing files:", old_exists.sum(), "/", len(old_exists))
    print("New existing files:", new_exists.sum(), "/", len(new_exists))

    old_missing = old_data.loc[~old_exists, ["image_name", "image_path"]]
    new_missing = new_data.loc[~new_exists, ["image_path"]]

    old_missing.to_csv(out_dir / "old_missing_files.csv", index=False)
    new_missing.to_csv(out_dir / "new_missing_files.csv", index=False)

    old_existing_paths = old_data.loc[old_exists, "image_path"].tolist()
    new_existing_paths = new_data.loc[new_exists, "image_path"].tolist()

    # -----------------------------------------------------------------
    # Hash images in parallel, with cache
    # -----------------------------------------------------------------

    old_cache_path = out_dir / f"old_{args.hash_type}_hash_cache.csv"
    new_cache_path = out_dir / f"new_{args.hash_type}_hash_cache.csv"

    old_hash_cache = compute_hash_cache(
        old_existing_paths,
        old_cache_path,
        hash_type=args.hash_type,
        hash_size=args.hash_size,
        workers=args.workers,
        chunksize=args.chunksize,
    )

    new_hash_cache = compute_hash_cache(
        new_existing_paths,
        new_cache_path,
        hash_type=args.hash_type,
        hash_size=args.hash_size,
        workers=args.workers,
        chunksize=args.chunksize,
    )

    # -----------------------------------------------------------------
    # Attach hashes back to metadata
    # -----------------------------------------------------------------

    old_hashed = old_data.merge(
        old_hash_cache,
        on="image_path",
        how="left",
    )

    new_hashed = new_data.merge(
        new_hash_cache,
        on="image_path",
        how="left",
    )

    old_hashed.to_csv(out_dir / "old_with_hashes.csv", index=False)
    new_hashed.to_csv(out_dir / "new_with_hashes.csv", index=False)

    # -----------------------------------------------------------------
    # Compare hashes
    # -----------------------------------------------------------------

    old_valid_hashes = set(
        old_hashed.loc[old_hashed["ok"] == True, "img_hash"].dropna()
    )

    new_valid_hashes = set(
        new_hashed.loc[new_hashed["ok"] == True, "img_hash"].dropna()
    )

    shared_hashes = old_valid_hashes & new_valid_hashes

    print("\n========== HASH OVERLAP ==========")
    print("Old valid unique hashes:", len(old_valid_hashes))
    print("New valid unique hashes:", len(new_valid_hashes))
    print("Shared hashes:", len(shared_hashes))
    print("New-only hashes:", len(new_valid_hashes - old_valid_hashes))
    print("Old-only hashes:", len(old_valid_hashes - new_valid_hashes))
    print("==================================")

    new_hashed["exists_in_old_by_hash"] = new_hashed["img_hash"].isin(old_valid_hashes)
    old_hashed["exists_in_new_by_hash"] = old_hashed["img_hash"].isin(new_valid_hashes)

    new_also_in_old = new_hashed[new_hashed["exists_in_old_by_hash"]].copy()
    new_not_in_old = new_hashed[
        (new_hashed["ok"] == True) & (~new_hashed["exists_in_old_by_hash"])
    ].copy()

    old_also_in_new = old_hashed[old_hashed["exists_in_new_by_hash"]].copy()

    print("\n========== IMAGE ROW OVERLAP ==========")
    print("New image rows also in old by hash:", len(new_also_in_old))
    print("New image rows NOT in old by hash:", len(new_not_in_old))
    print("Old image rows also in new by hash:", len(old_also_in_new))
    print("=======================================")

    # -----------------------------------------------------------------
    # Save split outputs
    # -----------------------------------------------------------------

    new_also_in_old.to_csv(out_dir / "new_images_also_in_old_by_hash.csv", index=False)
    new_not_in_old.to_csv(out_dir / "new_images_not_in_old_by_hash.csv", index=False)
    old_also_in_new.to_csv(out_dir / "old_images_also_in_new_by_hash.csv", index=False)

    # -----------------------------------------------------------------
    # Create matched old/new pair table
    # -----------------------------------------------------------------

    old_match_cols = [
        "image_path",
        "image_name",
        "species_id",
        "gbif_species_id",
        "species",
        "genus",
        "family",
        "license",
        "url",
        "img_hash",
    ]

    old_match_cols = [c for c in old_match_cols if c in old_hashed.columns]

    new_match_cols = [
        "image_path",
        "species_id",
        "gbif_species_id",
        "gbif_occurrence_id",
        "scientific_name",
        "license",
        "url",
        "img_hash",
    ]

    new_match_cols = [c for c in new_match_cols if c in new_hashed.columns]

    matches = new_also_in_old[new_match_cols].merge(
        old_hashed[old_match_cols],
        on="img_hash",
        how="left",
        suffixes=("_new", "_old"),
    )

    matches.to_csv(out_dir / "matched_new_old_pairs_by_hash.csv", index=False)

    print("\nSaved:")
    print(out_dir / "old_with_hashes.csv")
    print(out_dir / "new_with_hashes.csv")
    print(out_dir / "new_images_also_in_old_by_hash.csv")
    print(out_dir / "new_images_not_in_old_by_hash.csv")
    print(out_dir / "matched_new_old_pairs_by_hash.csv")

    # -----------------------------------------------------------------
    # Per-species overlap summary
    # -----------------------------------------------------------------

    if "species_id" in new_hashed.columns:
        species_summary = (
            new_hashed.groupby("species_id")
            .agg(
                new_total_images=("image_path", "count"),
                new_valid_hashes=("img_hash", lambda x: x.notna().sum()),
                new_images_also_in_old=("exists_in_old_by_hash", "sum"),
            )
            .reset_index()
        )

        species_summary["new_images_not_in_old"] = (
            species_summary["new_valid_hashes"]
            - species_summary["new_images_also_in_old"]
        )

        species_summary["pct_new_also_in_old"] = (
            species_summary["new_images_also_in_old"]
            / species_summary["new_valid_hashes"].replace(0, pd.NA)
        )

        species_summary = species_summary.sort_values(
            "new_images_also_in_old",
            ascending=False,
        )

        species_summary.to_csv(
            out_dir / "new_species_overlap_summary_by_hash.csv",
            index=False,
        )

        print(out_dir / "new_species_overlap_summary_by_hash.csv")


if __name__ == "__main__":
    main()