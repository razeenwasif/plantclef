"""
Multi-model BioCLIP zero-shot inference with enriched species prompts.
TPU/XLA-compatible version — optimised for high TPU utilisation.

Key throughput optimisations vs. the original version
------------------------------------------------------
1. Cross-image tile batching
   Tiles from *many* images are accumulated and dispatched as a single large
   forward pass (controlled by --device-batch-size, default 256).  The original
   code sent at most one image's worth of tiles per forward (~4-16 tiles),
   giving <10 % TPU utilisation.  Packing 256+ tiles per pass amortises the
   XLA dispatch overhead and fills the TPU matrix-multiply units.

2. Parallel CPU image loading
   A ThreadPoolExecutor (--num-workers, default 4) loads, decodes, tiles, and
   applies the preprocessing transform to the next batch of images while the
   TPU is busy computing the current batch.  This overlaps I/O with compute
   and eliminates the dominant idle gap.

3. Reduced host-device syncs
   Image features stay on the TPU device until the full cross-image batch is
   ready for aggregation.  mark_step is called once per forward pass (not once
   per image), and .cpu() is called only at the aggregation stage rather than
   inside every encoding loop.

4. Stable tensor shapes
   All tiles are 224×224 after tiling, so every forward pass has shape
   (N, 3, 224, 224) with a fixed N.  This gives the XLA compiler a stable
   graph that can be reused without recompilation.

5. Multi-shard / multi-device support
   --num-shards / --shard-id partition the image list across independent runs
   (e.g., one per TPU core).  Results can be merged externally.
   --multi-device auto-spawns one process per available XLA device using
   torch_xla.distributed.xla_multiprocessing.

Supports BioCLIP 1, 2, and 2.5 with SAHI-style tile max-pool aggregation
and prompt ensembling over multiple text templates per species.

Device selection
----------------
--device auto   Try TPU (XLA) first, then CUDA, then CPU  [default]
--device tpu    Require PyTorch/XLA; error if unavailable
--device cuda   Require CUDA; error if unavailable
--device cpu    Always use CPU

Usage examples
--------------
# Smoke test (5 images, CPU)
python run_inference.py \\
    --model-name hf-hub:imageomics/bioclip \\
    --species-csv path/to/species_lookup_with_gbif_cleaned_names.csv \\
    --images-root path/to/images \\
    --prompt-mode scientific \\
    --device cpu \\
    --limit 5

# TPU single-device, high-throughput run (recommended)
python run_inference.py \\
    --model-name hf-hub:imageomics/bioclip \\
    --species-csv path/to/species_lookup_with_gbif_cleaned_names.csv \\
    --images-root path/to/images \\
    --prompt-mode all \\
    --device tpu \\
    --device-batch-size 256 \\
    --num-workers 8

# TPU multi-device (8 cores, one process per core)
python run_inference.py \\
    --model-name hf-hub:imageomics/bioclip \\
    --images-root path/to/images \\
    --prompt-mode all \\
    --device tpu \\
    --multi-device \\
    --device-batch-size 256 \\
    --num-workers 4

# BioCLIP 2.5 on TPU (lower batch size due to larger model)
python run_inference.py \\
    --model-name hf-hub:imageomics/bioclip-2.5-vith14 \\
    --device-batch-size 64 \\
    --prompt-mode all \\
    --device tpu

Output directory structure
--------------------------
{output_dir}/{run_slug}/
    run_config.json         CLI args + runtime metadata
    prompt_table.csv        species_id, n_prompts, prompts (JSON list)
    prompt_summary.json     aggregate stats about prompts
    submission.csv          PlantCLEF format: quadrat_id, species_ids
    predictions_topk.csv    image_name, rank 1..k, species_id, score
    summary.json            timing, counts, device info
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import open_clip
from PIL import Image

# Local modules
from device_utils import resolve_device, mark_step, device_str
from utils import (
    get_tiles,
    encode_text_features_from_prompts,
    encode_image_tiles,
    compute_tile_logits,
    aggregate_tile_logits,
    image_top_k,
    preprocess_tiles_to_tensor,
    encode_tiles_tensor,
)
from prompt_builder import (
    load_species_labels,
    build_all_prompts,
    prompt_stats,
    PROMPT_MODES,
    SpeciesLabel,
)

# ---------------------------------------------------------------------------
# Known model defaults
# ---------------------------------------------------------------------------

# Fixed chunk size sent to the TPU per forward pass.
# Stable shapes = XLA reuses its compiled graph after the first pass.
# Larger values amortise dispatch overhead; smaller values use less HBM.
_MODEL_TPU_BATCH_SIZE = {
    "hf-hub:imageomics/bioclip":             1024,
    "hf-hub:imageomics/bioclip-2":           1024,
    "hf-hub:imageomics/bioclip-2.5-vith14":   256,
}

# How many tiles to accumulate (across images) before triggering a flush.
# One flush = several fixed-size TPU forward passes.
# Larger values increase images-per-flush, improving amortisation of the
# CPU→device transfer and mark_step overhead.
_MODEL_ACCUMULATE_TILES = {
    "hf-hub:imageomics/bioclip":             4096,
    "hf-hub:imageomics/bioclip-2":           4096,
    "hf-hub:imageomics/bioclip-2.5-vith14":  1024,
}

DEFAULT_SPECIES_CSV = (
    "/root/workspace/PlantCLEF2026/src_experiments/"
    "002_bioclip_tile_zero_shot_v2/data/"
    "species_lookup_with_gbif_cleaned_names.csv"
)
DEFAULT_IMAGES_ROOT = "/workspace/plantclef/kaggle_uploads/test/images"
DEFAULT_OUTPUT_DIR  = "./outputs"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-model BioCLIP zero-shot inference (TPU/XLA, high-throughput)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument(
        "--model-name",
        default="hf-hub:imageomics/bioclip",
        help=(
            "OpenCLIP model identifier. One of: "
            "hf-hub:imageomics/bioclip, "
            "hf-hub:imageomics/bioclip-2, "
            "hf-hub:imageomics/bioclip-2.5-vith14"
        ),
    )

    # Data paths
    parser.add_argument(
        "--species-csv", default=DEFAULT_SPECIES_CSV,
        help="Path to enriched species CSV (species_lookup_with_gbif_cleaned_names.csv)",
    )
    parser.add_argument(
        "--images-root", default=DEFAULT_IMAGES_ROOT,
        help="Directory containing test images (flat, jpg/jpeg/png)",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help="Root output directory. Outputs go under {output_dir}/{run_slug}/",
    )

    # Prompt configuration
    parser.add_argument(
        "--prompt-mode",
        default="scientific",
        choices=list(PROMPT_MODES),
        help=(
            "scientific: family A only | "
            "scientific_common: A+B+D | "
            "scientific_family: A+C | "
            "all: A+B+C+D"
        ),
    )
    parser.add_argument(
        "--max-common-names", type=int, default=3,
        help="Max extra English common names per species (beyond primary)",
    )
    parser.add_argument(
        "--max-synonyms", type=int, default=2,
        help="Max GBIF synonyms to include as additional scientific names",
    )

    # Tiling
    parser.add_argument(
        "--tile-size", type=int, default=224,
        help="Square tile side length in pixels",
    )
    parser.add_argument(
        "--tile-overlap", type=int, default=112,
        help="Overlap between adjacent tiles in pixels (stride = tile_size - tile_overlap)",
    )

    # Inference — TPU batching (two independent knobs)
    parser.add_argument(
        "--tpu-batch-size", type=int, default=None,
        help=(
            "FIXED chunk size (tiles) sent to the TPU in each forward pass. "
            "Every forward call sees this exact shape, so XLA compiles once and "
            "reuses the graph for all subsequent calls. "
            "Defaults: 1024 for BioCLIP 1/2, 256 for 2.5. "
            "Reduce if you get HBM OOM."
        ),
    )
    parser.add_argument(
        "--accumulate-tiles", type=int, default=None,
        help=(
            "Accumulate tiles from this many images before flushing to the TPU. "
            "One flush = several fixed-size forward passes (tpu-batch-size each). "
            "Larger values amortise the per-flush overhead and CPU prefetch cost. "
            "Defaults: 4096 for BioCLIP 1/2, 1024 for 2.5."
        ),
    )
    parser.add_argument(
        "--pad-final-batch", action="store_true",
        help=(
            "Pad the last (potentially short) chunk in each flush to tpu-batch-size "
            "with zero tiles before the forward pass, then discard padded outputs. "
            "This ensures every forward call — including the final chunk — has the "
            "same tensor shape, so XLA never needs a second compilation."
        ),
    )
    # Backward-compatible aliases
    parser.add_argument(
        "--device-batch-size", type=int, default=None,
        help="Alias for --tpu-batch-size (legacy flag).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Alias for --tpu-batch-size (legacy flag).",
    )
    parser.add_argument(
        "--text-batch-size", type=int, default=256,
        help="Text prompt encoding batch size",
    )

    # CPU-side parallelism
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help=(
            "Number of CPU worker threads for parallel image loading, tiling, "
            "and preprocessing.  Higher values overlap more CPU work with TPU "
            "compute.  0 disables threading (serial, useful for debugging)."
        ),
    )
    parser.add_argument(
        "--prefetch-factor", type=int, default=2,
        help=(
            "Number of images to prefetch per worker.  The pipeline keeps "
            "num_workers * prefetch_factor images ready ahead of the current "
            "TPU batch."
        ),
    )

    # Device selection
    parser.add_argument(
        "--device", default="auto",
        choices=["auto", "tpu", "cuda", "cpu"],
        help=(
            "Device: auto (TPU > CUDA > CPU), tpu, cuda, or cpu. "
            "'auto' selects TPU via PyTorch/XLA if available."
        ),
    )
    parser.add_argument(
        "--multi-device", action="store_true",
        help=(
            "Spawn one process per available XLA device (TPU only). "
            "Each process handles a shard of the image list and writes to a "
            "separate output sub-directory.  Requires --device tpu or auto."
        ),
    )

    # Sharding (used internally by --multi-device, or for manual parallelism)
    parser.add_argument(
        "--num-shards", type=int, default=1,
        help="Total number of shards to split the image list across.",
    )
    parser.add_argument(
        "--shard-id", type=int, default=0,
        help="Zero-based index of the shard this process handles.",
    )

    # Output / top-k
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of top species to predict per image",
    )

    # Limiting / debug
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N images (for smoke tests)",
    )
    parser.add_argument(
        "--log-interval", type=int, default=10,
        help=(
            "Print a progress line every N completed images. "
            "Lower values make hangs easier to spot."
        ),
    )
    parser.add_argument(
        "--debug-first-n", type=int, default=0,
        help=(
            "Print per-image and per-stage timing for the first N images and "
            "the first few forward passes.  0 disables (normal run).  "
            "Use --debug-first-n 10 to diagnose hangs."
        ),
    )
    parser.add_argument(
        "--profile", action="store_true",
        help="Enable XLA profiling (TPU only). Writes a trace to the run dir.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_run_slug(model_name: str, prompt_mode: str) -> str:
    """Derive a filesystem-safe run identifier from model + prompt mode."""
    model_slug = model_name.split("/")[-1].replace(".", "-")
    return f"{model_slug}_{prompt_mode}"


def find_images(images_root: str) -> list[Path]:
    exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    paths: list[Path] = []
    for ext in exts:
        paths.extend(Path(images_root).glob(ext))
    return sorted(set(paths))


def load_model(model_name: str, device):
    """Load OpenCLIP model, preprocessing transform, and tokenizer."""
    print(f"  Loading model '{model_name}' ...")
    t0 = time.perf_counter()
    model, _, transform = open_clip.create_model_and_transforms(model_name)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device)
    model.eval()
    elapsed = time.perf_counter() - t0
    print(f"  Model loaded in {elapsed:.1f}s")
    return model, transform, tokenizer


def save_prompt_table(
    out_dir: Path,
    labels: list[SpeciesLabel],
    prompt_lists: list[list[str]],
) -> None:
    path = out_dir / "prompt_table.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["species_id", "canonical_scientific", "primary_common", "n_prompts", "prompts"])
        for label, prompts in zip(labels, prompt_lists):
            writer.writerow([
                label.species_id,
                label.canonical_scientific,
                label.primary_common,
                len(prompts),
                json.dumps(prompts),
            ])
    print(f"  Prompt table saved: {path}")


def save_submission_csv(out_dir: Path, rows: list[dict]) -> None:
    path = out_dir / "submission.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["quadrat_id", "species_ids"], quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Submission CSV saved: {path}  ({len(rows)} rows)")


def save_topk_csv(out_dir: Path, topk_rows: list[dict]) -> None:
    path = out_dir / "predictions_topk.csv"
    fieldnames = ["image_name", "rank", "species_id", "species_name", "logit_score"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(topk_rows)
    print(f"  Top-k predictions saved: {path}  ({len(topk_rows)} rows)")


# ---------------------------------------------------------------------------
# Per-image CPU preparation (runs in worker threads)
# ---------------------------------------------------------------------------

def prepare_image(
    image_path: Path,
    transform,
    tile_size: int,
    stride: int,
) -> tuple[Path, torch.Tensor, int]:
    """
    Load an image, tile it, apply the preprocessing transform, and return a
    stacked tensor of tile features.

    Designed to run in a background thread so CPU work overlaps TPU compute.

    Returns
    -------
    image_path  : the input path (for bookkeeping)
    tile_tensor : (n_tiles, C, H, W) float tensor on CPU
    n_tiles     : number of tiles (same as tile_tensor.shape[0])
    """
    image = Image.open(image_path).convert("RGB")
    tiles, _ = get_tiles(image, tile_size, stride)
    tile_tensor = preprocess_tiles_to_tensor(tiles, transform)
    return image_path, tile_tensor, len(tiles)


# ---------------------------------------------------------------------------
# Batched cross-image inference  (the core throughput loop)
# ---------------------------------------------------------------------------

class _Timers:
    """Lightweight accumulator for per-stage wall-clock times."""

    def __init__(self):
        self._totals: dict[str, float] = collections.defaultdict(float)
        self._counts: dict[str, int]   = collections.defaultdict(int)

    def add(self, stage: str, elapsed: float, count: int = 1) -> None:
        self._totals[stage] += elapsed
        self._counts[stage] += count

    def report(self) -> dict[str, Any]:
        out = {}
        for stage, total in sorted(self._totals.items()):
            n = self._counts[stage]
            out[stage] = {
                "total_secs": round(total, 3),
                "count":      n,
                "avg_secs":   round(total / n, 6) if n else 0.0,
            }
        return out

    def print_report(self) -> None:
        print("\n--- Per-stage timing ---")
        for stage, d in self.report().items():
            print(
                f"  {stage:<25s}  total={d['total_secs']:8.2f}s  "
                f"n={d['count']:>6d}  avg={d['avg_secs']*1000:7.2f}ms"
            )


def run_batched_inference(
    image_paths: list[Path],
    model,
    transform,
    text_feats: torch.Tensor,
    species_ids: list[str],
    logit_scale: float,
    tile_size: int,
    stride: int,
    tpu_batch_size: int,
    accumulate_tiles: int,
    top_k: int,
    device,
    backend: str,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    log_interval: int = 10,
    debug_n: int = 0,
    pad_final_batch: bool = False,
) -> tuple[dict[Path, tuple[list[str], list[float]]], _Timers, dict]:
    """
    Run inference over all images using two-level batching and parallel
    CPU prefetching.

    Two-level batching design
    -------------------------
    Level 1 — accumulation (CPU):
        Images are loaded in parallel and their tiles are accumulated in a
        pending buffer until pending_tile_count >= accumulate_tiles.  This
        amortises the per-flush overhead across many images.

    Level 2 — TPU forward (device):
        Inside each flush, all accumulated tiles are processed in fixed-size
        chunks of exactly tpu_batch_size.  Every forward call has the same
        tensor shape (tpu_batch_size, C, H, W), so XLA compiles once and
        reuses the graph for all subsequent calls.

        If pad_final_batch is True, the last (short) chunk in each flush is
        zero-padded to tpu_batch_size before the forward, and the padded
        outputs are discarded afterwards.  This eliminates the second XLA
        compilation that would otherwise occur for the final partial chunk.

    Aggregation:
        All chunk logits are collected on CPU.  Per-image max-pool runs on
        CPU, avoiding variable XLA graph sizes from per-image slice+max.

    Parameters
    ----------
    tpu_batch_size    : tiles per TPU forward pass (fixed; stable XLA shape)
    accumulate_tiles  : flush threshold (how many CPU tiles to collect first)
    pad_final_batch   : pad last chunk → every forward has identical shape
    debug_n           : per-image and per-stage timing for first N images
    log_interval      : progress line every N completed images

    Returns
    -------
    results  : {image_path: (top_ids, top_scores)}  in input order
    timers   : per-stage timing accumulator
    counters : aggregate counts
    """
    _mark = lambda: mark_step(backend)  # noqa: E731
    timers   = _Timers()
    results: dict[Path, tuple[list[str], list[float]]] = {}
    t0_global = time.perf_counter()

    def _dbg(msg: str) -> None:
        if debug_n > 0:
            print(f"  [DBG +{time.perf_counter()-t0_global:6.2f}s] {msg}", flush=True)

    # Global counters
    n_tiles_total      = 0
    n_forward_passes   = 0
    n_padded_chunks    = 0
    n_flush_calls      = 0
    n_images_ok        = 0
    n_errors           = 0
    # Running total for the pending buffer (avoids O(N) sum in hot loop)
    pending_tile_count = 0

    # Pending buffer: (image_path, tile_tensor, n_tiles)
    pending: list[tuple[Path, torch.Tensor, int]] = []

    # -----------------------------------------------------------------------
    def flush_pending() -> None:
        """
        Cat all pending tiles, run fixed-size forward passes, collect logits
        on CPU, then aggregate per image (max-pool) on CPU.

        Shape contract
        --------------
        Every forward pass receives exactly tpu_batch_size tiles.  If the
        last chunk is shorter, it is either left short (one extra compile for
        the final shape) or padded to tpu_batch_size (--pad-final-batch),
        which guarantees every call sees an identical shape and XLA compiles
        exactly once.
        """
        nonlocal n_tiles_total, n_forward_passes, n_padded_chunks
        nonlocal n_flush_calls, pending_tile_count

        if not pending:
            return

        n_imgs_in_flush = len(pending)
        n_flush_calls  += 1
        first_flush     = (n_flush_calls == 1)
        verbose         = debug_n > 0 and (first_flush or n_flush_calls <= 3)

        if verbose:
            _dbg(
                f"flush #{n_flush_calls}: {n_imgs_in_flush} images  "
                f"{pending_tile_count} tiles  "
                f"tpu_batch={tpu_batch_size}  "
                f"chunks={max(1, (pending_tile_count + tpu_batch_size - 1) // tpu_batch_size)}"
            )

        # --- cat all pending tiles into one CPU tensor ---
        t_cat = time.perf_counter()
        all_tiles = torch.cat([t for _, t, _ in pending], dim=0)  # (N_total, C, H, W)
        total_pending = len(all_tiles)
        timers.add("cat_tiles", time.perf_counter() - t_cat)
        if verbose:
            _dbg(f"  cat_tiles done  shape={list(all_tiles.shape)}  "
                 f"{(time.perf_counter()-t_cat)*1000:.1f}ms")

        # --- chunked forward passes with stable shapes ---
        chunk_logits_cpu: list[torch.Tensor] = []
        pos       = 0
        chunk_idx = 0

        while pos < total_pending:
            end    = min(pos + tpu_batch_size, total_pending)
            chunk  = all_tiles[pos:end]
            real_n = len(chunk)
            padded = False

            if pad_final_batch and real_n < tpu_batch_size:
                # Zero-pad so XLA sees the same (tpu_batch_size, C, H, W) shape
                # it compiled for, avoiding a second compilation for the tail.
                pad_n  = tpu_batch_size - real_n
                chunk  = torch.cat(
                    [chunk, chunk.new_zeros(pad_n, *chunk.shape[1:])], dim=0
                )
                padded = True
                n_padded_chunks += 1

            is_first_ever = (n_forward_passes == 0)
            if is_first_ever:
                print(
                    f"\n  [DBG +{time.perf_counter()-t0_global:.2f}s] "
                    f"First TPU forward: shape={list(chunk.shape)}  "
                    f"({'padded, ' if padded else ''}real_n={real_n})  "
                    f"XLA graph compile may take 30-180s — this is expected.",
                    flush=True,
                )
            elif verbose:
                _dbg(
                    f"  chunk {chunk_idx}: shape={list(chunk.shape)}  "
                    f"real_n={real_n}  {'[PADDED]' if padded else ''}"
                )

            t_fwd  = time.perf_counter()
            feats  = encode_tiles_tensor(model, chunk, device, mark_step_fn=_mark)
            fwd_s  = time.perf_counter() - t_fwd

            if padded:
                feats = feats[:real_n]   # discard outputs for padded tiles

            # matmul: (real_n, dim) @ (dim, n_species) → (real_n, n_species)
            # This is lazy on XLA; the .cpu() below triggers the sync.
            t_logit = time.perf_counter()
            logits  = logit_scale * feats @ text_feats.T
            timers.add("matmul_logits", time.perf_counter() - t_logit)

            # Move chunk logits to CPU now to keep peak HBM usage bounded.
            chunk_logits_cpu.append(logits.cpu())
            _mark()

            timers.add("forward_pass", fwd_s, count=real_n)
            n_tiles_total  += real_n
            n_forward_passes += 1

            if is_first_ever or verbose:
                _dbg(
                    f"  chunk {chunk_idx}: forward done  {fwd_s:.2f}s  "
                    f"({'XLA compile included' if is_first_ever else 'graph reused'})"
                )

            pos       += tpu_batch_size   # advance by fixed stride, not real_n
            chunk_idx += 1

        # --- aggregate per image on CPU ---
        # All logits are already on CPU; no XLA graph involved here.
        t_agg = time.perf_counter()
        all_logits_cpu = torch.cat(chunk_logits_cpu, dim=0)   # (N_total, n_species)

        offset     = 0
        agg_list   = []
        paths_order = []
        for path, _, n in pending:
            chunk_l = all_logits_cpu[offset : offset + n]   # CPU slice
            agg     = chunk_l.max(dim=0).values             # CPU max-pool
            agg_list.append(agg)
            paths_order.append(path)
            offset += n
        cpu_logits = torch.stack(agg_list, dim=0)  # (n_images, n_species)
        timers.add("aggregate_cpu", time.perf_counter() - t_agg, count=n_imgs_in_flush)
        if verbose:
            _dbg(f"  aggregate_cpu done  {(time.perf_counter()-t_agg)*1000:.1f}ms")

        # --- top-k ---
        t_topk = time.perf_counter()
        for idx, path in enumerate(paths_order):
            top_ids, top_scores = image_top_k(cpu_logits[idx], species_ids, k=top_k)
            results[path] = (top_ids, top_scores)
        timers.add("topk", time.perf_counter() - t_topk, count=n_imgs_in_flush)
        if verbose:
            _dbg(
                f"  topk done  {(time.perf_counter()-t_topk)*1000:.1f}ms  "
                f"flush #{n_flush_calls} complete"
            )

        pending.clear()
        pending_tile_count = 0
    # -----------------------------------------------------------------------

    max_prefetch = max(1, num_workers * prefetch_factor)

    if num_workers == 0:
        # --- Serial mode (num_workers=0, useful for debugging) ---
        _dbg("Serial mode (num_workers=0)")
        t0_infer = time.perf_counter()
        for i, image_path in enumerate(image_paths):
            is_debug_img = (debug_n > 0 and i < debug_n)
            if is_debug_img:
                _dbg(f"img {i}: prepare_image start  {image_path.name}")
            t_load = time.perf_counter()
            try:
                _, tile_tensor, n_tiles = prepare_image(
                    image_path, transform, tile_size, stride
                )
            except Exception as exc:
                load_ms = (time.perf_counter() - t_load) * 1000
                print(
                    f"  [WARN +{time.perf_counter()-t0_global:.2f}s] "
                    f"{image_path.name}: prepare_image failed after {load_ms:.0f}ms — {exc}",
                    flush=True,
                )
                n_errors += 1
                continue
            load_ms = (time.perf_counter() - t_load) * 1000
            timers.add("image_load_tile_transform", time.perf_counter() - t_load)
            if is_debug_img:
                _dbg(
                    f"img {i}: prepare_image done  {load_ms:.0f}ms  "
                    f"tiles={n_tiles}  tensor={list(tile_tensor.shape)}"
                )

            pending.append((image_path, tile_tensor, n_tiles))
            pending_tile_count += n_tiles
            n_images_ok += 1

            if is_debug_img:
                _dbg(
                    f"img {i}: pending_tiles={pending_tile_count}/{accumulate_tiles}  "
                    f"({len(pending)} images buffered)"
                )

            if pending_tile_count >= accumulate_tiles:
                if debug_n > 0:
                    _dbg(
                        f"pending_tiles={pending_tile_count} >= "
                        f"accumulate_tiles={accumulate_tiles} → flushing "
                        f"({len(pending)} images)"
                    )
                flush_pending()

            if (i + 1) % log_interval == 0 or (i + 1) == len(image_paths):
                elapsed = time.perf_counter() - t0_infer
                ips = (i + 1) / elapsed if elapsed > 0 else 0
                avg_chunk = n_tiles_total / n_forward_passes if n_forward_passes else 0
                print(
                    f"  [{i+1:>6}/{len(image_paths)}] "
                    f"{ips:.2f} img/s  "
                    f"fwd={n_forward_passes}  tiles={n_tiles_total}  "
                    f"avg_chunk={avg_chunk:.0f}",
                    flush=True,
                )

        flush_pending()

    else:
        # --- Parallel mode: sliding prefetch window ---
        # futures_window stores (path, future) pairs so exceptions include filename.
        futures_window: deque = deque()   # deque of (Path, Future)
        path_iter = iter(image_paths)
        submitted = 0
        t0_infer  = time.perf_counter()

        _dbg(
            f"Parallel mode  num_workers={num_workers}  "
            f"prefetch_factor={prefetch_factor}  "
            f"max_prefetch={max_prefetch}  "
            f"accumulate_tiles={accumulate_tiles}  "
            f"tpu_batch_size={tpu_batch_size}  "
            f"pad_final_batch={pad_final_batch}"
        )

        def _submit_next(executor: ThreadPoolExecutor) -> bool:
            nonlocal submitted
            p = next(path_iter, None)
            if p is None:
                return False
            futures_window.append(
                (p, executor.submit(prepare_image, p, transform, tile_size, stride))
            )
            submitted += 1
            return True

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for _ in range(max_prefetch):
                if not _submit_next(executor):
                    break
            _dbg(f"Seeded {len(futures_window)} prefetch futures")

            completed = 0
            while futures_window:
                path, future = futures_window.popleft()
                _submit_next(executor)   # keep the window full

                is_debug_img = (debug_n > 0 and completed < debug_n)
                if is_debug_img:
                    _dbg(f"img {completed}: blocking on future.result()  {path.name}")
                t_wait = time.perf_counter()
                try:
                    _, tile_tensor, n_tiles = future.result()
                except Exception as exc:
                    wait_ms = (time.perf_counter() - t_wait) * 1000
                    timers.add("image_load_tile_transform", time.perf_counter() - t_wait)
                    print(
                        f"  [WARN +{time.perf_counter()-t0_global:.2f}s] "
                        f"{path.name}: prepare_image failed after {wait_ms:.0f}ms — {exc}",
                        flush=True,
                    )
                    n_errors += 1
                    completed += 1
                    continue
                wait_ms = (time.perf_counter() - t_wait) * 1000
                timers.add("image_load_tile_transform", time.perf_counter() - t_wait)

                if is_debug_img:
                    _dbg(
                        f"img {completed}: future.result() returned  "
                        f"wait={wait_ms:.0f}ms  tiles={n_tiles}  "
                        f"tensor={list(tile_tensor.shape)}  {path.name}"
                    )

                pending.append((path, tile_tensor, n_tiles))
                pending_tile_count += n_tiles
                n_images_ok += 1
                completed += 1

                if is_debug_img:
                    _dbg(
                        f"img {completed-1}: pending_tiles={pending_tile_count}/"
                        f"{accumulate_tiles}  ({len(pending)} images buffered)"
                    )

                if pending_tile_count >= accumulate_tiles:
                    if debug_n > 0:
                        _dbg(
                            f"pending_tiles={pending_tile_count} >= "
                            f"accumulate_tiles={accumulate_tiles} → flushing "
                            f"({len(pending)} images)"
                        )
                    flush_pending()

                if completed % log_interval == 0 or completed == len(image_paths):
                    elapsed = time.perf_counter() - t0_infer
                    ips     = completed / elapsed if elapsed > 0 else 0
                    avg_chunk = n_tiles_total / n_forward_passes if n_forward_passes else 0
                    print(
                        f"  [{completed:>6}/{len(image_paths)}] "
                        f"{ips:.2f} img/s  "
                        f"fwd={n_forward_passes}  tiles={n_tiles_total}  "
                        f"avg_chunk={avg_chunk:.0f}",
                        flush=True,
                    )

            if pending_tile_count > 0:
                if debug_n > 0:
                    _dbg(
                        f"Final flush: {len(pending)} images  "
                        f"{pending_tile_count} tiles"
                    )
                flush_pending()

    avg_chunk = round(n_tiles_total / n_forward_passes, 1) if n_forward_passes else 0.0
    counters = {
        "n_images_ok":          n_images_ok,
        "n_errors":             n_errors,
        "n_tiles_total":        n_tiles_total,
        "n_forward_passes":     n_forward_passes,
        "n_flush_calls":        n_flush_calls,
        "n_padded_chunks":      n_padded_chunks,
        "avg_tiles_per_img":    round(n_tiles_total / max(n_images_ok, 1), 2),
        "avg_chunk_size":       avg_chunk,
        "avg_images_per_flush": round(n_images_ok / max(n_flush_calls, 1), 1),
    }
    return results, timers, counters


# ---------------------------------------------------------------------------
# Multi-device support via xmp.spawn
# ---------------------------------------------------------------------------

def _xmp_worker(rank: int, args, image_paths: list[Path]) -> None:
    """
    Worker function for xmp.spawn multi-device inference.

    Each rank gets its own XLA device and processes a shard of image_paths.
    Results are written to {out_dir}/shard_{rank}/ and merged by rank 0.
    """
    import torch_xla.core.xla_model as xm

    device  = xm.xla_device()
    backend = "xla"
    world   = xm.xrt_world_size()

    # Shard this rank's slice of images
    shard_paths = image_paths[rank::world]
    print(f"  [rank {rank}/{world}] device={device}  images={len(shard_paths)}")

    # Each rank builds its own out_dir sub-directory
    base_out = Path(args.output_dir) / make_run_slug(args.model_name, args.prompt_mode)
    out_dir  = base_out / f"shard_{rank}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Re-run the single-device inference on this shard
    _run_single_device(args, device, backend, shard_paths, out_dir, shard_id=rank)

    xm.rendezvous("inference_done")

    # Rank 0 merges all shards
    if rank == 0:
        _merge_shards(base_out, world)


def _merge_shards(base_out: Path, world: int) -> None:
    """Merge per-shard submission and top-k CSVs into the run root directory."""
    print(f"\n[rank 0] Merging {world} shards ...")
    all_submission: list[dict] = []
    all_topk: list[dict] = []

    for rank in range(world):
        shard_dir = base_out / f"shard_{rank}"
        sub_path = shard_dir / "submission.csv"
        topk_path = shard_dir / "predictions_topk.csv"

        if sub_path.exists():
            with open(sub_path, newline="") as f:
                all_submission.extend(csv.DictReader(f))
        if topk_path.exists():
            with open(topk_path, newline="") as f:
                all_topk.extend(csv.DictReader(f))

    # Sort by quadrat_id for determinism
    all_submission.sort(key=lambda r: r["quadrat_id"])
    all_topk.sort(key=lambda r: (r["image_name"], int(r["rank"])))

    save_submission_csv(base_out, all_submission)
    save_topk_csv(base_out, all_topk)
    print(f"  Merged: {len(all_submission)} images across {world} shards")


# ---------------------------------------------------------------------------
# Single-device inference entry point
# ---------------------------------------------------------------------------

def _run_single_device(
    args,
    device,
    backend: str,
    image_paths: list[Path],
    out_dir: Path,
    shard_id: int = 0,
) -> None:
    """
    Full inference pipeline on a single device.  Called for both the
    non-multi-device path and by each xmp.spawn worker.
    """
    # Resolve tpu_batch_size: new flag > legacy aliases > model default
    tpu_batch_size = (
        args.tpu_batch_size
        or args.device_batch_size
        or args.batch_size
        or _MODEL_TPU_BATCH_SIZE.get(args.model_name, 1024)
    )
    # Resolve accumulate_tiles: new flag > model default
    accumulate_tiles = (
        args.accumulate_tiles
        or _MODEL_ACCUMULATE_TILES.get(args.model_name, 4096)
    )
    # accumulate_tiles must be >= tpu_batch_size to guarantee at least one full chunk
    if accumulate_tiles < tpu_batch_size:
        accumulate_tiles = tpu_batch_size

    stride = args.tile_size - args.tile_overlap

    t0_startup = time.perf_counter()
    print(f"\n[shard {shard_id}] Loading model ...")
    model, transform, tokenizer = load_model(args.model_name, device)
    logit_scale = model.logit_scale.exp().item()
    mark_step(backend)
    print(f"  logit_scale: {logit_scale:.4f}")
    startup_secs = time.perf_counter() - t0_startup

    # ----- Load species labels -----
    t0_species = time.perf_counter()
    print(f"[shard {shard_id}] Loading species from: {args.species_csv}")
    labels = load_species_labels(
        args.species_csv,
        max_common_names=args.max_common_names,
        max_synonyms=args.max_synonyms,
    )
    species_ids = [lbl.species_id for lbl in labels]
    id_to_name  = {lbl.species_id: lbl.canonical_scientific for lbl in labels}
    print(f"  {len(labels)} species loaded  ({time.perf_counter()-t0_species:.2f}s)")

    # ----- Build prompts -----
    t0_prompt = time.perf_counter()
    print(f"[shard {shard_id}] Building prompts (mode='{args.prompt_mode}') ...")
    prompt_lists = build_all_prompts(labels, args.prompt_mode)
    stats = prompt_stats(prompt_lists)
    print(
        f"  {stats['total_prompts']} total prompts  "
        f"({stats['min_per_species']}-{stats['max_per_species']} per species, "
        f"avg {stats['avg_per_species']})  "
        f"({time.perf_counter()-t0_prompt:.2f}s)"
    )
    if shard_id == 0:
        save_prompt_table(out_dir, labels, prompt_lists)
        with open(out_dir / "prompt_summary.json", "w") as f:
            json.dump({**stats, "prompt_mode": args.prompt_mode}, f, indent=2)

    # ----- Encode text features -----
    print(f"[shard {shard_id}] Encoding species text features "
          f"(text_batch_size={args.text_batch_size}) ...")
    t0_text = time.perf_counter()
    _mark = lambda: mark_step(backend)  # noqa: E731
    text_feats = encode_text_features_from_prompts(
        model, tokenizer, prompt_lists, device,
        batch_size=args.text_batch_size,
        mark_step_fn=_mark,
    )
    # Keep text features on device for efficient matmul during inference
    text_feats = text_feats.to(device)
    mark_step(backend)
    text_encoding_secs = time.perf_counter() - t0_text
    print(f"  text_feats: {text_feats.shape}  ({text_encoding_secs:.2f}s)")

    # ----- Run batched inference -----
    print(
        f"\n[shard {shard_id}] Running batched inference  "
        f"(accumulate_tiles={accumulate_tiles}  tpu_batch_size={tpu_batch_size}  "
        f"pad_final_batch={args.pad_final_batch}  "
        f"num_workers={args.num_workers}  images={len(image_paths)}) ...\n"
    )
    t0_infer = time.perf_counter()

    if args.profile and backend == "xla":
        import torch_xla.debug.profiler as xp
        server = xp.start_server(9012)
        print(f"  XLA profiler started on port 9012")

    results, timers, counters = run_batched_inference(
        image_paths      = image_paths,
        model            = model,
        transform        = transform,
        text_feats       = text_feats,
        species_ids      = species_ids,
        logit_scale      = logit_scale,
        tile_size        = args.tile_size,
        stride           = stride,
        tpu_batch_size   = tpu_batch_size,
        accumulate_tiles = accumulate_tiles,
        top_k            = args.top_k,
        device           = device,
        backend          = backend,
        num_workers      = args.num_workers,
        prefetch_factor  = args.prefetch_factor,
        log_interval     = args.log_interval,
        debug_n          = args.debug_first_n,
        pad_final_batch  = args.pad_final_batch,
    )
    total_infer_secs = time.perf_counter() - t0_infer

    # ----- Collect output rows (in original image order) -----
    submission_rows: list[dict] = []
    topk_rows: list[dict] = []

    for image_path in image_paths:
        if image_path not in results:
            continue
        quadrat_id = image_path.stem
        top_ids, top_scores = results[image_path]
        ids_str = "[" + ", ".join(top_ids) + "]"
        submission_rows.append({"quadrat_id": quadrat_id, "species_ids": ids_str})
        for rank_i, (sid, score) in enumerate(zip(top_ids, top_scores), start=1):
            topk_rows.append({
                "image_name":   quadrat_id,
                "rank":         rank_i,
                "species_id":   sid,
                "species_name": id_to_name.get(sid, ""),
                "logit_score":  f"{score:.6f}",
            })

    n_processed = len(submission_rows)

    # ----- Save outputs -----
    print()
    save_submission_csv(out_dir, submission_rows)
    save_topk_csv(out_dir, topk_rows)

    # ----- Print and save timing summary -----
    timers.print_report()

    ips = n_processed / total_infer_secs if total_infer_secs > 0 else 0
    tps = counters["n_tiles_total"] / total_infer_secs if total_infer_secs > 0 else 0

    print(f"\n--- Aggregate counters ---")
    print(f"  images processed    : {n_processed}")
    print(f"  errors              : {counters['n_errors']}")
    print(f"  total tiles         : {counters['n_tiles_total']}")
    print(f"  flush calls         : {counters['n_flush_calls']}")
    print(f"  forward passes      : {counters['n_forward_passes']}")
    print(f"  padded chunks       : {counters['n_padded_chunks']}")
    print(f"  avg tiles/image     : {counters['avg_tiles_per_img']:.1f}")
    print(f"  avg chunk size      : {counters['avg_chunk_size']:.1f}  "
          f"(target={tpu_batch_size})")
    print(f"  avg images/flush    : {counters['avg_images_per_flush']:.1f}  "
          f"(target≈{accumulate_tiles // max(1, round(counters['avg_tiles_per_img']))})")
    print(f"  throughput          : {ips:.2f} img/s  |  {tps:.1f} tiles/s")

    summary = {
        "run_slug":                make_run_slug(args.model_name, args.prompt_mode),
        "shard_id":                shard_id,
        "model_name":              args.model_name,
        "prompt_mode":             args.prompt_mode,
        "n_species":               len(labels),
        **stats,
        "n_images_found":          len(image_paths),
        "n_images_processed":      n_processed,
        "n_errors":                counters["n_errors"],
        "top_k":                   args.top_k,
        "tile_size":               args.tile_size,
        "tile_overlap":            args.tile_overlap,
        "stride":                  stride,
        "tpu_batch_size":          tpu_batch_size,
        "accumulate_tiles":        accumulate_tiles,
        "pad_final_batch":         args.pad_final_batch,
        "num_workers":             args.num_workers,
        "prefetch_factor":         args.prefetch_factor,
        "n_tiles_total":           counters["n_tiles_total"],
        "n_forward_passes":        counters["n_forward_passes"],
        "n_flush_calls":           counters["n_flush_calls"],
        "n_padded_chunks":         counters["n_padded_chunks"],
        "avg_tiles_per_image":     counters["avg_tiles_per_img"],
        "avg_chunk_size":          counters["avg_chunk_size"],
        "avg_images_per_flush":    counters["avg_images_per_flush"],
        "throughput_images_per_sec":  round(ips, 3),
        "throughput_tiles_per_sec":   round(tps, 1),
        "startup_secs":            round(startup_secs, 2),
        "text_encoding_secs":      round(text_encoding_secs, 2),
        "inference_total_secs":    round(total_infer_secs, 2),
        "inference_per_image_secs": round(total_infer_secs / max(n_processed, 1), 4),
        "stage_timers":            timers.report(),
        "device":                  device_str(device),
        "backend":                 backend,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved: {out_dir / 'summary.json'}")

    # ----- Final banner -----
    run_slug = make_run_slug(args.model_name, args.prompt_mode)
    print(f"\n{'='*60}")
    print(f"Run complete: {run_slug}  [shard {shard_id}]")
    print(f"  Device           : {device_str(device)}  (backend={backend})")
    print(f"  tpu_batch_size   : {tpu_batch_size}")
    print(f"  accumulate_tiles : {accumulate_tiles}")
    print(f"  pad_final_batch  : {args.pad_final_batch}")
    print(f"  num_workers      : {args.num_workers}")
    print(f"  Images processed : {n_processed} / {len(image_paths)}")
    if counters["n_errors"]:
        print(f"  Errors           : {counters['n_errors']}")
    print(f"  Startup          : {startup_secs:.1f}s")
    print(f"  Text encoding    : {text_encoding_secs:.1f}s")
    print(f"  Inference        : {total_infer_secs:.1f}s  ({ips:.2f} img/s)")
    print(f"  Outputs          : {out_dir}/")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ----- Device resolution -----
    device, backend = resolve_device(args.device)
    print(f"\nDevice: {device_str(device)}  (backend={backend})")

    stride = args.tile_size - args.tile_overlap
    if stride <= 0:
        raise ValueError(
            f"tile-overlap ({args.tile_overlap}) must be less than "
            f"tile-size ({args.tile_size})"
        )

    tpu_batch_size = (
        args.tpu_batch_size
        or args.device_batch_size
        or args.batch_size
        or _MODEL_TPU_BATCH_SIZE.get(args.model_name, 1024)
    )
    accumulate_tiles = (
        args.accumulate_tiles
        or _MODEL_ACCUMULATE_TILES.get(args.model_name, 4096)
    )
    if accumulate_tiles < tpu_batch_size:
        accumulate_tiles = tpu_batch_size

    run_slug = make_run_slug(args.model_name, args.prompt_mode)
    out_dir  = Path(args.output_dir) / run_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Run: {run_slug}")
    print(f"Device: {device_str(device)}  (backend={backend})")
    print(f"tile_size={args.tile_size}  stride={stride}  top_k={args.top_k}")
    print(
        f"tpu_batch_size={tpu_batch_size}  accumulate_tiles={accumulate_tiles}  "
        f"pad_final_batch={args.pad_final_batch}"
    )
    print(
        f"num_workers={args.num_workers}  prefetch_factor={args.prefetch_factor}"
    )
    print(f"Output dir: {out_dir}")
    print(f"{'='*60}\n")

    # ----- Save run config -----
    config = vars(args).copy()
    config.update({
        "device_resolved":       device_str(device),
        "backend":               backend,
        "stride":                stride,
        "tpu_batch_size_used":   tpu_batch_size,
        "accumulate_tiles_used": accumulate_tiles,
        "run_slug":              run_slug,
        "output_path":           str(out_dir),
    })
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ----- Find images -----
    if not Path(args.images_root).exists():
        sys.exit(f"ERROR: images-root not found: {args.images_root}")
    image_paths = find_images(args.images_root)
    if args.limit:
        image_paths = image_paths[: args.limit]
    print(f"Found {len(image_paths)} images in: {args.images_root}")

    if not image_paths:
        sys.exit("ERROR: no images found - check --images-root")

    # ----- Validate species CSV -----
    if not Path(args.species_csv).exists():
        sys.exit(f"ERROR: species CSV not found: {args.species_csv}")

    # ----- Multi-device via xmp.spawn -----
    if args.multi_device:
        if backend != "xla":
            sys.exit("ERROR: --multi-device requires --device tpu (or auto with TPU present)")
        import torch_xla.distributed.xla_multiprocessing as xmp
        import torch_xla.core.xla_model as xm
        n_devices = xm.xrt_world_size()
        print(f"Spawning {n_devices} processes (one per XLA device) ...")
        xmp.spawn(
            _xmp_worker,
            args=(args, image_paths),
            nprocs=n_devices,
            start_method="fork",
        )
        return

    # ----- Manual shard selection (for external parallelism) -----
    if args.num_shards > 1:
        image_paths = image_paths[args.shard_id :: args.num_shards]
        print(f"Shard {args.shard_id}/{args.num_shards}: {len(image_paths)} images")
        out_dir = out_dir / f"shard_{args.shard_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

    # ----- Single-device run -----
    _run_single_device(
        args        = args,
        device      = device,
        backend     = backend,
        image_paths = image_paths,
        out_dir     = out_dir,
        shard_id    = args.shard_id,
    )


if __name__ == "__main__":
    main()
