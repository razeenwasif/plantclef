"""
Download a SAM checkpoint for use with experiment 007.

Usage
-----
# Download the default (smallest) model — sam_vit_b (~375 MB)
python download_sam.py

# Explicitly choose a model size
python download_sam.py --model-type vit_l   # ~1.2 GB
python download_sam.py --model-type vit_h   # ~2.6 GB

# Custom save location
python download_sam.py --out ./my_checkpoints/sam_vit_b.pth
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

# Official Meta SAM checkpoint URLs
SAM_URLS: dict[str, tuple[str, str]] = {
    "vit_b": (
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "01ec64d29a2fca3f0661936605ae66f8",  # MD5 (informational)
    ),
    "vit_l": (
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "0b3195507c641ddb6910d2bb5adee89c",
    ),
    "vit_h": (
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "4b8939a88964f0f4ff5f5b2642c598a6",
    ),
}

DEFAULT_OUT: dict[str, str] = {
    "vit_b": "./checkpoints/sam_vit_b_01ec64.pth",
    "vit_l": "./checkpoints/sam_vit_l_0b3195.pth",
    "vit_h": "./checkpoints/sam_vit_h_4b8939.pth",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download SAM checkpoint")
    p.add_argument("--model-type", default="vit_b", choices=list(SAM_URLS))
    p.add_argument("--out", default=None,
                   help="Save path. Defaults to ./checkpoints/sam_vit_<type>.pth")
    return p.parse_args()


def download_with_progress(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)

    def _reporthook(count, block_size, total_size):
        if total_size <= 0:
            return
        pct = min(100, count * block_size * 100 // total_size)
        mb_done  = count * block_size / 1024 / 1024
        mb_total = total_size / 1024 / 1024
        print(f"\r  {pct:3d}%  {mb_done:.1f}/{mb_total:.1f} MB", end="", flush=True)

    print(f"Downloading from:\n  {url}")
    print(f"Saving to:\n  {dest}\n")
    urllib.request.urlretrieve(url, dest, reporthook=_reporthook)
    print()  # newline after progress bar


def main() -> None:
    args = parse_args()
    url, md5_hint = SAM_URLS[args.model_type]
    out_path = Path(args.out or DEFAULT_OUT[args.model_type])

    if out_path.exists():
        size_mb = out_path.stat().st_size / 1024 / 1024
        print(f"Checkpoint already exists at {out_path}  ({size_mb:.1f} MB)")
        print("Delete it first if you want to re-download.")
        return

    try:
        download_with_progress(url, out_path)
    except KeyboardInterrupt:
        print("\nDownload interrupted.")
        if out_path.exists():
            out_path.unlink()
        sys.exit(1)

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"Download complete: {out_path}  ({size_mb:.1f} MB)")
    print(f"\nYou can now use --scoring sam --sam-checkpoint {out_path} in run_inference.py")


if __name__ == "__main__":
    main()
