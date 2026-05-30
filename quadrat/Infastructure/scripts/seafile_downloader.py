"""
Seafile Shared Link Downloader

Downloads all files from a Seafile shared directory link.
Supports resume, parallel downloads, and progress tracking.

Usage:
    python3 scripts/seafile_downloader.py \
        --url https://lab.plantnet.org/seafile/d/303fec50b1a544c6a2ed \
        --output /workspace/plantclef/raw/train

    python3 scripts/seafile_downloader.py \
        --url https://lab.plantnet.org/seafile/d/f3a63defc5f44220b194 \
        --output /workspace/plantclef/raw/pseudo_quadrats
"""

import argparse
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "requests", "tqdm"])
    import requests
    from tqdm import tqdm


def extract_share_token(url):
    match = re.search(r"/d/([a-f0-9]+)", url)
    if not match:
        raise ValueError(f"Could not extract share token from URL: {url}")
    return match.group(1)


def get_base_url(url):
    match = re.match(r"(https?://[^/]+)", url)
    if not match:
        raise ValueError(f"Could not extract base URL from: {url}")
    return match.group(1)


def list_files(base_url, token, path="/"):
    """List all files in a Seafile shared directory recursively."""
    api_url = f"{base_url}/api/v2.1/share-links/{token}/dirents/"
    resp = requests.get(api_url, params={"path": path}, timeout=30)

    if resp.status_code != 200:
        print(f"  Warning: API returned {resp.status_code} for path={path}")
        print(f"  Response: {resp.text[:500]}")
        return []

    data = resp.json()
    files = []

    for item in data.get("dirent_list", []):
        if "file_name" in item:
            files.append({
                "name": item["file_name"],
                "path": f"{path.rstrip('/')}/{item['file_name']}",
                "size": item.get("file_size", 0),
            })
        elif "folder_name" in item:
            subpath = f"{path.rstrip('/')}/{item['folder_name']}"
            files.extend(list_files(base_url, token, subpath))

    return files


def download_file(base_url, token, file_info, output_dir):
    """Download a single file from Seafile."""
    rel_path = file_info["path"].lstrip("/")
    out_path = Path(output_dir) / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and out_path.stat().st_size == file_info["size"]:
        return f"SKIP {rel_path}"

    encoded_path = quote(file_info["path"])
    url = f"{base_url}/d/{token}/files/?p={encoded_path}&dl=1"

    try:
        resp = requests.get(url, stream=True, timeout=600)
        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0))
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192 * 16):
                f.write(chunk)

        return f"OK   {rel_path} ({file_info['size'] / 1e6:.1f} MB)"
    except Exception as e:
        return f"FAIL {rel_path}: {e}"


def main():
    parser = argparse.ArgumentParser(description="Download files from Seafile shared links")
    parser.add_argument("--url", required=True, help="Seafile shared link URL")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--workers", type=int, default=4, help="Parallel download workers")
    parser.add_argument("--dry-run", action="store_true", help="List files without downloading")
    args = parser.parse_args()

    base_url = get_base_url(args.url)
    token = extract_share_token(args.url)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    print(f"Seafile base: {base_url}")
    print(f"Share token:  {token}")
    print(f"Output dir:   {output}")
    print()

    print("Listing files...")
    files = list_files(base_url, token)
    total_size = sum(f["size"] for f in files)
    print(f"Found {len(files)} files, total size: {total_size / 1e9:.1f} GB")

    if args.dry_run:
        for f in files:
            print(f"  {f['path']}  ({f['size'] / 1e6:.1f} MB)")
        return

    print(f"\nDownloading with {args.workers} workers...")
    completed = 0
    skipped = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_file, base_url, token, f, output): f
            for f in files
        }
        pbar = tqdm(total=len(files), unit="file")
        for future in as_completed(futures):
            result = future.result()
            if result.startswith("OK"):
                completed += 1
            elif result.startswith("SKIP"):
                skipped += 1
            else:
                failed += 1
                tqdm.write(f"  {result}")
            pbar.update(1)
        pbar.close()

    print(f"\nDone: {completed} downloaded, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
