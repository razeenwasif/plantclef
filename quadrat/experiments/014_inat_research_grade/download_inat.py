"""
Async iNaturalist Research-grade image puller for the PlantCLEF 2026 species
set, via the GBIF occurrence search API filtered to dataset
50c9509d-22c7-4a22-a47d-8c48425ef4a7.

Per species:
  1. Page through GBIF occurrences (limit=300/page, mediaType=StillImage).
  2. For each occurrence, take media[0].identifier as the photo URL.
  3. Rewrite iNat S3 'original' URLs to 'medium' (≤500px) to keep storage sane.
  4. Download up to --cap-per-species images to <out-dir>/<species_id>/.
  5. Append a manifest row per saved image.

Resumable: at startup, counts files already on disk per species and skips
species at cap. Manifest is append-only (mode='a').

Run:
    python download_inat.py \
        --species-csv .../species_lookup_with_gbif_cleaned_names.csv \
        --out-dir /workspace/plantclef/raw/inat_research_grade \
        --manifest /workspace/plantclef/processed/inat_research_grade_manifest.csv \
        --cap-per-species 500 --workers 32
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import aiohttp
import pandas as pd

INAT_DATASET_KEY = "50c9509d-22c7-4a22-a47d-8c48425ef4a7"
GBIF_API = "https://api.gbif.org/v1/occurrence/search"
PAGE_SIZE = 300

# iNat S3 URL pattern: rewrite /original.jpg → /medium.jpg
INAT_RE = re.compile(
    r"^(https?://[^/]*inaturalist[^/]*\.s3\.amazonaws\.com/photos/\d+/)"
    r"(original|large)(\.[a-zA-Z]+)$"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--species-csv", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--cap-per-species", type=int, default=500)
    p.add_argument("--max-species", type=int, default=0,
                   help="0 = all species; otherwise truncate (smoke test).")
    p.add_argument("--workers", type=int, default=32,
                   help="Concurrent download workers per pod.")
    p.add_argument("--api-workers", type=int, default=8,
                   help="Concurrent GBIF search requests.")
    p.add_argument("--connect-timeout", type=float, default=15.0)
    p.add_argument("--read-timeout", type=float, default=60.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shuffle", action="store_true",
                   help="Shuffle species order — useful when running 2+ pods.")
    return p.parse_args()


def rewrite_inat_to_medium(url: str) -> str:
    m = INAT_RE.match(url)
    if m:
        return f"{m.group(1)}medium{m.group(3)}"
    return url


async def gbif_search_page(
    sess: aiohttp.ClientSession, taxon_key: int, offset: int,
    api_sem: asyncio.Semaphore,
) -> dict[str, Any] | None:
    params = {
        "datasetKey": INAT_DATASET_KEY,
        "taxonKey": str(taxon_key),
        "mediaType": "StillImage",
        "limit": PAGE_SIZE,
        "offset": offset,
    }
    async with api_sem:
        for attempt in range(3):
            try:
                async with sess.get(GBIF_API, params=params, timeout=60) as r:
                    if r.status == 200:
                        return await r.json()
                    await asyncio.sleep(1.0 * (attempt + 1))
            except Exception as exc:
                logger.warning(f"  taxon={taxon_key} ofs={offset} {exc}")
                await asyncio.sleep(1.0 * (attempt + 1))
    return None


async def collect_urls_for_species(
    sess: aiohttp.ClientSession, taxon_key: int, cap: int,
    api_sem: asyncio.Semaphore,
) -> list[dict[str, str]]:
    """Return [{url, occ_id, license, scientific_name}, ...] up to ~cap."""
    out: list[dict[str, str]] = []
    offset = 0
    while len(out) < cap:
        page = await gbif_search_page(sess, taxon_key, offset, api_sem)
        if not page or not page.get("results"):
            break
        for occ in page["results"]:
            media = occ.get("media") or []
            if not media:
                continue
            url = media[0].get("identifier")
            if not url:
                continue
            out.append({
                "url": rewrite_inat_to_medium(url),
                "occ_id": str(occ.get("key", "")),
                "license": str(occ.get("license", "")),
                "sci_name": str(occ.get("scientificName", "")),
            })
            if len(out) >= cap:
                break
        if page.get("endOfRecords"):
            break
        offset += PAGE_SIZE
    return out


async def download_one(
    sess: aiohttp.ClientSession, item: dict[str, str], dest: Path,
    dl_sem: asyncio.Semaphore,
) -> bool:
    if dest.exists() and dest.stat().st_size > 1024:
        return True
    async with dl_sem:
        for attempt in range(3):
            try:
                async with sess.get(item["url"], timeout=60) as r:
                    if r.status != 200:
                        await asyncio.sleep(0.5 * (attempt + 1))
                        continue
                    data = await r.read()
                    if len(data) < 1024:
                        return False
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    tmp = dest.with_suffix(dest.suffix + ".tmp")
                    tmp.write_bytes(data)
                    os.replace(tmp, dest)
                    return True
            except Exception as exc:
                logger.debug(f"  dl fail {item['url']} attempt={attempt} {exc}")
                await asyncio.sleep(0.5 * (attempt + 1))
    return False


async def process_species(
    sess: aiohttp.ClientSession,
    species_id: int, gbif_id: int, sci_name: str,
    out_dir: Path, cap: int, api_sem: asyncio.Semaphore,
    dl_sem: asyncio.Semaphore, manifest_writer: csv.DictWriter,
    manifest_lock: asyncio.Lock,
) -> tuple[int, int]:
    """Returns (saved_count, skipped_already_on_disk)."""
    sp_dir = out_dir / str(species_id)
    sp_dir.mkdir(parents=True, exist_ok=True)
    existing = sum(1 for p in sp_dir.iterdir() if p.is_file() and p.suffix.lower() == ".jpg")
    if existing >= cap:
        return 0, existing

    need = cap - existing
    items = await collect_urls_for_species(sess, gbif_id, need, api_sem)
    if not items:
        return 0, existing

    saved = 0
    tasks = []
    for item in items:
        ext = ".jpg"
        # Take everything after the last '/' before '.' as a stable photo id.
        url_path = item["url"].rsplit("?", 1)[0]
        fname_base = url_path.rsplit("/", 2)[-2] if url_path.count("/") >= 2 else item["occ_id"]
        dest = sp_dir / f"{item['occ_id']}_{fname_base}{ext}"
        tasks.append((item, dest))

    async def _do(item, dest):
        ok = await download_one(sess, item, dest, dl_sem)
        if not ok:
            return None
        return (item, dest)

    futures = [asyncio.create_task(_do(it, ds)) for it, ds in tasks]
    rows: list[dict[str, str]] = []
    for fut in asyncio.as_completed(futures):
        res = await fut
        if res is None:
            continue
        item, dest = res
        saved += 1
        rows.append({
            "image_path": str(dest),
            "species_id": str(species_id),
            "gbif_species_id": str(gbif_id),
            "gbif_occurrence_id": item["occ_id"],
            "license": item["license"],
            "scientific_name": item["sci_name"],
            "url": item["url"],
        })
        if existing + saved >= cap:
            for f in futures:
                f.cancel()
            break

    if rows:
        async with manifest_lock:
            manifest_writer.writerows(rows)
    return saved, existing


async def run(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.species_csv)
    df = df[df["gbif_species_id"].notna()].copy()
    df["gbif_species_id"] = df["gbif_species_id"].astype("int64")
    if args.shuffle:
        df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    if args.max_species > 0:
        df = df.head(args.max_species)
    logger.info(
        f"Pulling iNat dataset {INAT_DATASET_KEY} for {len(df)} species, "
        f"cap={args.cap_per_species}/sp, dl_workers={args.workers}, "
        f"api_workers={args.api_workers}"
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    new_manifest = not manifest_path.exists()
    mf = open(manifest_path, "a", newline="", encoding="utf-8")
    fields = [
        "image_path", "species_id", "gbif_species_id",
        "gbif_occurrence_id", "license", "scientific_name", "url",
    ]
    writer = csv.DictWriter(mf, fieldnames=fields)
    if new_manifest:
        writer.writeheader()
        mf.flush()
    manifest_lock = asyncio.Lock()

    api_sem = asyncio.Semaphore(args.api_workers)
    dl_sem = asyncio.Semaphore(args.workers)

    timeout = aiohttp.ClientTimeout(
        sock_connect=args.connect_timeout, sock_read=args.read_timeout,
    )
    connector = aiohttp.TCPConnector(limit=args.workers + args.api_workers + 8)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector,
                                     headers={"User-Agent": "PlantCLEF2026/0.1 (+training)"}) as sess:
        t0 = time.time()
        total_saved = 0
        total_skipped = 0
        # Process species sequentially — already-fanned-out via concurrent
        # downloads inside each species. Sequential outer loop keeps GBIF
        # search QPS sane and lets us emit clean per-species progress lines.
        for i, row in enumerate(df.itertuples(index=False), 1):
            sp_id = int(row.species_id)
            gb_id = int(row.gbif_species_id)
            sci = str(getattr(row, "species", ""))
            try:
                saved, existing = await process_species(
                    sess, sp_id, gb_id, sci, out_dir,
                    args.cap_per_species, api_sem, dl_sem,
                    writer, manifest_lock,
                )
                mf.flush()
            except Exception as exc:
                logger.error(f"species_id={sp_id} gbif={gb_id} ERROR {exc}")
                continue
            total_saved += saved
            total_skipped += existing
            if i % 5 == 0 or i == len(df):
                rate = i / max(time.time() - t0, 1e-6)
                eta = (len(df) - i) / max(rate, 1e-6)
                logger.info(
                    f"  [{i}/{len(df)}] sp={sp_id} +{saved} (existing={existing}) "
                    f"| saved_total={total_saved} skipped_total={total_skipped} "
                    f"| {rate:.2f} sp/s ETA {eta/60:.1f}m"
                )

    mf.close()
    logger.info(f"Done. saved={total_saved}, already_on_disk={total_skipped}")


def main() -> None:
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
