"""
Coverage audit: how many iNat Research-grade observations exist per
PlantCLEF 2026 species in the GBIF iNat dataset?

Hits the GBIF occurrence search API with `count=true` (no records returned,
just the total) for each species's `gbif_species_id`. Cheap (~1 req/species,
no media downloaded). Use a small `--sample` to size up the full pull.

Output: one CSV row per audited species with the record count under
`mediaType=StillImage`. Stdout: distribution summary (median, p10, p90,
% with >=N obs).
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import sys
import time
from pathlib import Path

import aiohttp
import pandas as pd

INAT_DATASET_KEY = "50c9509d-22c7-4a22-a47d-8c48425ef4a7"
GBIF_API = "https://api.gbif.org/v1/occurrence/search"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--species-csv", required=True,
                   help="species_lookup_with_gbif_cleaned_names.csv")
    p.add_argument("--out", required=True, help="Output CSV path.")
    p.add_argument("--sample", type=int, default=0,
                   help="Random-sample N species (0 = all 7806).")
    p.add_argument("--workers", type=int, default=16,
                   help="Concurrent API requests (be polite to GBIF).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


async def count_for_species(
    sess: aiohttp.ClientSession, gbif_id: int, sem: asyncio.Semaphore,
) -> int:
    params = {
        "datasetKey": INAT_DATASET_KEY,
        "taxonKey": str(gbif_id),
        "mediaType": "StillImage",
        "limit": 0,
    }
    async with sem:
        for attempt in range(3):
            try:
                async with sess.get(GBIF_API, params=params, timeout=30) as r:
                    if r.status != 200:
                        await asyncio.sleep(1.0 * (attempt + 1))
                        continue
                    j = await r.json()
                    return int(j.get("count", 0))
            except Exception as exc:
                logger.warning(f"  taxon={gbif_id} attempt={attempt} {exc}")
                await asyncio.sleep(1.0 * (attempt + 1))
    return -1


async def run(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.species_csv)
    df = df[df["gbif_species_id"].notna()].copy()
    df["gbif_species_id"] = df["gbif_species_id"].astype("int64")
    if args.sample > 0:
        df = df.sample(n=min(args.sample, len(df)), random_state=args.seed)
    df = df.reset_index(drop=True)
    logger.info(f"Auditing {len(df)} species against iNat dataset {INAT_DATASET_KEY}")

    sem = asyncio.Semaphore(args.workers)
    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout) as sess:
        t0 = time.time()
        tasks = [count_for_species(sess, int(g), sem) for g in df["gbif_species_id"]]
        counts: list[int] = []
        for i, fut in enumerate(asyncio.as_completed(tasks), 1):
            counts.append(await fut)
            if i % 50 == 0 or i == len(tasks):
                rate = i / max(time.time() - t0, 1e-6)
                eta = (len(tasks) - i) / max(rate, 1e-6)
                logger.info(f"  {i}/{len(tasks)} ({rate:.1f} req/s, ETA {eta:.0f}s)")

    df["inat_obs_count"] = counts
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df[["species_id", "gbif_species_id", "species", "inat_obs_count"]].to_csv(
        out_path, index=False, quoting=csv.QUOTE_MINIMAL,
    )
    logger.info(f"Wrote {out_path}")

    # Distribution summary
    s = df.loc[df["inat_obs_count"] >= 0, "inat_obs_count"].astype(int)
    if len(s) == 0:
        logger.error("No successful counts. API failure?")
        sys.exit(1)
    print()
    print(f"=== Coverage summary (n={len(s)}, errors={(df['inat_obs_count'] < 0).sum()}) ===")
    print(f"  mean    = {s.mean():>10.1f}")
    print(f"  median  = {s.median():>10.0f}")
    print(f"  p10     = {s.quantile(0.10):>10.0f}")
    print(f"  p25     = {s.quantile(0.25):>10.0f}")
    print(f"  p75     = {s.quantile(0.75):>10.0f}")
    print(f"  p90     = {s.quantile(0.90):>10.0f}")
    print(f"  max     = {s.max():>10.0f}")
    print()
    for thresh in (10, 50, 100, 250, 500, 1000):
        pct = (s >= thresh).mean() * 100
        print(f"  >= {thresh:>4} obs : {pct:5.1f}% of audited species")
    print()


def main() -> None:
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
