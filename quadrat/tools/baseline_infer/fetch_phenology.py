"""Fetch per-species month-of-observation histograms from GBIF.

For each species we hit:
    /v1/occurrence/search?taxonKey={gbif_id}&facet=month&facetLimit=12&limit=0

This returns a 12-bin month histogram (worldwide) in a single call. Output is
appended to a JSONL file as we go, so the run is resumable.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

GBIF_URL = "https://api.gbif.org/v1/occurrence/search"


def fetch_one(session: requests.Session, gbif_id: str, retries: int = 4) -> dict:
    params = {"taxonKey": gbif_id, "facet": "month", "facetLimit": 12, "limit": 0}
    backoff = 1.0
    for attempt in range(retries):
        try:
            r = session.get(GBIF_URL, params=params, timeout=30,
                            headers={"User-Agent": "plantclef-phenology/1.0"})
            if r.status_code in (429, 502, 503, 504):
                time.sleep(backoff); backoff *= 2; continue
            r.raise_for_status()
            data = r.json()
            counts = {int(b["name"]): int(b["count"])
                      for f in data.get("facets", []) for b in f.get("counts", [])}
            return {"gbif_id": gbif_id, "total": int(data.get("count", 0)), "months": counts}
        except requests.RequestException as e:
            if attempt == retries - 1:
                return {"gbif_id": gbif_id, "error": str(e)}
            time.sleep(backoff); backoff *= 2
    return {"gbif_id": gbif_id, "error": "exhausted retries"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", default="/workspace/scratch_space_arjun/PlantCLEF2026/src_experiments/i001_data_download/data/training_usage/metadata_filled_genus_family.csv")
    p.add_argument("--out", default="/workspace/PlantCLEF2026/data/phenology/gbif_month_histograms.jsonl")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--limit", type=int, default=None, help="Cap species for testing")
    args = p.parse_args()

    df = pd.read_csv(args.metadata, usecols=["species_id", "gbif_species_id"], dtype=str)
    df = df.drop_duplicates("species_id")
    df["gbif_species_id"] = df["gbif_species_id"].fillna("").str.split(".").str[0]
    species = [(r["species_id"], r["gbif_species_id"]) for _, r in df.iterrows()
               if r["gbif_species_id"]]
    if args.limit:
        species = species[:args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids = set()
    if out_path.exists():
        for line in out_path.open():
            try:
                rec = json.loads(line)
                if "error" not in rec:
                    done_ids.add(rec["gbif_id"])
            except json.JSONDecodeError:
                pass
    print(f"Already have {len(done_ids)} successful records; total target {len(species)}")

    todo = [(sid, gid) for sid, gid in species if gid not in done_ids]
    print(f"Fetching {len(todo)} new species with {args.workers} workers...")

    session = requests.Session()
    t0 = time.time()
    n_ok = n_err = 0
    with out_path.open("a") as fout, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_one, session, gid): (sid, gid) for sid, gid in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            sid, gid = futs[fut]
            try:
                rec = fut.result()
            except Exception as e:
                rec = {"gbif_id": gid, "error": f"thread:{e}"}
            rec["species_id"] = sid
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            if "error" in rec:
                n_err += 1
            else:
                n_ok += 1
            if i % 100 == 0 or i == len(todo):
                rate = i / (time.time() - t0)
                eta = (len(todo) - i) / rate if rate > 0 else 0
                print(f"  [{i:>5}/{len(todo)}] ok={n_ok} err={n_err}  {rate:.1f} req/s  ETA {eta/60:.1f} min")

    print(f"Done in {time.time()-t0:.0f}s. ok={n_ok} err={n_err}. Output -> {out_path}")


if __name__ == "__main__":
    main()
