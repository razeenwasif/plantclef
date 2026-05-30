import argparse
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import requests


SPECIES_URL = "https://api.gbif.org/v1/species/{key}"
VERNACULARS_URL = "https://api.gbif.org/v1/species/{key}/vernacularNames"


def fetch_json(session: requests.Session, url: str, timeout: int = 30) -> Optional[Dict[str, Any]]:
    try:
        r = session.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "plantclef-gbif-enricher/1.0"},
        )
        if r.status_code == 404:
            print(f"404 for {url}")
            return None
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        print(f"Request failed for {url}: {e}")
        return None


def normalize_key(value: Any) -> Optional[int]:
    if pd.isna(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None

def fetch_synonyms(session, gbif_key, scientific_name):
    # search by scientific name, then keep only synonym rows
    url = "https://api.gbif.org/v1/species/search"
    params = {
        "q": scientific_name,
        "limit": 100,
        "rank": "SPECIES",
        "qField": "SCIENTIFIC",
    }
    r = session.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    syns = []
    for row in data.get("results", []):
        status = str(row.get("taxonomicStatus", "")).upper()
        accepted_key = row.get("acceptedKey")
        sci = row.get("scientificName")

        if status == "SYNONYM" and accepted_key == gbif_key and sci:
            syns.append(sci)

    return sorted(set(syns))

def fetch_gbif_record(session: requests.Session, gbif_key: Optional[int]) -> Dict[str, Any]:
    if gbif_key is None:
        return {}

    core = fetch_json(session, SPECIES_URL.format(key=gbif_key))
    if not core:
        return {}

    vernaculars = fetch_json(session, VERNACULARS_URL.format(key=gbif_key)) or {}
    vernacular_rows = vernaculars.get("results", [])

    common_names_all = sorted({
        row.get("vernacularName", "").strip()
        for row in vernacular_rows
        if row.get("vernacularName")
    })

    common_names_en = sorted({
        row.get("vernacularName", "").strip()
        for row in vernacular_rows
        if row.get("vernacularName") and str(row.get("language", "")).lower().startswith("eng")
    })

    scientific_name = core.get("scientificName") or core.get("canonicalName") or ""
    synonym_names = fetch_synonyms(session, gbif_key, scientific_name)

    return {
        "gbif_scientific_name": core.get("scientificName"),
        "gbif_canonical_name": core.get("canonicalName"),
        "gbif_rank": core.get("rank"),
        "gbif_taxonomic_status": core.get("taxonomicStatus"),
        "gbif_accepted_key": core.get("acceptedKey"),
        "gbif_kingdom": core.get("kingdom"),
        "gbif_phylum": core.get("phylum"),
        "gbif_class": core.get("class"),
        "gbif_order": core.get("order"),
        "gbif_family": core.get("family"),
        "gbif_genus": core.get("genus"),
        "gbif_species": core.get("species"),
        "gbif_num_synonyms": len(synonym_names),
        "gbif_synonyms": " | ".join(synonym_names),
        "gbif_common_names_en": " | ".join(common_names_en),
        "gbif_common_names_all": " | ".join(common_names_all),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a species-level lookup from PlantCLEF metadata and enrich it with GBIF.")
    parser.add_argument("--input", required=True, help="Path to PlantCLEF metadata CSV")
    parser.add_argument("--output", required=True, help="Path to output species lookup CSV")
    parser.add_argument("--sleep", type=float, default=0.05, help="Sleep between GBIF requests")
    args = parser.parse_args()

    df = pd.read_csv(args.input, sep=";")

    wanted_csv_cols = [
        "species_id",
        "gbif_species_id",
        "species",
        "genus",
        "family",
        "dataset",
        "publisher",
        "license",
    ]
    available_cols = [c for c in wanted_csv_cols if c in df.columns]
    if "species_id" not in available_cols:
        raise ValueError("Input CSV must contain a species_id column")

    base = df[available_cols].drop_duplicates(subset=["species_id"]).copy()

    if "organ" in df.columns:
        organs = (
            df.groupby("species_id")["organ"]
            .agg(lambda s: " | ".join(sorted({str(x) for x in s.dropna()})))
            .rename("organs_seen")
            .reset_index()
        )
        base = base.merge(organs, on="species_id", how="left")

    image_count = df.groupby("species_id").size().rename("image_count").reset_index()
    base = base.merge(image_count, on="species_id", how="left")

    if "obs_id" in df.columns:
        obs_count = (
            df.groupby("species_id")["obs_id"]
            .nunique()
            .rename("n_unique_obs")
            .reset_index()
        )
        base = base.merge(obs_count, on="species_id", how="left")

    if "gbif_species_id" in base.columns:
        base["gbif_species_id"] = base["gbif_species_id"].apply(normalize_key)

        session = requests.Session()
        unique_keys = base["gbif_species_id"].dropna().drop_duplicates().tolist()
        cache: Dict[int, Dict[str, Any]] = {}

        for key in unique_keys:
            cache[key] = fetch_gbif_record(session, int(key))
            time.sleep(args.sleep)

        gbif_df = pd.DataFrame([cache.get(key, {}) if pd.notna(key) else {} for key in base["gbif_species_id"]])
        out = pd.concat([base.reset_index(drop=True), gbif_df.reset_index(drop=True)], axis=1)
    else:
        out = base

    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out):,} rows to {args.output}")


if __name__ == "__main__":
    main()
