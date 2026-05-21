# 014 — iNaturalist Research-grade fine-tune data pull

Pull habitat / whole-plant images from the iNaturalist Research-grade dataset on
GBIF (`50c9509d-22c7-4a22-a47d-8c48425ef4a7`, DOI `10.15468/ab3s5x`) for the
PlantCLEF 2026 species set. The hypothesis: PC24 single-plant Pl@ntNet
close-ups are visually too narrow to bridge the close-up→quadrat domain gap;
iNat habitat shots match the test-set visual statistics better.

## Scripts

- `coverage_audit.py` — quick pre-flight: for a sample of species, query the
  GBIF occurrence count under the iNat dataset key. Prints distribution
  (median / p10 / p90 / % with ≥100 obs / % with ≥500 obs). Cheap.

- `download_inat.py` — async downloader. For each species in the lookup
  CSV, queries GBIF for occurrences in dataset `50c9509d…` with
  `mediaType=StillImage`, downloads up to `--cap-per-species` images (default
  500) to `<out-dir>/<species_id>/{occurrenceID}_{photo_idx}.jpg`. Writes a
  manifest CSV (`image_path, species_id, gbif_species_id, gbif_occurrence_id,
  license, original_url`). Resumable — skips species that already have ≥cap
  images on disk.

## Filter

- `datasetKey = 50c9509d-22c7-4a22-a47d-8c48425ef4a7` (iNat Research-grade)
- `mediaType = StillImage`
- `taxonKey = <gbif_species_id>` (from `species_lookup_with_gbif_cleaned_names.csv`)
- license: dataset publishes mix CC0 / CC-BY / CC-BY-NC; manifest records the
  per-occurrence license so we can filter post-hoc.

## Storage budget

- 7,806 species × 500 images × ~150 KB (medium-derivative iNat photo) ≈ **585 GB**
- Pod `/workspace` has hundreds of TB free; not a constraint.
- Output: `/workspace/plantclef/raw/inat_research_grade/{species_id}/*.jpg`
- Manifest: `/workspace/plantclef/processed/inat_research_grade_manifest.csv`

## Usage

```bash
# Audit (5 min on 100 species)
python src_experiments/014_inat_research_grade/coverage_audit.py \
    --species-csv /workspace/working/PlantCLEF2026/src_experiments/002_bioclip_tile_zero_shot_v2/data/species_lookup_with_gbif_cleaned_names.csv \
    --sample 100 \
    --out audit_sample100.csv

# Smoke test (5 species)
python src_experiments/014_inat_research_grade/download_inat.py \
    --species-csv /workspace/working/PlantCLEF2026/src_experiments/002_bioclip_tile_zero_shot_v2/data/species_lookup_with_gbif_cleaned_names.csv \
    --out-dir /workspace/plantclef/raw/inat_research_grade \
    --manifest /workspace/plantclef/processed/inat_research_grade_manifest.csv \
    --cap-per-species 20 \
    --max-species 5 \
    --workers 16

# Full pull
python src_experiments/014_inat_research_grade/download_inat.py \
    --species-csv .../species_lookup_with_gbif_cleaned_names.csv \
    --out-dir /workspace/plantclef/raw/inat_research_grade \
    --manifest /workspace/plantclef/processed/inat_research_grade_manifest.csv \
    --cap-per-species 500 \
    --workers 32
```

## Followup experiments (not in this dir)

- **015** = `010 BioCLIP25MultiTask` + `--use-taxonomy-heads` + PC24⊕iNat 50:50
  sampler. Anchor: must beat **0.38333** (010 last_blocks ALONE on PC24).
- **014b** = control: 010 + `--use-taxonomy-heads` on **PC24 alone** (no iNat).
  Run in parallel with the pull to isolate tax-head lift from data lift.
