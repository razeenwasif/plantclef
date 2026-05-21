# 007 — PlantNet-2024 DINOv2 on pre-made collages

## Why

The PlantCLEF 2024 organisers released a DINOv2 ViT-B/14 fine-tuned end-to-end
on single-plant data over 7806 species (76.16% top-1 on PC24 single-plant test).
7804 of those species are identical to PlantCLEF 2026's 7806, so the pretrained
classifier head is directly usable — we just remap rows into our sorted-species
ordering and zero-init the 2 dead classes.

On top of that starting point we fine-tune on the **pre-made labelled collages**
(`/workspace/plantclef/processed/collages/`, 50 000 images, per-image `species_ids`
in `synthetic_collages.csv`). These are the exact multi-species training
distribution the test set looks like — no on-the-fly mosaic synthesis needed.

## Files

- `model.py` — `PlantNetDINOv2MultiLabel`, head remap logic, LoRA wrap.
- `collage_dataset.py` — reads `synthetic_collages.csv`, returns `(image, k-hot)`.
- `train.py` — AdamW + OneCycleLR, ASL (γ_neg=4, γ_pos=1, clip=0.05), optional LoRA.
- `dump_test_probs.py` — tiled inference → `.npz` in the same schema as 005 so the
  existing `005/ensemble_with_other.py` and `005/apply_thresholds_to_npz.py` work
  unchanged.

## Run

```bash
cd src_experiments/007_plantnet_dinov2_collages
python train.py \
  --collage-csv   /workspace/plantclef/processed/synthetic_collages.csv \
  --collages-root /workspace/plantclef/processed/collages \
  --species-csv   /workspace/working/PlantCLEF2026/src_experiments/002_bioclip_tile_zero_shot_v2/data/species_lookup_with_gbif_cleaned_names.csv \
  --pc24-ckpt     /workspace/plantclef/raw/models/pretrained_models/vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all/model_best.pth.tar \
  --pc24-classes  /workspace/plantclef/raw/models/pretrained_models/class_mapping.txt \
  --img-size 336 --batch-size 32 --accum 2 --bf16 \
  --epochs 8 --output-dir /workspace/working/PlantCLEF2026/models/plantnet_collage_v1
```

Inference:

```bash
python dump_test_probs.py \
  --checkpoint /workspace/working/PlantCLEF2026/models/plantnet_collage_v1/phase_collage_best.pth \
  --images-root /workspace/plantclef/raw/test/PlantCLEF2025TestImages \
  --output /workspace/working/PlantCLEF2026/models/plantnet_collage_v1/test_probs.npz \
  --tile-size 336 --tile-overlap 112 --batch-size 16 --bf16
```

Then convert to submission via 005's threshold script or ensemble with Arjun's
006 probs via `005/ensemble_with_other.py`.
