# Dataset Description

> **Note on Data Versions:** To allow for historical comparison, this 2026 edition uses the same datasets as the previous rounds. Consequently, files and folders retain their original "2024" or "2025" names.

> **Note:** Only the test set and metadata are hosted directly on Kaggle (~8.3 GB). The training set images (>160 GB) must be downloaded via external links.

---

## 1. Training Set: Individual Plant Images

A subset of the Pl@ntNet training data focusing on south-western Europe, covering **7,806 plant species**.

| Property | Value |
|----------|-------|
| Images | ~1.4 million |
| Source | Pl@ntNet + trusted labels from GBIF |
| Resolution | Min or max of 800px on the biggest side |
| Metadata file | `PlantCLEF2024singleplanttrainingdata.csv` |
| Species list | `species_ids.csv` |

### Download Links (Direct Tar Files)

- **Full resolution** (~281 GB): [PlantCLEF2024singleplanttrainingdata.tar](https://lab.plantnet.org/LifeCLEF/PlantCLEF2024/single_plant_training_data/PlantCLEF2024singleplanttrainingdata.tar)
- **Max side size — 800px** (~160 GB): [PlantCLEF2024singleplanttrainingdata_800_max_side_size.tar](https://lab.plantnet.org/LifeCLEF/PlantCLEF2024/single_plant_training_data/PlantCLEF2024singleplanttrainingdata_800_max_side_size.tar)

### Organization

Images are pre-organized into:
- Subfolders by class (species_id)
- Split into `train` / `validation` / `test` sets

Participants can gather all images into a single `train` directory or create their own splits.

---

## 2. Complementary Training Set: Pseudo-Quadrat Images (Unlabeled)

A large, **unannotated** dataset of pseudo-quadrat images — ground vegetation images framed similarly to vertical quadrats, but not necessarily covering a precise 50 cm × 50 cm area.

| Property | Value |
|----------|-------|
| Images | 212,782 |
| Source | LUCAS Cover Photos 2006–2018 |
| Labels | None |
| Metadata | `pseudoquadrats_without_labels_complementary_training_set_urls.csv` |

### Purpose

- Self-supervised pre-training
- Reduce domain shift between single-plant training images and multi-species test quadrats
- Help models become more familiar with the pseudo-quadrat domain

### Download (Direct Tar File)

- [PlantCLEF2025_pseudoquadrats_without_labels_complementary_training_set.tar](https://lab.plantnet.org/LifeCLEF/PlantCLEF2025/pseudoquadrats_without_labels_complementary_training_set/PlantCLEF2025_pseudoquadrats_without_labels_complementary_training_set.tar) (~170 GB)

---

## 3. Test Set: Vegetation Quadrat Images

A compilation of several image datasets of plots in different floristic contexts, including Pyrenean and Mediterranean floras.

| Property | Value |
|----------|-------|
| Total images | 2,105 high-resolution images |
| Produced by | Expert botanists |
| Metadata | `PlantCLEF2025_test.csv` |
| Images directory | `PlantCLEF2025test/` |

### Variability

- Use of wooden frames or measuring tape to delimit the plot (or not)
- Angles of view more or less perpendicular to the ground
- Image quality varies depending on weather (shadows, blurry areas, etc.)

### Alternative Download

- [Test dataset (alternative link)](https://lab.plantnet.org/seafile/d/6a91e3de6b5b49e5a70c/)

---

## 4. Pre-Trained Models

Two state-of-the-art models trained using the `timm` library, hosted on Hugging Face. Both are based on a **ViT-base architecture** pre-trained with **DINOv2** (self-supervised learning) on 142M images.

### Available Models

| Model | Description |
|-------|-------------|
| `dinov2_patch14_reg4_onlyclassifier_then_all` | Classifier head trained first, then full model fine-tuned |
| `dinov2_patch14_reg4_dinov2_onlyclassifier` | Only classifier head trained on top of frozen DINOv2 |

### Download

Available on [Zenodo](https://doi.org/10.5281/zenodo.10848263) or from the Models tab of the competition.

### Citation

```bibtex
@misc{goeau_2024_10848263,
  author    = {Goëau, Hervé and Lombardo, Jean-Chirstophe and Affouard, Antoine
               and Espitalier, Vincent and Bonnet, Pierre and Joly, Alexis},
  title     = {{PlantCLEF 2024 pretrained models on the flora of the south western
               Europe based on a subset of Pl@ntNet collaborative images and a
               ViT base patch 14 dinoV2}},
  month     = mar,
  year      = 2024,
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.10848263},
  url       = {https://doi.org/10.5281/zenodo.10848263}
}
```
