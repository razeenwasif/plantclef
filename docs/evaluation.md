# Evaluation Metric

## Overview

The goal is to predict the presence of one or more plant species in each high-resolution quadrat image, selecting from thousands of possible species. Quadrat images typically cover 50×50 cm, meaning that while multiple species may be present, it is uncommon to have dozens of species simultaneously.

## Metric: Macro-Averaged F1 Score per Sample

We use an **F1 score-based metric**, designed to balance recall and precision. This ensures that models do not over-predict species (leading to low precision) nor under-predict species (resulting in low recall).

For this challenge, we use the **macro-averaged F1 score per sample**, which is particularly suited for multi-label classification problems. This approach ensures a fair evaluation of each image independently, rather than being influenced by class distribution.

## Transect-Level Averaging

The competition test set consists of several quadrat images, each representing a sample of a specific area within a selected site (e.g., a 5m × 1m area in the Pyrenees). This surveying process, commonly known as a **transect**, is designed to assess biodiversity across a broad section of the site.

To mitigate bias from over-sampled areas, the macro-averaged F1 scores are first computed across all quadrats within each transect, then averaged across transects to return the final score.

## Formula

The final score is computed as:

```
Final Score = (1 / T) * Σ_t F1_macro(t)
```

Where:

- `T` is the total number of transects
- `F1_macro(t)` is the macro-averaged F1-score per sample of transect `t`

```
F1_macro(t) = (1 / N_t) * Σ_i F1_i
```

Where `N_t` is the number of quadrats in transect `t`.

For each test image `i`:

```
F1_i = 2 * (Precision_i * Recall_i) / (Precision_i + Recall_i)
```

```
Precision_i = TP_i / (TP_i + FP_i)
Recall_i    = TP_i / (TP_i + FN_i)
```

Where:

- **TP_i** (True Positives): number of plant species correctly predicted
- **FP_i** (False Positives): number of plant species incorrectly predicted
- **FN_i** (False Negatives): number of plant species missed

## Reference

For more details on the F1 score, see: [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
