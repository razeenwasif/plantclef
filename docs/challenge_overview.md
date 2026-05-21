# PlantCLEF 2026 Challenge Overview

## Goal

Identify all plant species present in a high-resolution image of a vegetation plot defined by a square frame placed on the ground to mark out a specific area for sampling. The challenge focuses on addressing the complexities of fine-grained classification and the impact of environmental variability.

## Context

Vegetation plot images provide a standardized way to assess biodiversity, support long-term monitoring, and enable large-scale ecological surveys. These images usually cover 50 × 50 cm quadrats, where botanists identify all visible species and quantify their abundance using biomass, cover, or related indicators.

Integrating AI can improve the efficiency of specialists and help scale up ecological studies. However, building AI models for multi-label vegetation identification remains technically challenging.

In theory, robust models would be trained on large collections of high-resolution plot images fully annotated with all species present. In practice, producing such datasets at scale is extremely difficult because a single flora can contain thousands of species. By contrast, very large datasets of single plant images already exist, which makes them easier to use for training highly efficient classification models.

## The Challenge

The PlantCLEF 2026 challenge focuses on **predicting the species present in vegetation plot images**, while relying on training data composed of **single-label images of individual plants**.

The task is framed as a **multi-label classification problem**, where the objective is to predict all plant species visible in high-resolution quadrat images. The main technical bottleneck is the **domain shift** between the training data (single plant images) and the test data (complex, high-resolution, multi-species vegetation scenes).

Building on the strong participation and results of the 2025 edition, the 2026 challenge follows a similar format, using the same datasets and the same Kaggle platform.

## Timeline

| Date | Event |
|------|-------|
| November 2025 | Registration opens (free of charge) |
| 1 February 2026 | Competition Start |
| 23 April 2026 | Registration closes |
| 7 May 2026 | Competition Deadline |
| 28 May 2026 | Working note papers deadline (CEUR-WS) |
| 30 June 2026 | Notification of acceptance |
| 6 July 2026 | Camera-ready deadline |
| 21-24 Sept 2026 | CLEF 2026, Jena, Germany |

All deadlines are at 11:59 PM CET unless otherwise stated.

## Venues

- **FGVC13 at CVPR 2026** — Denver, Colorado, USA, June 3-7, 2026
- **LifeCLEF 2026 at CLEF 2026** — Jena, Germany, September 21-24, 2026

## Organizers

- Giulio Martellucci, Inrae, LISAH, Montpellier
- Ilyass Moummad, Inria, LIRMM, Montpellier
- Hervé Goëau, Cirad, UMR AMAP, Montpellier
- Pierre Bonnet, Cirad, UMR AMAP, Montpellier
- Fabrice Vinatier, Inrae, LISAH, Montpellier
- Alexis Joly, Inria, LIRMM, Montpellier
