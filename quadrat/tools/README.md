# Tools

This directory contains standalone utility scripts for data engineering, environmental modeling, and system infrastructure.

Typical tools include:
- `build_taxonomic_graph.py` / `build_ecological_db.py`: Scripts to parse GBIF/EOL data into the adjacency matrices used by the Neuro-Symbolic GCN.
- `audit_species_health.py`: Diagnostic scripts to evaluate long-tail class imbalance and identify candidates for FAISS retrieval boosting.
- `vram_watchdog.py`: Infrastructure monitors that autonomously tune batch sizes to prevent OOM errors during high-resolution training.
- `optimize_thresholds.py`: Post-training scripts to calibrate the optimal confidence thresholds for maximizing the Macro-F1 score.