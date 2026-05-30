# plantclef project Progress

This file tracks the evolution of the plantclef system from a plant-specific identifier to a Universal Neuro-Symbolic Knowledge Engine.

## 🏁 Milestones Completed

### 1. Global Rebranding (May 7, 2026)
- [x] Renamed all "Nitro" instances to "plantclef" across the entire codebase.
- [x] Renamed all directories and files.
- [x] Updated all internal strings and documentation.
- [x] Verified system integrity with a full test suite pass (28/28 tests).

### 2. Universal plantclef infrastructure
- [x] Drafted the `docs/UNIVERSAL_NITRO_ROADMAP.md`.
- [x] Initialized PostgreSQL database (`oracle-db`) with `ltree` and `pgvector` extensions.
- [x] Created the `taxonomy_nodes` table schema for dynamic ontology management.

## 🚧 Current Phase: Phase 1 (The Relational Backbone)
- [ ] Task 1.1: Install Docker (Done)
- [ ] Task 1.2: Spin up PostgreSQL with `pgvector` (Done)
- [ ] Task 1.3: Initialize `TaxonomyNodes` schema (Done)
- [ ] **Task 1.4: Migrate existing taxonomic data to PostgreSQL (Next)**

## 📍 Next Steps
- Write a Python migration script to populate the `taxonomy_nodes` table using the existing species/genus mapping data.
- Explore recursive SQL queries to replace the static GCN adjacency matrix logic.

## 📝 Recent Context
- Successfully integrated Gemma 4 for Agentic Dataset Auto-Discovery.
- Configured dynamic sharding and stratified splitting for PlantNet-300K.
- Hardened environment for WSL2 + RTX 4090 + External D: drive.
