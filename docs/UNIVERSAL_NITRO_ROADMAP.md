# The Universal ORACLE Roadmap
**From Static Scripts to a Dynamic Neuro-Symbolic Knowledge Engine**

This roadmap is your step-by-step guide to rebuilding the Oracle system into a Universal Identifier. We will take this one phase at a time. 

---

## 🛠️ Phase 1: The Relational Backbone (Taxonomy & Ontology)
**Goal:** Replace static `json` files with a relational PostgreSQL database. Learn how to model hierarchies.

### Concepts to Learn:
1. **Relational Modeling:** Primary Keys, Foreign Keys, and Normalization.
2. **PostgreSQL `ltree`:** A specific extension in Postgres designed for storing and querying tree-like structures (perfect for taxonomy like Domain -> Kingdom -> Phylum -> ... -> Species).
3. **Python ORMs:** Using `SQLAlchemy` or `psycopg2` to talk to your database from Python.

### Your Action Items:
* [X] **Task 1.1:** Install **Docker** on your WSL/Windows machine. We will run PostgreSQL inside a Docker container (this is industry standard and keeps your system clean).
* [X] **Task 1.2:** Spin up a PostgreSQL container with the `pgvector` image (which includes `ltree`).
```sh 
docker run --name oracle-db -e POSTGRES_PASSWORD=oracle -p 5432:5432 -d pgvector/pgvector:pg16
```
* [ ] **Task 1.3:** Write a Python SQLAlchemy script to create your first table: `TaxonomyNodes`. This will migrate existing taxonomic data from JSON/CSV mapping files into a new relational structure.
* [ ] **Task 1.4:** Write a Python script that reads your old `species_to_genus.json` and `INSERT`s that data into your new PostgreSQL database.
* [ ] **Task 1.5:** Write a SQL query using a **Recursive CTE** (Common Table Expression) to fetch the full lineage of a single species.

---

## 🧠 Phase 2: The Feature Store (Vector Databases)
**Goal:** Replace `.pt` and `.npy` cache files with a searchable Vector Database.

### Concepts to Learn:
1. **Vector Embeddings:** Storing float arrays (like DINOv3 features) in a database.
2. **`pgvector`:** The Postgres extension that allows you to do cosine similarity searches directly in SQL.

### Your Action Items:
* [ ] **Task 2.1:** Add a new column to your database: `embedding vector(1024)` (or whatever dimension BioCLIP uses).
* [ ] **Task 2.2:** Write a Python script that takes an image, runs it through frozen BioCLIP, and `INSERT`s the resulting tensor into the database alongside its metadata (location, time, etc.).
* [ ] **Task 2.3:** Write a SQL query to do a "Similarity Search": *Find the 5 most similar images in the database to this new tensor.* 

---

## 🏭 Phase 3: The Universal Ingestion Engine (ETL)
**Goal:** Build a robust pipeline that can take ANY dataset and prepare it for training.

### Concepts to Learn:
1. **ETL (Extract, Transform, Load):** The core philosophy of Data Engineering.
2. **Parquet / Arrow:** High-performance columnar data formats that are better than CSVs for massive datasets.
3. **Data Contracts:** Defining strict schemas (using `Pydantic` in Python) so bad data crashes the pipeline *before* it reaches the database.

### Your Action Items:
* [ ] **Task 3.1:** Define a `Pydantic` model representing a "Universal Data Sample" (ID, Image Path, Labels, Metadata dict).
* [ ] **Task 3.2:** Write a script that parses the `PlantNet-300K` dataset, validates it against your Pydantic model, and saves the metadata as a `.parquet` file.
* [ ] **Task 3.3:** Hook this into your Rust WebDataset packer so it reads from Parquet instead of CSV.

---

## ⚖️ Phase 4: Neuro-Symbolic (NeSy) Integration
**Goal:** Replace hardcoded AC-3 logic with a differentiable logic framework.

### Concepts to Learn:
1. **Probabilistic Logic Programming (PLP):** Frameworks like **Dolphin** or **Scallop**.
2. **Logic Tensor Networks (LTN):** Frameworks like **LTNtorch** (best for injecting FOL axioms as a regularizer).

### 4090 Hardware Strategy (Crucial):
*   **Split Compute:** Neural forward/backward runs on GPU (BF16). Symbolic enumeration runs on CPU. Probabilistic aggregation runs on GPU (FP32). 
*   **Vectorized Provenance:** Do not use exact #SAT (it's #P-hard). Use top-k proofs (start with $k=3$).
*   **Curriculum:** Train depth 2 -> depth 5 -> depth 10. Do not start at depth 10 or the gradients will vanish.

### Your Action Items:
* [ ] **Task 4.1:** Choose your framework. If you want "Rules + Probabilities" (Perception -> Reasoning), install **Dolphin**. If you want "NN + Constraints", install **LTNtorch**.
* [ ] **Task 4.2:** Rewrite your AC-3 constraints as Logical Axioms in Python. 
* [ ] **Task 4.3:** Connect your PyTorch dataloader (pulling from Postgres/WebDataset) to feed tensors into the Neural module, and pass the outputs to the NeSy layer.
* [ ] **Task 4.4:** Implement Gradient Checkpointing on the perception net (not the reasoner) to save VRAM.

---

## 📊 Phase 5: Automated Model Lineage
**Goal:** Track exactly what data trained what model.

### Concepts to Learn:
1. **MLOps Tracking:** Building a relational model registry.

### Your Action Items:
* [ ] **Task 5.1:** Create `Models`, `Datasets`, and `TrainingRuns` tables in Postgres.
* [ ] **Task 5.2:** Update `oracle.py` so that when a training run starts, it logs the `run_id`, hyperparameters, and dataset hash into Postgres.
