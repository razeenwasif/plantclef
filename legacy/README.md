# Legacy code and pre-paper planning notes

Everything in this directory predates the system reported in
`report/main.tex`. It is kept solely for traceability of the
development trace, and **is not used by the paper code**.

If you are reading this as the assessor, you can safely ignore this
directory. The paper system lives in:

* `src/inf_script.py` - headline inference recipe (0.41826 public Macro F1)
* `src/inf_script_phen.py` - seasonal-phenology pivot (paper pivot 3)
* `src/train.py` - thin shim that forwards to the i002 training script
* `experiments/i002_bioclip25_cap_image/` - i002 training code (the paper model)
* `experiments/i001_data_download/` - manifest construction
* `experiments/i003_bioclip25_cap_image_extra500/` - per-species-cap counterfactual

## Contents

### Legacy source code

The early plan for this project was a triple-backbone
(BioCLIP + DINOv2-Large + ConvNeXt-V2-Large) LoRA ensemble with NVIDIA
DALI for GPU-side image decoding, Asymmetric Loss for multi-label
training, and a DeepSpeed-orchestrated two-phase pipeline. The
following directories implement that approach:

* `config.py`, `data/`, `models/`, `training/`, `extensions/`

We abandoned this direction in favour of partially fine-tuning
BioCLIP 2.5 ViT-H/14 with per-head taxonomy MLPs (the i002 model). The
paper's experiment table records the lines that did make it to a
submission (e.g. experiments 005, 008, 010, 011) under
`experiments/`.

### Stale planning documents

* `research_proposal.md` - the initial project proposal (24 Mar 2026),
  written when the plan was still the triple-backbone ensemble. Does
  not reflect the final system.
* `todo.md` - master TODO from the same early plan; the unchecked
  items are mostly directions later abandoned or superseded.
* `background.txt` - background research notes the team collected at
  the start of the project.

All three are useful for the development trace but contradict the
paper, so they have been moved out of the repository root to avoid
confusing a reader.
