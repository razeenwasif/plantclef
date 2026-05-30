"use client";

import { useState } from "react";
import Reveal from "./Reveal";

const TASKS = [
  {
    key: "task",
    label: "The task",
    title: "Multi label species ID in 1m² quadrats",
    desc: [
      "Given a top down photograph of a LUCAS vegetation plot, predict every vascular plant species visible. Plots contain 1 to 10 species at varying scale.",
      "Models train on isolated single plant photos and then compose those representations into a multi label prediction on plots where species co occur and occlude each other. The 7,806 species axis is fixed.",
    ],
    spec: [
      ["Input",       "1 RGB image, 800px max side"],
      ["Output",      "Adaptive species set, k ∈ [2, 10]"],
      ["Train",       "2,653,781 single plant photos (i001)"],
      ["Test",        "2,105 LUCAS quadrats"],
      ["Metric",      "Macro F1 per quadrat"],
      ["Our public",  "0.41826"],
      ["Our private", "0.40283"],
    ],
  },
  {
    key: "pipeline",
    label: "Our pipeline",
    title: "BioCLIP 2.5 with taxonomic auxiliary heads, partial unfreeze",
    desc: [
      "Single backbone fine tune of BioCLIP 2.5 ViT H/14. Only the last 4 of 32 transformer blocks are unfrozen plus the final layer norm and projection. The Tree of Life pretraining in the lower 28 blocks is the load bearing prior. Unfreezing more or fewer blocks costs Kaggle F1.",
      "Three independent per task MLP heads predict species, genus and family. Inference partitions each quadrat into a 4 by 4 grid, runs the encoder at 224 and 336 pixels, averages the per tile softmax distributions across tiles and resolutions, applies class prior logit adjustment (τ = 0.25), and emits every species whose probability exceeds T = 0.03 (clamped to k ∈ [2, 10]).",
    ],
    spec: [
      ["Backbone",   "BioCLIP 2.5 ViT H/14"],
      ["Unfreeze",   "last 4 blocks plus ln_post, proj"],
      ["Heads",      "3 per task MLPs: species, genus, family"],
      ["Loss",       "species CE + 0.30 genus + 0.15 family"],
      ["Inference",  "grid 4x4, 224 + 336 px ensemble"],
      ["Selection",  "adaptive probT, T = 0.03, k ∈ [2, 10]"],
    ],
  },
  {
    key: "data",
    label: "Data and domain",
    title: "Single plant to multi species shift",
    desc: [
      "Train images are 800px specimen close ups. Test images are LUCAS in situ quadrats with a wider field of view and many overlapping plants. Models that fit the train distribution well do not always transfer.",
      "More clean training data without a per species cap is what gave the best result. Capping at 500 images per species and adding 161,686 GBIF tail images (experiment i003) lowered public F1 from 0.41826 to 0.40041; mixing in iNaturalist habitat shots at 50:50 (experiment 015) raised validation but lowered public score.",
    ],
    spec: [
      ["Train shape", "aspect 1.0, single plant"],
      ["Test shape",  "aspect 1.34, multi species"],
      ["Long tail",   "2,786 species with under 100 images"],
      ["i003 cap",    "500/species + 161k extra: −0.018 F1"],
      ["Val ⇄ Kaggle","Pearson r ≈ 0.4 (decoupled)"],
      ["Lesson",      "distribution shift, not capacity"],
    ],
  },
];

export default function Tasks() {
  const [active, setActive] = useState("task");
  const task = TASKS.find((t) => t.key === active);

  return (
    <section className="soft" id="task">
      <div className="shell narrow">
        <div className="section-head">
          <div>
            <p className="eyebrow">The task</p>
            <h2>Task overview</h2>
          </div>
          <p className="lede">
            PlantCLEF 2026 is a single multi label track over 7,806 species.
            Training images are single plant close ups; test images are
            multi species LUCAS quadrats.
          </p>
        </div>

        <Reveal className="tabs" variant="fade" role="tablist">
          {TASKS.map((t, i) => (
            <button
              key={t.key}
              role="tab"
              aria-selected={active === t.key}
              onClick={() => setActive(t.key)}
            >
              <span className="num">0{i + 1}</span>
              {t.label}
            </button>
          ))}
        </Reveal>

        <div className="task-panel fade-in" key={active}>
          <div className="desc">
            <h3>{task.title}</h3>
            {task.desc.map((p, i) => (
              <p key={i}>{p}</p>
            ))}
          </div>
          <div className="spec">
            {task.spec.map(([k, v]) => (
              <div className="row" key={k}>
                <span className="k">{k}</span>
                <span className="v">{v}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
