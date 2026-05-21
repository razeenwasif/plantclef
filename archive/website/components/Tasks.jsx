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
      "Models train on 1.4M isolated single plant photos from Pl@ntNet 2024, then compose those representations into a multi label prediction on plots where species co occur and occlude each other.",
    ],
    spec: [
      ["Input",       "1 RGB image, 800px max side"],
      ["Output",      "Top K species set, we use K=3"],
      ["Train",       "1,408,033 single plant photos"],
      ["Test",        "2,105 LUCAS quadrats"],
      ["Metric",      "Macro F1 per sample"],
      ["Our public",  "0.41826"],
      ["Our private", "0.40283"],
    ],
  },
  {
    key: "pipeline",
    label: "Our pipeline",
    title: "BioCLIP 2.5 plus tax aux heads, partial unfreeze",
    desc: [
      "Single backbone fine tune of BioCLIP 2.5 ViT H/14. Only the last 4 of 32 transformer blocks are unfrozen. The Tree of Life pretraining in the lower 28 blocks is the load bearing prior. Unfreezing more than 4 blocks costs Kaggle F1.",
      "Head is a SharedMLP plus species head, with auxiliary heads on genus, family, order, class. Inference uses grid 4 by 4 tiling at tile size 448, softmax mean across tiles, top 3 selection.",
    ],
    spec: [
      ["Backbone",   "BioCLIP 2.5 ViT H/14"],
      ["Unfreeze",   "last 4 blocks plus ln_post, proj"],
      ["Head",       "SharedMLP plus 4 tax aux heads"],
      ["Loss",       "joint CE with tax weights"],
      ["Inference",  "grid 4 by 4, softmax mean"],
      ["Selection",  "top 3 or probT 0.05"],
    ],
  },
  {
    key: "data",
    label: "Data and domain",
    title: "Single plant to multi species shift",
    desc: [
      "Train images are 800px specimen close ups. Test images are LUCAS in situ quadrats with 4 times the pixel count and a wider field of view. Models that fit the train distribution well do not always transfer.",
      "Experiment 015 mixed PC24 with 1.25M iNaturalist habitat shots at 50:50. Validation top 1 climbed to a project best 0.8255 but Kaggle F1 dropped below the no iNat anchor. i002 and i003 attacked the long tail instead.",
    ],
    spec: [
      ["Train",       "aspect 1.0, single plant"],
      ["Test",        "aspect 1.34, multi species"],
      ["Long tail",   "1,644 species with under 100 images"],
      ["Extra data",  "iNat 1.25M, cap plus extra"],
      ["Val Kaggle",  "Pearson r ≈ 0.4"],
      ["Lesson",      "data, not capacity"],
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
