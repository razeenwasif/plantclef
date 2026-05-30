import Reveal from "./Reveal";

export default function Abstract() {
  return (
    <section id="overview">
      <div className="shell narrow">
        <Reveal className="abstract" variant="fade-up">
          <div>
            <p className="eyebrow">Overview</p>
            <p className="meta">
              7th edition<br />
              LifeCLEF 2026<br />
              September 2026
            </p>
          </div>
          <div className="body">
            <p>
              We train on isolated single plant photographs and predict every
              species visible in a 1m² in situ vegetation quadrat. Training
              and test data live in different visual worlds.
            </p>
            <p>
              Our pipeline is a single fine tuned BioCLIP 2.5 ViT H/14 with
              only the last 4 of 32 transformer blocks unfrozen, per task
              MLP heads at species, genus and family, and a 4 by 4 tile
              inference at two resolutions. More clean training data
              without a per species cap (the 2.65 million image i001
              manifest combining PlantCLEF 2024 with research grade
              iNaturalist) is what gave the best result; an explicit cap
              and tail augmentation in i003 lowered public F1.
            </p>
            <p>
              Across more than 20 experiments and 600+ Kaggle submissions,
              the bottleneck kept being the <em>single plant to multi
              species distribution shift</em>, not model capacity.
              Held out single plant validation accuracy correlates with
              quadrat F1 at only r ≈ 0.4. Best public F1 of <b>0.41826</b>,
              private F1 of <b>0.40283</b>, from a 224 plus 336 px
              ensemble of our i002 checkpoint with class prior logit
              adjustment (τ = 0.25) and an adaptive probability threshold
              (T = 0.03, k ∈ [2, 10]).
            </p>
          </div>
        </Reveal>
      </div>
    </section>
  );
}
