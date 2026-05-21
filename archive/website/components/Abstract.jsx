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
              the last 4 transformer blocks unfrozen, taxonomic auxiliary
              heads at genus, family, order, class, and 4 by 4 tile inference.
              Long tail data capping with extra under 100 species images is
              what pushed us to the best score.
            </p>
            <p>
              Across 18 experiments and 240 Kaggle submissions, the bottleneck
              kept being <em>training data distribution</em>, not model
              capacity. Validation accuracy proved to be a weak Kaggle
              predictor. Best public F1 of <b>0.41826</b>, private F1 of
              <b> 0.40283</b>, from a 224 plus 336 px ensemble of our
              i002 checkpoint.
            </p>
          </div>
        </Reveal>
      </div>
    </section>
  );
}
