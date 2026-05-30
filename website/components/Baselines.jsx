import Reveal from "./Reveal";

export default function Baselines() {
  const cards = [
    {
      tag: "FORTHCOMING",
      title: "Method paper",
      copy: "Partial unfreeze sweet spot, long tail capping, validation versus leaderboard decoupling, and ablations across the experiment ladder. Will be linked here when published.",
      meta: ["CEUR WS", "In preparation"],
      href: undefined,
    },
    {
      tag: "REFERENCE",
      title: "Experiment ladder",
      copy: "Per experiment summaries from 001 through i003 with recipe, validation metrics, Kaggle public and private scores, and the lessons that carried forward.",
      meta: ["28 directions", "600+ submissions"],
      href: undefined,
    },
    {
      tag: "REFERENCE",
      title: "Data manifest",
      copy: "The deployed i002 model trains on the 2.65 million image i001 manifest combining PlantCLEF 2024 with a research grade iNaturalist pull, stratified 90/10 by species with seed 42.",
      meta: ["2,653,781 images", "7,806 species"],
      href: undefined,
    },
  ];
  return (
    <section id="code">
      <div className="shell">
        <div className="section-head">
          <div>
            <p className="eyebrow">Resources</p>
            <h2>References</h2>
          </div>
          <p className="lede">
            Method paper and per experiment notes. Public release pending.
          </p>
        </div>
        <Reveal className="cards" stagger>
          {cards.map((c) => {
            const Tag = c.href ? "a" : "div";
            return (
              <Tag className="card" href={c.href} key={c.title} data-disabled={c.href ? undefined : 1}>
                <div className="tag">{c.tag}</div>
                <h4>{c.title}</h4>
                <p>{c.copy}</p>
                <div className="meta">
                  <span>{c.meta[0]}</span>
                  <span className="dot">·</span>
                  <span>{c.meta[1]}</span>
                </div>
              </Tag>
            );
          })}
        </Reveal>
      </div>
    </section>
  );
}
