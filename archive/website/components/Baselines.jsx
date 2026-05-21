import Reveal from "./Reveal";

export default function Baselines() {
  const cards = [
    {
      tag: "OURS",
      title: "GitHub",
      copy: "End to end training scripts, manifest builders, DDP wrappers, per experiment summaries, and the Kaggle submission log.",
      meta: ["github.com/arm-wision/arm-wision.github.io", "PyTorch 2.x"],
      href: "https://github.com/arm-wision/arm-wision.github.io",
    },
    {
      tag: "REFERENCE",
      title: "Experiments index",
      copy: "Per experiment summaries from 001 through i003 with recipe, val metrics, Kaggle scores, and lessons.",
      meta: ["docs/experiments_summary.md", "18 experiments"],
      href: "https://github.com/arm-wision/arm-wision.github.io/blob/main/docs/experiments_summary.md",
    },
    {
      tag: "FORTHCOMING",
      title: "Method paper",
      copy: "Partial unfreeze sweet spot, long tail capping, and ablations across all 18 experiments. Will be linked here when published.",
      meta: ["CEUR WS", "In preparation"],
      href: undefined,
    },
  ];
  return (
    <section id="code">
      <div className="shell">
        <div className="section-head">
          <div>
            <p className="eyebrow">Resources</p>
            <h2>Code and references</h2>
          </div>
          <p className="lede">
            Code and per experiment writeups. Every numbered experiment
            under <code className="k-mono">src_experiments/</code> has its
            own <code className="k-mono">summary.md</code>.
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
