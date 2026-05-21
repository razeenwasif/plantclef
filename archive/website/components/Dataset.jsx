import Reveal from "./Reveal";

export default function Dataset() {
  const cells = [
    {
      num: "01 · TRAIN",
      title: "PlantCLEF 2024 single plant",
      copy: "Pl@ntNet close up photographs across 7,806 species. The supervised training set for every experiment.",
      stat: "1,408,033 images",
      meta: "PC24",
    },
    {
      num: "02 · TEST",
      title: "LUCAS vegetation plots",
      copy: "In situ 1m² quadrats with 1 to 10 species each, expert annotated. The distribution we need to transfer to.",
      stat: "2,105 plots",
      meta: "LUCAS",
    },
    {
      num: "03 · EXTRA",
      title: "iNaturalist research grade",
      copy: "Habitat shot photographs pulled from GBIF in experiment 014a, used in the 015 PC24 plus iNat mix.",
      stat: "1.25M images",
      meta: "iNat",
    },
    {
      num: "04 · TAXONOMY",
      title: "GBIF taxonomic backbone",
      copy: "7,806 species into 1,446 genera, 181 families, 61 orders, 6 classes. Drives the multi task aux losses.",
      stat: "7,806 nodes",
      meta: "GBIF",
    },
  ];
  return (
    <section id="data">
      <div className="shell">
        <div className="section-head">
          <div>
            <p className="eyebrow">Datasets</p>
            <h2>Data sources</h2>
          </div>
          <p className="lede">
            All splits share the canonical 7,806 species axis. Taxonomy joins
            on <span className="k-mono">gbif_species_id</span>.
          </p>
        </div>
        <Reveal className="dataset-grid" stagger>
          {cells.map((c) => (
            <div className="data-card" key={c.num}>
              <div className="num">{c.num}</div>
              <h4>{c.title}</h4>
              <p>{c.copy}</p>
              <div className="stat-line">
                <span>{c.stat}</span>
                <span className="sep">·</span>
                <span style={{ color: "var(--muted)" }}>{c.meta}</span>
              </div>
            </div>
          ))}
        </Reveal>
      </div>
    </section>
  );
}
