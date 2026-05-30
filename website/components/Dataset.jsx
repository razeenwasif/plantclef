import Reveal from "./Reveal";

export default function Dataset() {
  const cells = [
    {
      num: "01 · TRAIN",
      title: "PlantCLEF 2024 single plant",
      copy: "Pl@ntNet close up photographs across 7,806 species. The original supervised training set.",
      stat: "1,408,033 images",
      meta: "PC24",
    },
    {
      num: "02 · TRAIN ADD",
      title: "iNaturalist research grade",
      copy: "Research grade iNaturalist images, de duplicated and image verified, added to the manifest during the i001 data rebuild.",
      stat: "1,245,748 images",
      meta: "iNat",
    },
    {
      num: "03 · TRAIN UNION",
      title: "i001 manifest (deployed)",
      copy: "What the deployed i002 model trains on: stratified 90/10 by species (seed 42), species with under 5 images held entirely in train.",
      stat: "2,653,781 images",
      meta: "i001",
    },
    {
      num: "04 · TEST",
      title: "LUCAS vegetation plots",
      copy: "In situ 1m² quadrats with 1 to 10 species each, expert annotated. The distribution we transfer to.",
      stat: "2,105 plots",
      meta: "LUCAS",
    },
    {
      num: "05 · TAXONOMY",
      title: "GBIF taxonomic backbone",
      copy: "7,806 species into 1,446 genera and 181 families. Drives the auxiliary taxonomy heads at training time.",
      stat: "7,806 / 1,446 / 181",
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
