import { TEAM } from "@/lib/team";
import { asset } from "@/lib/paths";

const CELLS = [
  { src: "/quadrats/CBN-PdlC-A1-20130807.jpg",                        id: "CBN PdlC A1" },
  { src: "/quadrats/CBN-Pla-A1-20130808.jpg",                          id: "CBN Pla A1"  },
  { src: "/quadrats/GUARDEN-CBNMed-10-1-16-34-20240429.jpg",           id: "GUARDEN 10"  },
  { src: "/quadrats/LISAH-JAS-0-1-20220505.jpg",                       id: "LISAH JAS"   },
  { src: "/quadrats/OPTMix-0108-P1-312-20231006.jpg",                  id: "OPTMix 0108" },
  { src: "/quadrats/RNNB-1-1-20230512.jpg",                            id: "RNNB 1 1"    },
];

export default function Hero() {
  return (
    <section className="hero" id="top">
      <div className="hero-inner">
        <div className="badges">
          <span className="badge-pill">PlantCLEF · 2026</span>
          <span className="badge-pill">Submission recap</span>
          <span className="badge-pill">7th place · private F1</span>
        </div>

        <h1>
          <em>PlantCLEF&nbsp;2026</em><br />
          multi species identification in 1m² quadrats
        </h1>

        <p className="subhead">
          Submission to the 7th LifeCLEF plant identification challenge.
          Best public Kaggle F1 <b>0.41826</b>, private F1 <b>0.40283</b>,
          good for <b>7th place</b> on the private leaderboard.
        </p>

        <div className="cta-row">
          <a className="btn" href="https://www.imageclef.org/PlantCLEF2026">
            Competition page&nbsp;<span className="arrow">↗</span>
          </a>
        </div>
      </div>

      <div className="hero-strip">
        <div className="hero-strip-inner">
          <div className="hero-strip-head">
            <span className="label">LUCAS test quadrats · 1m²</span>
            <span className="cap">Sample of the 2,105 image evaluation set</span>
          </div>
          <div className="hero-strip-row">
            {CELLS.map((c, i) => (
              <div key={i} className="hero-cell">
                <img src={asset(c.src)} alt={`LUCAS quadrat ${c.id}`} loading="lazy" />
                <div className="label">
                  <span style={{ color: "var(--muted)" }}>{String(i + 1).padStart(2, "0")}</span>
                  <em>{c.id}</em>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="authors-strip">
        <div className="authors-inner">
          <div className="lbl">Team</div>
          <div>
            <div className="au-line">
              {TEAM.map((a, i) => (
                <span key={a.name}>
                  {a.link ? (
                    <a className="au" href={a.link} target="_blank" rel="noreferrer">{a.name}</a>
                  ) : (
                    <span className="au">{a.name}</span>
                  )}
                  {i < TEAM.length - 1 && <span className="sep">·</span>}
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
