const TIMELINE = [
  { date: "2026 03",     ev: "001, 002 BioCLIP zero shot tile, ~0.05 floor" },
  { date: "2026 04",     ev: "005 DINOv3 multi label, 0.13. No plant prior." },
  { date: "2026 04",     ev: "006 BioCLIP 2.5 frozen prototype, 0.33" },
  { date: "2026 04",     ev: "008 DINOv3 PlantNet FT plus RRF fusion, 0.34642" },
  { date: "2026 04",     ev: "009 full FT collapses to 0.207. Partial unfreeze hypothesis." },
  { date: "2026 04 27",  ev: "010 last_blocks 4 multi task breakthrough, 0.38333", tag: "milestone" },
  { date: "2026 04 29",  ev: "012, 013 SSL on LUCAS, net negative at OpenCLIP LR" },
  { date: "2026 05 02",  ev: "014b unfreeze sweep, n=4 is the sweet spot" },
  { date: "2026 05 03",  ev: "015 PC24 plus iNat 50:50, dead recipe at 0.37956" },
  { date: "2026 05 04",  ev: "i002 long tail cap plus extra data, single model at 0.41165" },
  { date: "2026 05 06",  ev: "i002 224 plus 336 px ensemble, team best at 0.41826 (private 0.40283)", tag: "milestone" },
  { date: "2026 05 07",  ev: "i003 500 cap variant, 0.40041" },
  { date: "07 May 2026",  ev: "Submission deadline", tag: "competition" },
  { date: "Sep 2026",         ev: "CLEF 2026", tag: "venue" },
];

import Reveal from "./Reveal";

export default function Timeline() {
  return (
    <section id="timeline">
      <div className="shell narrow">
        <div className="section-head">
          <div>
            <p className="eyebrow">How the project ran</p>
            <h2>Timeline.</h2>
          </div>
          <p className="lede">
            18 experiments, 240 submissions, two breakthroughs: partial
            unfreeze (010) and long tail capping (i002).
          </p>
        </div>

        <Reveal as="ul" className="timeline" stagger threshold={0.05}>
          {TIMELINE.map((t, i) => (
            <li key={i} data-done={t.date.startsWith("2026") ? 1 : 0}>
              <span className="date">{t.date}</span>
              <span className="dot" />
              <span className="ev">
                {t.ev}
                {t.tag && <span className="tag">· {t.tag}</span>}
              </span>
            </li>
          ))}
        </Reveal>
      </div>
    </section>
  );
}
