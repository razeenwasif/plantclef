import { TEAM } from "@/lib/team";
import Reveal from "./Reveal";

export default function TeamBlock() {
  return (
    <section className="soft" id="team">
      <div className="shell">
        <div className="section-head">
          <div>
            <p className="eyebrow">Team</p>
            <h2>Authors</h2>
          </div>
          <p className="lede">ANU Deep Learning, Semester 1 2026.</p>
        </div>

        <Reveal className="org-grid org-grid-no-av" stagger>
          {TEAM.map((o) => (
            <div className="org" key={o.uid}>
              {o.link ? (
                <a className="name" href={o.link} target="_blank" rel="noreferrer">
                  {o.name} <span className="ext-arrow" aria-hidden="true">↗</span>
                </a>
              ) : (
                <span className="name">{o.name}</span>
              )}
              <span className="aff k-mono">{o.uid}</span>
            </div>
          ))}
        </Reveal>
      </div>
    </section>
  );
}
