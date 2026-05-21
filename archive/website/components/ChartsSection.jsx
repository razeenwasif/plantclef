import ScoreProgression from "./charts/ScoreProgression";
import UnfreezeSweep from "./charts/UnfreezeSweep";
import ValVsKaggle from "./charts/ValVsKaggle";
import E015PerEpoch from "./charts/E015PerEpoch";
import SpeciesLongTail from "./charts/SpeciesLongTail";
import Reveal from "./Reveal";

export default function ChartsSection() {
  return (
    <section id="charts" className="soft">
      <div className="shell">
        <div className="section-head">
          <div>
            <p className="eyebrow">Results</p>
            <h2>Figures</h2>
          </div>
          <p className="lede">
            Scores pulled from the Kaggle public API for
            <code className="k-mono"> plantclef-2026</code> and merged with
            our local submission logs (240 submissions across 18 experiments).
          </p>
        </div>

        <div className="chart-grid">
          <Reveal variant="fade-up"><ScoreProgression /></Reveal>
          <div className="chart-grid-row">
            <Reveal variant="fade-up"><UnfreezeSweep /></Reveal>
            <Reveal variant="fade-up" delay={120}><ValVsKaggle /></Reveal>
          </div>
          <div className="chart-grid-row">
            <Reveal variant="fade-up"><E015PerEpoch /></Reveal>
            <Reveal variant="fade-up" delay={120}><SpeciesLongTail /></Reveal>
          </div>
        </div>
      </div>
    </section>
  );
}
