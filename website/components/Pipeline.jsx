import { asset } from "@/lib/paths";

export default function Pipeline() {
  return (
    <section className="pipeline-section" id="pipeline">
      <div className="shell narrow">
        <div className="section-head">
          <div>
            <p className="eyebrow">Pipeline</p>
            <h2>End to end animation</h2>
          </div>
          <p className="lede">
            Quadrat in, species set out. The same i002 BioCLIP 2.5 checkpoint
            runs at 224 and 336 pixels over a 4 by 4 tile grid, the per tile
            softmax distributions are averaged, class prior logit adjustment
            is applied, and an adaptive probability threshold emits the final
            species set.
          </p>
        </div>
        <div className="video-frame">
          <video
            src={asset("/pipeline.mp4")}
            autoPlay
            muted
            loop
            playsInline
            preload="auto"
          />
        </div>
      </div>
    </section>
  );
}
