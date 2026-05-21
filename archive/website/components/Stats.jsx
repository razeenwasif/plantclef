import Reveal from "./Reveal";

export default function Stats() {
  const items = [
    { k: <>7<em>,</em>806</>,            l: "Species in catalogue",          sup: "vasc." },
    { k: <>1.4<em>M</em></>,             l: "Single-plant train images",     sup: "PC24"  },
    { k: <>2<em>,</em>105</>,            l: "Test quadrats (LUCAS)",         sup: null    },
    { k: <>0.418<em>26</em></>,          l: "Best public F1, private 0.40283", sup: "ours"  },
  ];
  return (
    <div className="stats-section">
      <div className="shell">
        <Reveal variant="fade-up" stagger className="stats">
          {items.map((it, i) => (
            <div className="stat" key={i}>
              <div className="k">
                {it.k}
                {it.sup && <sup>{it.sup}</sup>}
              </div>
              <div className="l">{it.l}</div>
            </div>
          ))}
        </Reveal>
      </div>
    </div>
  );
}
