export default function Footer() {
  return (
    <footer className="foot">
      <div className="foot-inner-recap">
        <div className="brand-block">
          <div className="brand-row">
            <span className="mark" aria-hidden="true" />
            <span>PlantCLEF&nbsp;2026 · Submission recap</span>
          </div>
          <p className="brand-tag">
            By Arjun Raj, Manindra de Mel, Razeen Wasif, Will Brake.
          </p>
        </div>
        <div className="foot-links">
          <a href="https://www.imageclef.org/PlantCLEF2026">Competition page ↗</a>
          <a href="#cite">Cite</a>
        </div>
      </div>
      <div className="foot-legal">
        <span>© 2026 Raj, de Mel, Wasif, Brake. ANU Deep Learning.</span>
      </div>
    </footer>
  );
}
