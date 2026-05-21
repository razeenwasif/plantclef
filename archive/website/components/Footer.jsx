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
            By Arjun Raj, Razeen Wasif, Manindra de Mel, Will Brake. Our
            submission to the 7th edition of the LifeCLEF plant
            identification challenge.
          </p>
        </div>
        <div className="foot-links">
          <a href="https://github.com/arm-wision/arm-wision.github.io">GitHub</a>
          <a href="https://www.imageclef.org/PlantCLEF2026">Competition page ↗</a>
          <a href="#cite">Cite</a>
        </div>
      </div>
      <div className="foot-legal">
        <span>© 2026 Raj, Wasif, de Mel, Brake. ANU Deep Learning.</span>
        <span>Built for the LifeCLEF research evaluation initiative.</span>
      </div>
    </footer>
  );
}
