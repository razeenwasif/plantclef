"use client";

import { useEffect, useState } from "react";

export default function Topbar() {
  const [scrolled, setScrolled] = useState(false);
  useEffect(() => {
    const on = () => setScrolled(window.scrollY > 8);
    on();
    window.addEventListener("scroll", on, { passive: true });
    return () => window.removeEventListener("scroll", on);
  }, []);

  return (
    <header className="topbar" data-scrolled={scrolled ? 1 : 0}>
      <div className="topbar-inner">
        <a href="#top" className="brand">
          <span className="mark" aria-hidden="true" />
          <span>
            PlantCLEF<span style={{ color: "var(--muted-soft)" }}>&nbsp;/&nbsp;2026</span>
          </span>
        </a>
        <nav>
          <a href="#overview">Overview</a>
          <a href="#task">Task</a>
          <a href="#data">Data</a>
          <a href="#charts">Charts</a>
          <a href="#leaderboard">Results</a>
          <a href="#cite">Cite</a>
        </nav>
        <div className="nav-right">
          <a href="https://github.com/arm-wision/arm-wision.github.io" className="nav-link">GitHub</a>
          <a href="https://www.imageclef.org/PlantCLEF2026" className="nav-link external">
            Competition&nbsp;<span className="ext-arrow">↗</span>
          </a>
        </div>
      </div>
    </header>
  );
}
