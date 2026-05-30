"use client";

import { useState } from "react";
import Reveal from "./Reveal";

const BIB = `@inproceedings{armwision2026plantclef,
  title     = {ARM Wision at PlantCLEF 2026: Our Submission to
               the Multi Species Vegetation Plot Challenge},
  author    = {Raj, Arjun and de Mel, Manindra and
               Wasif, Razeen and Brake, Will},
  booktitle = {Working Notes of CLEF 2026},
  series    = {CEUR Workshop Proceedings},
  year      = {2026}
}`;

export default function BibTeX() {
  const [copied, setCopied] = useState(false);
  const onCopy = async () => {
    try {
      await navigator.clipboard.writeText(BIB);
      setCopied(true);
      setTimeout(() => setCopied(false), 1400);
    } catch (e) {
      /* ignore */
    }
  };
  return (
    <section id="cite">
      <div className="shell narrow">
        <div className="section-head">
          <div>
            <p className="eyebrow">Cite this work</p>
            <h2>BibTeX.</h2>
          </div>
          <p className="lede">
            If you build on this work, please also cite the PlantCLEF 2026
            overview paper and the Pl@ntNet observation pipeline.
          </p>
        </div>
        <Reveal className="bibtex" variant="fade-up">
          <button className="copy" data-copied={copied ? 1 : 0} onClick={onCopy}>
            {copied ? "Copied ✓" : "Copy"}
          </button>
          {BIB}
        </Reveal>
      </div>
    </section>
  );
}
