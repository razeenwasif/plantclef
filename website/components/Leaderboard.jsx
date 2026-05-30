"use client";

import { useMemo, useState } from "react";
import Reveal from "./Reveal";

// Public + private F1 sourced from the Kaggle competition API. Rows lacking a
// private score fall outside the API submission window and are not retrievable.
const SUBMISSIONS = [
  { exp: "i002",        recipe: "ensemble 224 + 336 px, LA tau 0.25, probT 0.03",        public: 0.41826, private: 0.40283, family: "cap"   },
  { exp: "i002",        recipe: "224 + 336 + GBIF DOY phenology Boltzmann prior",        public: 0.41346, private: 0.38766, family: "cap"   },
  { exp: "i002",        recipe: "224 + 336, LA tau 0.25, genus rerank, probT 0.035",     public: 0.41246, private: 0.39946, family: "cap"   },
  { exp: "i002",        recipe: "last_blocks 4, softmax mean, probT 0.05",               public: 0.41165, private: 0.38132, family: "cap"   },
  { exp: "i002",        recipe: "last_blocks 336 px (pos_embed bicubic resize)",         public: 0.41117, private: 0.38376, family: "cap"   },
  { exp: "i002",        recipe: "LA tau 0.25, softmax mean probT 0.03",                  public: 0.40590, private: 0.40600, family: "cap"   },
  { exp: "i002",        recipe: "last_blocks 8 epoch 5, softmax mean top 3",             public: 0.40486, private: 0.35614, family: "cap"   },
  { exp: "i002",        recipe: "last_blocks 4, softmax mean top 3",                     public: 0.40451, private: 0.36380, family: "cap"   },
  { exp: "i003",        recipe: "last_blocks 4, softmax mean top 3",                     public: 0.40041, private: 0.36078, family: "cap"   },
  { exp: "i003",        recipe: "last_blocks 4, softmax mean probT 0.05",                public: 0.39673, private: 0.37571, family: "cap"   },
  { exp: "i003",        recipe: "last_blocks 4, softmax mean probT 0.03",                public: 0.39603, private: 0.38937, family: "cap"   },
  { exp: "010",         recipe: "last_blocks 4, softmax mean top 3 (anchor)",            public: 0.39140, private: 0.38321, family: "ours"  },
  { exp: "i002 + 010",  recipe: "ensemble i002_last_blocks + 010_last_blocks_8",         public: 0.38874, private: 0.40534, family: "fuse"  },
  { exp: "010",         recipe: "last_blocks 8, grid_4x4, gap 0.5 + hflip TTA",          public: 0.38545, private: 0.39192, family: "ours"  },
  { exp: "011",         recipe: "alpha sweep, 0.75 mean x noisy or",                     public: 0.38278, private: 0.33186, family: "ours"  },
  { exp: "010",         recipe: "last_blocks 8, multiscale (whole + 2x2 + 4x4)",         public: 0.37942, private: 0.37721, family: "ours"  },
  { exp: "015",         recipe: "PC24 + iNat 50:50, unfreeze 4, 5 ep",                   public: 0.37956, private: 0.33684, family: "inat"  },
  { exp: "015",         recipe: "PC24 + iNat 50:50, unfreeze 4, 1 ep",                   public: 0.37506, private: 0.34190, family: "inat"  },
  { exp: "014b",        recipe: "unfreeze 3 sweep",                                      public: 0.37455, private: 0.34247, family: "sweep" },
  { exp: "014b",        recipe: "unfreeze 5 sweep",                                      public: 0.36919, private: 0.34265, family: "sweep" },
  { exp: "008 + 006",   recipe: "RRF alpha 0.65 PhaseA x frozen BioCLIP",                public: 0.34671, private: 0.33699, family: "fuse"  },
  { exp: "008 + 006",   recipe: "RRF alpha 0.70 PhaseA x frozen BioCLIP",                public: 0.34642, private: 0.31222, family: "fuse"  },
  { exp: "013",         recipe: "SSL + multitask + tax heads",                           public: 0.33185, private: null,    family: "ssl"   },
  { exp: "006",         recipe: "frozen BioCLIP 2.5 prototype",                          public: 0.33000, private: null,    family: "base"  },
  { exp: "008",         recipe: "DINOv3 PhaseA multi scale, top 3 max",                  public: 0.30500, private: null,    family: "base"  },
  { exp: "012",         recipe: "SSL + 006 plain Linear",                                public: 0.30261, private: null,    family: "ssl"   },
  { exp: "009",         recipe: "BioCLIP 2.5 full FT (collapsed)",                       public: 0.20777, private: null,    family: "base"  },
  { exp: "005",         recipe: "DINOv3 multi label two phase",                          public: 0.13000, private: null,    family: "base"  },
];

const COLS = [
  { key: "rank",    label: "#",         num: true,  sortable: false, width: 48 },
  { key: "exp",     label: "Exp",       num: false, sortable: true,  width: 90 },
  { key: "recipe",  label: "Recipe",    num: false, sortable: true },
  { key: "public",  label: "Public F1", num: true,  sortable: true },
  { key: "private", label: "Private F1",num: true,  sortable: true },
];

const FAMILY_LABEL = {
  cap:   "Cap, extra",
  ours:  "010 family",
  inat:  "iNat mix",
  sweep: "Sweep",
  fuse:  "Fusion",
  ssl:   "SSL",
  base:  "Baseline",
};

const COLLAPSED_COUNT = 5;

export default function Leaderboard() {
  const [filter, setFilter] = useState("all");
  const [sort, setSort] = useState({ key: "public", dir: "desc" });
  const [q, setQ] = useState("");
  const [expanded, setExpanded] = useState(false);

  const sortedRows = useMemo(() => {
    let r = SUBMISSIONS.slice();
    if (filter !== "all") r = r.filter((x) => x.family === filter);
    if (q.trim()) {
      const Q = q.trim().toLowerCase();
      r = r.filter((x) => x.exp.toLowerCase().includes(Q) || x.recipe.toLowerCase().includes(Q));
    }
    r.sort((a, b) => {
      const va = a[sort.key], vb = b[sort.key];
      if (typeof va === "number" || typeof vb === "number") {
        const av = va == null ? -Infinity : va;
        const bv = vb == null ? -Infinity : vb;
        return sort.dir === "asc" ? av - bv : bv - av;
      }
      return sort.dir === "asc"
        ? String(va).localeCompare(String(vb))
        : String(vb).localeCompare(String(va));
    });
    return r;
  }, [filter, sort, q]);

  // Show only top N unless the user expanded or actively narrowed the list.
  const isFiltered = filter !== "all" || q.trim() !== "";
  const rows = expanded || isFiltered ? sortedRows : sortedRows.slice(0, COLLAPSED_COUNT);
  const hiddenCount = Math.max(0, sortedRows.length - rows.length);
  const maxF1 = Math.max(...sortedRows.map((r) => r.public));

  const onSort = (key, sortable) => {
    if (!sortable) return;
    setSort((s) =>
      s.key === key ? { key, dir: s.dir === "asc" ? "desc" : "asc" } : { key, dir: "desc" }
    );
  };

  const bestPublic  = Math.max(...SUBMISSIONS.map((x) => x.public));
  const bestPrivate = Math.max(...SUBMISSIONS.filter((x) => x.private != null).map((x) => x.private));

  return (
    <section className="soft" id="leaderboard">
      <div className="shell">
        <div className="section-head">
          <div>
            <p className="eyebrow">Submissions</p>
            <h2>Per experiment ranking</h2>
          </div>
          <p className="lede">
            Ranked by public F1. Click any column to re sort.
            Highest public was <b>0.41826</b>; highest private was
            <b> 0.40600</b>.
          </p>
        </div>

        <div className="lb-controls">
          <div className="lb-filter" role="tablist">
            {[
              ["all",   "All"],
              ["cap",   "Cap, extra"],
              ["ours",  "010 family"],
              ["sweep", "Sweeps"],
              ["fuse",  "Fusion"],
              ["ssl",   "SSL"],
              ["inat",  "iNat"],
              ["base",  "Baselines"],
            ].map(([k, v]) => (
              <button key={k} aria-pressed={filter === k} onClick={() => setFilter(k)}>
                {v}
              </button>
            ))}
          </div>
          <input
            className="lb-search"
            placeholder="Search experiment or recipe"
            value={q}
            onChange={(e) => setQ(e.target.value)}
          />
        </div>

        <Reveal className="lb-card" variant="fade-up" style={{ overflowX: "auto" }}>
          <table className="lb-table">
            <thead>
              <tr>
                {COLS.map((c) => (
                  <th
                    key={c.key}
                    data-num={c.num ? 1 : 0}
                    onClick={() => onSort(c.key, c.sortable)}
                    style={{ cursor: c.sortable ? "pointer" : "default", width: c.width }}
                  >
                    {c.label}
                    {c.sortable && sort.key === c.key && (
                      <span className="sort">{sort.dir === "asc" ? "↑" : "↓"}</span>
                    )}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => {
                const isPublicBest  = r.public === bestPublic;
                const isPrivateBest = r.private != null && r.private === bestPrivate;
                const isBest = isPublicBest || isPrivateBest;
                return (
                  <tr key={i} data-rank={i + 1} data-ours={isBest ? 1 : 0}>
                    <td className="rank">{String(i + 1).padStart(2, "0")}</td>
                    <td>
                      <span className="badge">{r.exp}</span>
                      <span className="badge" style={{ marginLeft: 6, opacity: 0.65 }}>
                        {FAMILY_LABEL[r.family]}
                      </span>
                    </td>
                    <td>
                      {r.recipe}
                      {isPublicBest  && <span className="badge ours" style={{ marginLeft: 8 }}>best public</span>}
                      {isPrivateBest && <span className="badge ours" style={{ marginLeft: 8 }}>best private</span>}
                    </td>
                    <td data-num="1">
                      <span className="lb-bar" style={{ width: Math.round((r.public / maxF1) * 80) + "px" }} />
                      {r.public.toFixed(5)}
                    </td>
                    <td data-num="1">{r.private != null ? r.private.toFixed(5) : "not on API"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </Reveal>

        {hiddenCount > 0 && (
          <div className="lb-more">
            <button
              type="button"
              className="btn btn--outline"
              onClick={() => setExpanded(true)}
            >
              View all {sortedRows.length} submissions
              <span className="arrow">↓</span>
            </button>
          </div>
        )}
        {expanded && !isFiltered && sortedRows.length > COLLAPSED_COUNT && (
          <div className="lb-more">
            <button
              type="button"
              className="btn btn--outline"
              onClick={() => setExpanded(false)}
            >
              Show fewer
              <span className="arrow">↑</span>
            </button>
          </div>
        )}
      </div>
    </section>
  );
}
