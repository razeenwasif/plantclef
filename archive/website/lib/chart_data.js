// Chart datasets, regenerated from data/build_data.py output.
// These literals are the source of truth for the five chart components.
// The JSON copies in /public/data/ are useful for ad-hoc fetches and for
// regeneration scripts; the literals here are what gets server-rendered.

export const SCORE_PROGRESSION = [
  { id: "001",  label: "001",                this: 0.05000, best_so_far: 0.05000 },
  { id: "002",  label: "002",                this: 0.07000, best_so_far: 0.07000 },
  { id: "003",  label: "003",                this: 0.18000, best_so_far: 0.18000 },
  { id: "004",  label: "004",                this: 0.20000, best_so_far: 0.20000 },
  { id: "005",  label: "005",                this: 0.13000, best_so_far: 0.20000 },
  { id: "006",  label: "006",                this: 0.33000, best_so_far: 0.33000 },
  { id: "007",  label: "007",                this: 0.17476, best_so_far: 0.33000 },
  { id: "008",  label: "008",                this: 0.30500, best_so_far: 0.33000 },
  { id: "009",  label: "009",                this: 0.20777, best_so_far: 0.33000 },
  { id: "010",  label: "010",                this: 0.38333, best_so_far: 0.38333 },
  { id: "010f", label: "010 fusion",         this: 0.34671, best_so_far: 0.38333 },
  { id: "011",  label: "011 agg-sweep",      this: 0.38278, best_so_far: 0.38333 },
  { id: "012",  label: "012 SSL+linear",     this: 0.30261, best_so_far: 0.38333 },
  { id: "013",  label: "013 SSL+multitask",  this: 0.33185, best_so_far: 0.38333 },
  { id: "014b", label: "014b unfreeze-N",    this: 0.37455, best_so_far: 0.38333 },
  { id: "015",  label: "015 PC24+iNat",      this: 0.37956, best_so_far: 0.38333 },
  { id: "i002", label: "i002 cap+extra",     this: 0.41826, best_so_far: 0.41826 },
  { id: "i003", label: "i003 cap500",        this: 0.40041, best_so_far: 0.41826 },
];

export const UNFREEZE_SWEEP = [
  { n: 0,  label: "head only", val_top5: 0.9180, kaggle: 0.33000, note: "= 006 frozen"  },
  { n: 3,  label: "n=3",       val_top5: 0.9320, kaggle: 0.37455, note: ""              },
  { n: 4,  label: "n=4",       val_top5: 0.9323, kaggle: 0.38333, note: "anchor"        },
  { n: 5,  label: "n=5",       val_top5: 0.9327, kaggle: 0.36919, note: ""              },
  { n: 32, label: "full FT",   val_top5: 0.6800, kaggle: 0.20777, note: "009 collapse"  },
];

export const VAL_VS_KAGGLE = [
  { x: 0.7475, y: 0.38333, label: "010 last_blocks-4",     color: "ours"  },
  { x: 0.7726, y: 0.30261, label: "012 SSL + linear",      color: "ssl"   },
  { x: 0.7329, y: 0.33185, label: "013 SSL + multitask",   color: "ssl"   },
  { x: 0.7858, y: 0.37421, label: "i003 head-only",        color: "cap"   },
  { x: 0.8103, y: 0.40041, label: "i003 last_blocks-4",    color: "cap"   },
  { x: 0.8204, y: 0.38761, label: "i003 last_blocks-8",    color: "cap"   },
  { x: 0.7320, y: 0.37455, label: "014b n=3",              color: "sweep" },
  { x: 0.7475, y: 0.38333, label: "014b n=4",              color: "sweep" },
  { x: 0.7327, y: 0.36919, label: "014b n=5",              color: "sweep" },
  { x: 0.8046, y: 0.37506, label: "015 ep1",               color: "inat"  },
  { x: 0.8255, y: 0.37956, label: "015 ep5",               color: "inat"  },
  { x: 0.8103, y: 0.41826, label: "i002 ensemble ★",       color: "best"  },
];

export const E015_PER_EPOCH = [
  { epoch: 1, top1: 0.8046, top5: 0.9519, genus: 0.9149, family: 0.9490, kaggle: 0.37506 },
  { epoch: 2, top1: 0.8143, top5: 0.9549, genus: 0.9198, family: 0.9528, kaggle: null    },
  { epoch: 3, top1: 0.8207, top5: 0.9570, genus: 0.9229, family: 0.9547, kaggle: null    },
  { epoch: 4, top1: 0.8244, top5: 0.9582, genus: 0.9238, family: 0.9551, kaggle: null    },
  { epoch: 5, top1: 0.8255, top5: 0.9584, genus: 0.9240, family: 0.9553, kaggle: 0.37956 },
];

// Bin labels are the upper bound of each bin; the first two are exact counts.
// Reading: "≤5" means 3 to 5 inclusive, "≤10" means 6 to 10, and so on.
export const SPECIES_LONG_TAIL = {
  bins:       ["1", "2", "≤5", "≤10", "≤20", "≤50", "≤100", "≤250", "≤500", "≤1k", ">1k"],
  before_cap: [145, 109, 445, 349, 445, 698, 595, 1358, 1387, 2263, 12],
  after_cap:  [  8,   9,  58,  71, 138, 496, 1120, 2244, 3662,   0,   0],
};
