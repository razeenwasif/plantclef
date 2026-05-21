import torch
import os
import json
import numpy as np

class ThermodynamicPhenology:
    """
    Atmospheric Physics & Phenology Constraint.
    
    Models the accumulated thermal energy required for a species to be visible.
    Uses circular Gaussian smoothing of GBIF month histograms to build a seasonal PDF.
    """
    def __init__(self, 
                 matrix_path: str = "models/priors/phenology_matrix.pt", 
                 histograms_path: str = "data/phenology/gbif_month_histograms.jsonl",
                 num_classes: int = 7806,
                 sigma_days: float = 18.0,
                 uniform_floor: float = 0.05):
        self.num_classes = num_classes
        self.sigma_days = sigma_days
        self.uniform_floor = uniform_floor
        
        if os.path.exists(matrix_path):
            # [C, 12] tensor where 1.0 means perfectly in-season
            self.pheno_matrix = torch.load(matrix_path, map_location="cpu", weights_only=True)
            print(f"[Phenology] Loaded pre-computed matrix from {matrix_path}")
        elif os.path.exists(histograms_path):
            print(f"[Phenology] Pre-computed matrix missing; building from {histograms_path}...")
            self.pheno_matrix = self._build_from_histograms(histograms_path)
            # Optionally save it
            os.makedirs(os.path.dirname(matrix_path), exist_ok=True)
            torch.save(self.pheno_matrix, matrix_path)
        else:
            # Fallback to neutral
            print(f"[Phenology] WARN: No phenology data found; using neutral prior.")
            self.pheno_matrix = torch.ones((num_classes, 12), dtype=torch.float32)

    def _build_from_histograms(self, jsonl_path: str) -> torch.Tensor:
        """Builds [C, 12] matrix from GBIF histograms using circular smoothing."""
        # Approximate DOY at the centre of each calendar month
        MONTH_CENTRE_DOY = np.array([15, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349], dtype=np.float32)
        DAYS = 365

        sid_to_hist = {}
        with open(jsonl_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if "error" not in rec:
                        sid_to_hist[str(rec["species_id"])] = rec.get("months", {})
                except: continue

        # We need a mapping from index to species_id. 
        # For simplicity, we assume species_ids are 0 to num_classes-1 or we load them.
        # In this codebase, species_ids are often stringified integers.
        
        matrix = torch.ones((self.num_classes, 12), dtype=torch.float32)
        
        for i in range(self.num_classes):
            sid = str(i) # This might need to be adjusted based on actual mapping
            h = sid_to_hist.get(sid)
            if not h:
                continue
            
            counts = np.zeros(12, dtype=np.float32)
            for k, v in h.items():
                counts[int(k) - 1] = v
            
            # Smooth
            pdf = np.zeros(12, dtype=np.float32)
            for m in range(12):
                c = counts[m]
                if c <= 0: continue
                for target_m in range(12):
                    d = np.abs(MONTH_CENTRE_DOY[target_m] - MONTH_CENTRE_DOY[m])
                    d = np.minimum(d, DAYS - d)
                    pdf[target_m] += c * np.exp(-0.5 * (d / self.sigma_days) ** 2)
            
            if pdf.sum() > 0:
                pdf /= pdf.sum()
                # Blend with uniform floor
                pdf = (1 - self.uniform_floor) * pdf + self.uniform_floor / 12
                pdf /= pdf.sum()
                matrix[i] = torch.from_numpy(pdf)
        
        # Normalize each row to max 1.0 for "thermal efficiency"
        max_vals, _ = matrix.max(dim=1, keepdim=True)
        matrix /= max_vals.clamp(min=1e-9)
        
        return matrix

    def apply_thermal_penalty(self, logits: torch.Tensor, current_month: int, beta: float = 1.0) -> torch.Tensor:
        """
        Applies a thermodynamic penalty to the logits.
        
        log p_final = log p_visual + beta * log P(month | species)
        """
        device = logits.device
        m_idx = max(0, min(11, current_month - 1))
        
        # efficiency in [0, 1]
        efficiency = self.pheno_matrix[:, m_idx].to(device)
        
        # log prior (multiplicative in probability space)
        # We use log(efficiency + eps) as the penalty.
        # If efficiency=1.0, penalty=0.
        # If efficiency=0.05, penalty=log(0.05) ~= -3.0.
        penalty = beta * torch.log(efficiency + 1e-9)
        
        return logits + penalty
