import torch
import numpy as np

class GeochemMasker:
    """
    Biogeochemistry & Edaphic Masking.
    
    Filters species based on soil chemistry (pH, Oraclegen, Carbon).
    Plants are chemically bound to their substrate; calcifuges (acid-loving)
    cannot survive in basic limestone soils.
    """
    def __init__(self, num_classes: int = 7806):
        self.num_classes = num_classes
        # Simulated Soil Preference Database
        # 0: Acidic-loving, 1: Neutral, 2: Basic/Alkaline
        np.random.seed(42)
        self.soil_prefs = torch.from_numpy(np.random.randint(0, 3, size=(num_classes,)))

    def apply_mask(self, logits: torch.Tensor, local_ph: float = 7.0) -> torch.Tensor:
        """
        Applies a chemical substrate mask to the logits.
        
        Parameters
        ----------
        logits : torch.Tensor
            (C,) raw model logits.
        local_ph : float
            The soil pH at the quadrat location (simulated or from SoilGrids).
        """
        device = logits.device
        prefs = self.soil_prefs.to(device)
        
        # Heuristic: 
        # If pH < 5.5 (Acidic), penalize Basic-loving plants (pref=2)
        # If pH > 7.5 (Alkaline), penalize Acid-loving plants (pref=0)
        mask = torch.ones_like(logits)
        
        if local_ph < 5.5:
            mask[prefs == 2] = 0.1 # Severe penalty
        elif local_ph > 7.5:
            mask[prefs == 0] = 0.1
            
        # Apply as a logit penalty
        penalty = torch.log(mask + 1e-9)
        return logits + penalty
