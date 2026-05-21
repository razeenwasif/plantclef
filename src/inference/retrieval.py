import torch
import numpy as np
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

class RetrievalEngine:
    """
    ORACLE SOTA: Retrieval-Augmented Classification.
    Uses FAISS to boost tail species recall via prototype matching.
    """
    def __init__(self, prototype_path: str, device: str = "cuda"):
        if not HAS_FAISS:
            self.index = None
            return

        print(f"[RAC] Loading species prototypes from {prototype_path}...")
        # prototypes: [7806, 6656]
        self.prototypes = torch.load(prototype_path).cpu().numpy().astype('float32')
        
        # Build L2 index
        self.d = self.prototypes.shape[1]
        self.index = faiss.IndexFlatIP(self.d) # Inner product for cosine similarity
        
        # Normalize for cosine similarity
        faiss.normalize_L2(self.prototypes)
        self.index.add(self.prototypes)
        
        if device == "cuda":
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

    def boost_logits(self, query_features: np.ndarray, logits: np.ndarray, alpha: float = 0.3) -> np.ndarray:
        """
        Fused Score = alpha * Logit + (1-alpha) * Retrieval_Similarity
        """
        if self.index is None:
            return logits

        # Normalize query
        faiss.normalize_L2(query_features)
        
        # Search all 7806 species
        D, I = self.index.search(query_features, 7806)
        
        # Re-map distances to original indices
        retrieval_scores = np.zeros_like(logits)
        for i in range(len(query_features)):
            retrieval_scores[i, I[i]] = D[i]

        return (alpha * logits) + ((1.0 - alpha) * retrieval_scores)
