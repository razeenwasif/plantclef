"""
Core few-shot scoring: prototype mode and KNN mode.

SupportBank
-----------
Holds per-species support embeddings and pre-computed L2-normalised prototypes.
Loaded from a cached .pt file built by build_support_bank.py.

Scoring functions
-----------------
score_prototype  : cosine similarity between tile and per-species mean embedding
score_knn        : aggregated similarity to all support embeddings per species

Both functions return a (n_tiles, n_species) tensor of cosine-similarity scores
(values in [-1, 1], higher = more similar).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Support bank
# ---------------------------------------------------------------------------

class SupportBank:
    """
    Container for support-set embeddings and species prototypes.

    Attributes
    ----------
    species_ids     : ordered list of species ID strings (length n_species)
    prototypes      : (n_species, dim) L2-normalised mean embeddings, on CPU
    support_counts  : dict mapping species_id -> number of support images used
    """

    def __init__(
        self,
        species_ids: list[str],
        embeddings_by_species: dict[str, torch.Tensor],
    ) -> None:
        """
        Parameters
        ----------
        species_ids             : ordered species ID list
        embeddings_by_species   : dict species_id -> (k_i, dim) tensor, L2-normalised
        """
        if not species_ids:
            raise ValueError("SupportBank: species_ids must be non-empty")

        missing = [sid for sid in species_ids if sid not in embeddings_by_species]
        if missing:
            raise ValueError(
                f"SupportBank: {len(missing)} species_ids have no embeddings "
                f"(e.g. {missing[:5]})"
            )

        self.species_ids = species_ids
        self._emb: dict[str, torch.Tensor] = {
            sid: embeddings_by_species[sid].cpu() for sid in species_ids
        }
        self.support_counts: dict[str, int] = {
            sid: self._emb[sid].shape[0] for sid in species_ids
        }
        self.prototypes: torch.Tensor = self._compute_prototypes()

        logger.info(
            f"SupportBank ready: {len(species_ids)} species, "
            f"embed_dim={self.prototypes.shape[1]}, "
            f"total_support={sum(self.support_counts.values()):,}"
        )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _compute_prototypes(self) -> torch.Tensor:
        """Average support embeddings per species and L2-normalise."""
        protos: list[torch.Tensor] = []
        for sid in self.species_ids:
            emb = self._emb[sid]          # (k_i, dim)
            proto = emb.mean(dim=0)        # (dim,)
            proto = F.normalize(proto, dim=0)
            protos.append(proto)
        return torch.stack(protos, dim=0)  # (n_species, dim)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save bank to a .pt file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "species_ids": self.species_ids,
                "embeddings_by_species": self._emb,
                "prototypes": self.prototypes,
            },
            path,
        )
        logger.info(f"SupportBank saved: {path}")

    @classmethod
    def load(cls, path: str | Path) -> "SupportBank":
        """Load a bank previously saved with .save()."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Support bank file not found: {path}")
        data = torch.load(path, map_location="cpu", weights_only=False)
        bank = cls.__new__(cls)
        bank.species_ids = data["species_ids"]
        bank._emb = data["embeddings_by_species"]
        bank.support_counts = {sid: bank._emb[sid].shape[0] for sid in bank.species_ids}
        # Use pre-computed prototypes if present (avoids recomputation)
        if "prototypes" in data:
            bank.prototypes = data["prototypes"]
        else:
            bank.prototypes = bank._compute_prototypes()
        logger.info(
            f"SupportBank loaded from {path}: "
            f"{len(bank.species_ids)} species"
        )
        return bank

    # ------------------------------------------------------------------
    # Access helpers
    # ------------------------------------------------------------------

    def all_embeddings(self) -> tuple[torch.Tensor, list[int]]:
        """
        Return all support embeddings concatenated and a species-index list.

        Returns
        -------
        flat_emb    : (total_support, dim) tensor on CPU
        species_idx : list[int], length total_support - maps each row to a
                      position in self.species_ids
        """
        emb_list: list[torch.Tensor] = []
        species_idx: list[int] = []
        for si, sid in enumerate(self.species_ids):
            emb = self._emb[sid]
            emb_list.append(emb)
            species_idx.extend([si] * emb.shape[0])
        return torch.cat(emb_list, dim=0), species_idx


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def score_prototype(
    query_emb: torch.Tensor,
    bank: SupportBank,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Prototype-mode scoring.

    Compare each query tile embedding to the per-species mean (prototype)
    using cosine similarity.  Both query and prototypes are assumed to be
    L2-normalised so dot-product == cosine similarity.

    Parameters
    ----------
    query_emb : (n_tiles, dim) L2-normalised tile embeddings
    bank      : SupportBank with pre-computed prototypes
    device    : torch device string for the matrix multiply

    Returns
    -------
    (n_tiles, n_species) float32 tensor on CPU, values in [-1, 1]
    """
    protos = bank.prototypes.to(device)  # (n_species, dim)
    q = query_emb.to(device)             # (n_tiles, dim)
    sim = q @ protos.T                   # (n_tiles, n_species)
    return sim.cpu()


def score_knn(
    query_emb: torch.Tensor,
    bank: SupportBank,
    device: str = "cpu",
) -> torch.Tensor:
    """
    KNN-mode scoring.

    Compute cosine similarity between each tile and every support embedding,
    then aggregate per species by summing similarities and normalising by
    the number of support images per species (to avoid biasing towards species
    with more support images).

    Parameters
    ----------
    query_emb   : (n_tiles, dim) L2-normalised tile embeddings
    bank        : SupportBank
    device      : torch device string

    Returns
    -------
    (n_tiles, n_species) float32 tensor on CPU, values in [-1, 1]
    """
    flat_emb, species_idx = bank.all_embeddings()
    flat_emb = flat_emb.to(device)      # (total_support, dim)
    q = query_emb.to(device)            # (n_tiles, dim)

    sim = q @ flat_emb.T                # (n_tiles, total_support)

    n_species = len(bank.species_ids)
    n_tiles = q.shape[0]

    # Build index tensor for scatter_add
    idx_tensor = torch.tensor(species_idx, dtype=torch.long, device=device)  # (total_support,)
    idx_expanded = idx_tensor.unsqueeze(0).expand(n_tiles, -1)               # (n_tiles, total_support)

    # Sum similarities per species per tile
    species_scores = torch.zeros(n_tiles, n_species, device=device)
    species_scores.scatter_add_(1, idx_expanded, sim)

    # Normalise by number of support images per species
    species_counts = torch.zeros(n_species, device=device)
    species_counts.scatter_add_(
        0, idx_tensor, torch.ones(len(species_idx), device=device)
    )
    species_scores = species_scores / species_counts.unsqueeze(0).clamp(min=1.0)

    return species_scores.cpu()


# ---------------------------------------------------------------------------
# Optional text-fusion
# ---------------------------------------------------------------------------

def fuse_scores(
    image_scores: torch.Tensor,
    text_scores: Optional[torch.Tensor],
    alpha: float = 1.0,
) -> torch.Tensor:
    """
    Fuse image few-shot scores with optional text-prompt scores.

    Final score = alpha * image_scores + (1 - alpha) * text_scores

    If text_scores is None or alpha == 1.0, image_scores is returned as-is.

    Parameters
    ----------
    image_scores : (n_tiles, n_species) image few-shot similarity scores
    text_scores  : (n_tiles, n_species) text zero-shot similarity scores, or None
    alpha        : weight for image component (0.0 = text-only, 1.0 = image-only)

    Returns
    -------
    (n_tiles, n_species) fused scores
    """
    if text_scores is None or alpha >= 1.0:
        return image_scores
    if alpha <= 0.0:
        return text_scores
    return alpha * image_scores + (1.0 - alpha) * text_scores
