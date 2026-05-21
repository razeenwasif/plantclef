import torch
import numpy as np
import pytest
import json
from pathlib import Path
from unittest.mock import MagicMock
from src.inference.pav_solver import PAVTreeSolver
from src.inference.frank_wolfe import FrankWolfeSolver
from src.inference.model_runner import ModelRunner, _retinex_normalize, _TileDataset
from src.inference.config import ModelConfig
from PIL import Image

# --- PAV-TREE TESTS ---

def test_pav_tree_sanity(tmp_path):
    """Verify that PAV-Tree solver initializes and returns valid shapes."""
    tree = {
        "name": "Plantae",
        "children": {
            "Order_1": {
                "children": {
                    "Family_1": {
                        "children": {
                            "Genus_A": {
                                "children": {
                                    "Species_0": {"idx": 0, "weight": 1.0}
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    tree_path = tmp_path / "mock_tree.json"
    tree_path.write_text(json.dumps(tree))
    
    solver = PAVTreeSolver(str(tree_path))
    probs = torch.zeros(7808)
    probs[0] = 0.9
    
    calibrated = solver.solve(probs)
    assert calibrated.shape[0] == 7808
    assert isinstance(calibrated, torch.Tensor)

# --- FRANK-WOLFE TESTS ---

def test_frank_wolfe_consistency(tmp_path):
    """Verify that Frank-Wolfe respects co-occurrence constraints."""
    # Adj: Species 0 is only compatible with 0 and 2.
    adj = [[0, 2], [1, 2], [0, 1, 2], [3], [4]]
    
    adj_path = tmp_path / "mock_adj.json"
    adj_path.write_text(json.dumps(adj))
    
    solver = FrankWolfeSolver(str(adj_path))
    logits = torch.tensor([10.0, 9.0, 1.0, 0.0, 0.0])
    
    # Target k=2. Species 0 and 1 have highest logits but are incompatible.
    y_opt = solver.solve(logits, max_iters=5, sparsity_k=2)
    indices = torch.nonzero(y_opt > 1e-3).squeeze().tolist()
    
    assert 0 in indices
    assert 1 not in indices # Constraint enforced

# --- SUBMODULAR SELECTION TESTS ---

class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(10, 7808)
    def forward(self, x):
        return self.proj(torch.randn(x.size(0), 10, device=x.device))

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_submodular_selection():
    """Verify that submodular selection reduces tile count."""
    from src.inference.types import TileSpec
    from unittest.mock import MagicMock
    
    # 1. Setup Mock Config
    m_cfg = ModelConfig(num_classes=7808)
    
    # 2. Setup Adapter Mock
    adapter = MagicMock()
    m_model = MockModel().cuda()
    adapter.get_model.return_value = m_model
    adapter.num_classes = 7808
    adapter.input_size = 224
    
    # 3. Setup Runner
    runner = ModelRunner(adapter, m_cfg)
    # Patch internal stats to avoid real processing
    runner.inference_batch = 1 
    runner.input_size = 224
    
    # 4. Create 5 mock tiles
    tiles = []
    for i in range(5):
        spec = TileSpec(x0=0, y0=0, x1=10, y1=10, image_id="test", tile_id=i)
        # Create a tiny 1x1 image to speed up test
        img = Image.new('RGB', (224, 224))
        tiles.append((spec, img))
        
    # 5. Select top 2
    selected = runner.select_informative_tiles(tiles, k=2, scan_res=224)
    assert len(selected) == 2
    assert isinstance(selected[0][0], TileSpec)

# --- RETINEX TESTS ---

def test_retinex_output_range():
    """Retinex output must stay in [0, 1] and match input shape."""
    pytest.importorskip("scipy", reason="scipy required for Retinex")
    rng = np.random.default_rng(42)
    arr = rng.random((64, 64, 3), dtype=np.float32)
    out = _retinex_normalize(arr, sigmas=(5.0, 20.0, 60.0))
    assert out.shape == arr.shape
    assert out.dtype == np.float32
    assert out.min() >= 0.0 - 1e-6
    assert out.max() <= 1.0 + 1e-6


def test_retinex_removes_uniform_illumination():
    """A uniformly-lit patch should collapse toward a near-constant reflectance."""
    pytest.importorskip("scipy", reason="scipy required for Retinex")
    # Bright patch: channel 0 entirely at 0.9, channels 1/2 at 0.5
    arr = np.zeros((32, 32, 3), dtype=np.float32)
    arr[:, :, 0] = 0.9
    arr[:, :, 1] = 0.5
    arr[:, :, 2] = 0.5
    out = _retinex_normalize(arr, sigmas=(5.0, 20.0, 60.0))
    # Uniform illumination → Retinex should produce very low spatial variance
    assert out[:, :, 0].std() < 0.05


def test_retinex_tile_dataset_integration():
    """_TileDataset._preprocess with use_retinex=True produces valid tensors."""
    pytest.importorskip("scipy", reason="scipy required for Retinex")
    from src.inference.types import TileSpec
    spec = TileSpec(x0=0, y0=0, x1=64, y1=64, image_id="test", tile_id=0)
    img = Image.fromarray(
        np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    )
    ds = _TileDataset(
        tiles=[(spec, img)],
        input_size=64,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        use_retinex=True,
        retinex_sigmas=(5.0, 20.0, 60.0),
    )
    tensor, idx = ds[0]
    assert tensor.shape == (3, 64, 64)
    assert idx == 0
    assert torch.isfinite(tensor).all()


# --- FRANK-WOLFE ISLAND BIOGEOGRAPHY PRIOR TESTS ---

def test_fw_prior_reduces_species_count(tmp_path):
    """Island biogeography prior should suppress over-prediction."""
    # All-connected adjacency (no ecological constraints)
    n = 10
    adj = [list(range(n)) for _ in range(n)]
    adj_path = tmp_path / "adj.json"
    adj_path.write_text(json.dumps(adj))

    # High logits for ALL 10 species — without prior, FW selects sparsity_k=8
    logits = torch.ones(n) * 5.0

    # Without prior: all k species selected at the vertex
    solver_no_prior = FrankWolfeSolver(str(adj_path), count_lambda=0.0, species_count_k=3.0)
    y_no_prior = solver_no_prior.solve(logits, max_iters=20, sparsity_k=8)

    # With strong prior at K=3: effective gradient penalises mass above 3
    solver_prior = FrankWolfeSolver(str(adj_path), count_lambda=1.0, species_count_k=3.0)
    y_prior = solver_prior.solve(logits, max_iters=20, sparsity_k=8)

    # The prior should reduce the total probability mass
    assert y_prior.sum() <= y_no_prior.sum()


def test_fw_prior_zero_is_baseline(tmp_path):
    """count_lambda=0 must produce identical results to the original solver."""
    adj = [[0, 1, 2], [0, 1, 2], [0, 1, 2]]
    adj_path = tmp_path / "adj.json"
    adj_path.write_text(json.dumps(adj))

    logits = torch.tensor([3.0, 1.0, 0.5])

    solver_ref = FrankWolfeSolver(str(adj_path), count_lambda=0.0)
    solver_zero = FrankWolfeSolver(str(adj_path), count_lambda=0.0, species_count_k=8.0)

    y_ref = solver_ref.solve(logits, max_iters=10, sparsity_k=2)
    y_zero = solver_zero.solve(logits, max_iters=10, sparsity_k=2)

    assert torch.allclose(y_ref, y_zero, atol=1e-6)


# --- PHYLOGENETIC ADJACENCY BUILDER TESTS ---

def test_build_phylo_adj_script(tmp_path):
    """build_phylo_adj produces a valid row-normalised adjacency."""
    import importlib.util, types

    # Build minimal mock taxonomy: 4 species, 2 genera, 1 family
    s2g = [0, 0, 1, 1]   # species→genus
    g2f = [0, 0]          # genus→family

    (tmp_path / "species_to_genus.json").write_text(json.dumps(s2g))
    (tmp_path / "genus_to_family.json").write_text(json.dumps(g2f))

    # Import and call the builder function directly
    spec = importlib.util.spec_from_file_location(
        "build_phylo_adj",
        Path(__file__).parents[1] / "tools/data/build_phylo_adj.py",
    )
    mod = importlib.util.module_from_spec(spec)

    # Patch paths to use tmp_path
    import tools.data.build_phylo_adj as bpa
    orig_s2g = bpa.S2G_PATH
    orig_g2f = bpa.G2F_PATH
    orig_out = bpa.OUT_PATH
    orig_eco = bpa.ECO_ADJ_PATH

    bpa.S2G_PATH = tmp_path / "species_to_genus.json"
    bpa.G2F_PATH = tmp_path / "genus_to_family.json"
    bpa.OUT_PATH = tmp_path / "phylo_adj.npy"
    bpa.ECO_ADJ_PATH = tmp_path / "does_not_exist.npy"  # force n=4

    try:
        adj = bpa.build_phylo_adj(n_species=4)
    finally:
        bpa.S2G_PATH = orig_s2g
        bpa.G2F_PATH = orig_g2f
        bpa.OUT_PATH = orig_out
        bpa.ECO_ADJ_PATH = orig_eco

    assert adj.shape == (4, 4)
    assert adj.dtype == np.float32
    # Row sums should be ~1.0 (row-normalised)
    np.testing.assert_allclose(adj.sum(axis=1), np.ones(4), atol=1e-5)
    # Species 0 and 1 share genus 0 → should have highest weight pair
    assert adj[0, 1] > adj[0, 2]  # same genus > same family


if __name__ == "__main__":
    pytest.main([__file__])
