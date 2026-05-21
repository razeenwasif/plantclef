import unittest
import torch
import torch.nn.functional as F
import numpy as np
import os
import tempfile

try:
    import plantclef_ext
    HAS_EXT = True
except ImportError:
    HAS_EXT = False

try:
    import data_auditor
    HAS_AUDITOR = True
except ImportError:
    HAS_AUDITOR = False

class TestCUDAExtensions(unittest.TestCase):
    @unittest.skipIf(not HAS_EXT, "plantclef_ext not installed")
    def test_fused_gfam_projection(self):
        batch_size = 4
        d_bio = 768
        d_dino = 1024
        d_conv = 1536
        fusion_dim = 1536
        total_dim = d_bio + d_dino + d_conv

        feat_bio = torch.randn(batch_size, d_bio, device='cuda')
        feat_dino = torch.randn(batch_size, d_dino, device='cuda')
        feat_conv = torch.randn(batch_size, d_conv, device='cuda')
        gating = torch.softmax(torch.randn(batch_size, 3, device='cuda'), dim=1)

        weight = torch.randn(fusion_dim, total_dim, device='cuda')
        bias = torch.randn(fusion_dim, device='cuda')
        agg_w = torch.ones(fusion_dim, device='cuda')
        agg_b = torch.zeros(fusion_dim, device='cuda')
        ln_gamma = torch.ones(fusion_dim, device='cuda')
        ln_beta = torch.zeros(fusion_dim, device='cuda')

        gfam_outputs = plantclef_ext.fused_gfam_projection(
            feat_bio, feat_dino, feat_conv, gating,
            weight, bias, agg_w, agg_b, ln_gamma, ln_beta
        )
        
        output = gfam_outputs[0]
        self.assertEqual(output.shape, (batch_size, fusion_dim))
        self.assertFalse(torch.isnan(output).any())

    @unittest.skipIf(not HAS_EXT, "plantclef_ext not installed")
    def test_fused_loss(self):
        batch_size = 4
        num_classes = 100
        logits = torch.randn(batch_size, num_classes, device='cuda')
        targets = torch.rand(batch_size, num_classes, device='cuda')
        logit_adjustments = torch.zeros(num_classes, device='cuda')
        genus_ids = torch.arange(num_classes, device='cuda', dtype=torch.int32)
        gamma_neg_tensor = torch.ones(num_classes, device='cuda') * 4.0
        target_genus_ids = torch.zeros(batch_size, device='cuda', dtype=torch.int32)

        losses = plantclef_ext.fused_asl_forward(
            logits, targets, logit_adjustments, genus_ids, gamma_neg_tensor,
            1.0, 0.05, 1e-8, 0.1, target_genus_ids
        )
        self.assertEqual(losses[0].shape, (batch_size,))
        
    @unittest.skipIf(not HAS_EXT, "plantclef_ext not installed")
    def test_fused_gcn(self):
        num_species = 100
        trait_dim = 19
        out_dim = 1536
        
        adj = torch.eye(num_species, device='cuda')
        traits = torch.randn(num_species, trait_dim, device='cuda')
        theta = torch.randn(trait_dim, out_dim, device='cuda')
        
        out_weights = plantclef_ext.fused_gcn_forward(adj, traits, theta)
        self.assertEqual(out_weights.shape, (num_species, out_dim))

class TestDataAuditor(unittest.TestCase):
    @unittest.skipIf(not HAS_AUDITOR, "data_auditor not installed")
    def test_audit_dataset(self):
        # Create some temporary files, one small, one large
        with tempfile.TemporaryDirectory() as tmpdir:
            file_good = os.path.join(tmpdir, "good.jpg")
            file_bad = os.path.join(tmpdir, "bad.jpg")
            
            with open(file_good, "wb") as f:
                f.write(b"0" * 2048) # 2KB
                
            with open(file_bad, "wb") as f:
                f.write(b"0" * 500) # 500B
                
            paths = [file_good, file_bad, os.path.join(tmpdir, "nonexistent.jpg")]
            
            valid_indices = data_auditor.audit_dataset(paths, False)
            
            # Only the first file (good.jpg) should be valid (>1024 bytes)
            self.assertEqual(valid_indices, [0])

    @unittest.skipIf(not HAS_AUDITOR, "data_auditor not installed")
    def test_taxonomic_filter(self):
        import numpy as np
        allowed_neighbors = [[0, 1], [0, 1, 2], [1, 2]]
        # TaxonomicFilter now requires species→genus (list[int]), genus→family (list[int]),
        # and bioclim (dict) — matching the actual JSON data format
        s2g = [0, 0, 1]      # species_idx → genus_idx
        g2f = [0, 0]          # genus_idx  → family_idx
        bioclim = [list(range(3))]  # list[list[int]], one entry per bioclim zone
        tf = data_auditor.TaxonomicFilter(allowed_neighbors, s2g, g2f, bioclim)
        
        # Test class 0 (anchor)
        preds = np.array([0.8, 0.1, 0.4], dtype=np.float32)
        filtered = tf.filter_predictions(preds)
        # Class 2 should be filtered out (0.0) since allowed_neighbors[0] is [0, 1]
        self.assertEqual(filtered[2], 0.0)
        self.assertEqual(filtered[0], 0.8)

if __name__ == '__main__':
    unittest.main()
