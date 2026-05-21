import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image

class TestTimeAdaptor:
    """
    ORACLE SOTA: Test-Time Training (TTT) via Entropy Minimization.
    Adapts the model's confidence for each quadrat by minimizing the 
    prediction entropy across all tiles.
    """
    def __init__(self, model: nn.Module, steps: int = 5, lr: float = 1e-5):
        self.model = model
        self.steps = steps
        self.lr = lr
        
        # ORACLE: Episodic Memory Backup
        # We must backup the original weights of the layers we intend to adapt.
        # Otherwise, the model will accumulate adaptations across thousands of images 
        # and suffer catastrophic forgetting (Entropy collapse to 0.0000).
        self.target_layers = ["proj_ln", "gating_network", "agg_weights", "species_head", "head"]
        self.base_state = {}
        for name, param in self.model.named_parameters():
            if any(t in name for t in self.target_layers):
                self.base_state[name] = param.data.clone().detach()
        
    def adapt(self, tiles: torch.Tensor, image_id: str = ""):
        """
        Run entropy minimization on a batch of tiles.
        """
        device = next(self.model.parameters()).device

        # 1. Restore Base State (wipe short-term memory from previous image)
        for name, param in self.model.named_parameters():
            if name in self.base_state:
                param.data.copy_(self.base_state[name])

        # 2. Identify parameters to adapt.
        # Note: model is intentionally kept in eval() mode — calling train() would
        # re-enable BioCLIP's DropPath(0.1) and make every forward pass stochastic,
        # which masks any real entropy-reduction signal.
        trainable_params = []
        for name, param in self.model.named_parameters():
            if any(t in name for t in self.target_layers):
                param.requires_grad = True
                trainable_params.append(param)

        if not trainable_params: return 0.0

        optimizer = optim.Adam(trainable_params, lr=self.lr)

        # 3. Adaptation Loop: Minimize Shannon Entropy
        initial_entropy = 0.0
        final_entropy = 0.0
        with torch.enable_grad():
            for i in range(self.steps):
                optimizer.zero_grad()
                logits = self.model(tiles.to(device))
                if isinstance(logits, tuple): logits = logits[0]

                probs = torch.softmax(logits, dim=1)
                entropy = -(probs * torch.log(probs + 1e-9)).sum(1).mean()
                if i == 0: initial_entropy = entropy.item()

                entropy.backward()
                optimizer.step()
            final_entropy = entropy.item()

        for p in trainable_params:
            p.requires_grad = False
            
        # We DO NOT restore base_state here!
        # The adapted weights must be preserved for the forward pass in _infer_one_image
        # which happens immediately after this function returns. The weights will be 
        # reset at the START of the next call to adapt().
            
        # Only print periodically to avoid spam
        rank = int(os.environ.get("RANK", 0))
        if rank == 0:
            if sum(ord(c) for c in image_id) % 20 == 0:
                print(f" [TTT-SOTA] {image_id} | Entropy: {initial_entropy:.4f} -> {final_entropy:.4f}")
            return final_entropy
        return 0.0
