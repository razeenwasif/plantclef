import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image

class TestTimeAdaptor:
    """
    PLANTCLEF SOTA: Test-Time Training (TTT).
    Briefly adapts the model to each test quadrat using self-supervision.
    """
    def __init__(self, model: nn.Module, steps: int = 10, lr: float = 1e-5):
        self.model = model
        self.steps = steps
        self.lr = lr
        
    def adapt(self, image: Image.Image, image_id: str = ""):
        """
        Run 10-20 gradient steps on a self-supervised task (Rotation Prediction).
        """
        # plantclef: Detect device and resolution from the model
        device = next(self.model.parameters()).device
        res = getattr(self.model, "input_res", 224)

        # 1. Prepare 4 rotations
        tf = transforms.Compose([
            transforms.Resize((res, res)),
            transforms.ToTensor(),
        ])
        
        img_t = tf(image).unsqueeze(0).to(device)
        rotations = [
            img_t,
            torch.rot90(img_t, 1, [2, 3]),
            torch.rot90(img_t, 2, [2, 3]),
            torch.rot90(img_t, 3, [2, 3])
        ]
        batch = torch.cat(rotations, dim=0) # [4, 3, res, res]
        labels = torch.tensor([0, 1, 2, 3]).to(device)
        
        # 2. Setup transient optimizer
        # plantclef: Support both Ensemble and Legacy (best.pt) layers
        trainable_params = []
        target_layers = ["proj_linear", "gating_network", "species_head", "head"]
        for name, param in self.model.named_parameters():
            if any(t in name for t in target_layers):
                param.requires_grad = True
                trainable_params.append(param)
        
        if not trainable_params: return 0.0 # Skip if no layers found
        
        optimizer = optim.Adam(trainable_params, lr=self.lr)
        self.model.train()
        
        # 3. Adaptation Loop
        initial_loss = 0.0
        final_loss = 0.0
        with torch.enable_grad():
            for i in range(self.steps):
                optimizer.zero_grad()
                outputs = self.model(batch)
                if isinstance(outputs, tuple): outputs = outputs[0]
                
                # Simple rotation loss proxy (using first 4 outputs as rotation classes)
                loss = nn.CrossEntropyLoss()(outputs[:, :4], labels) 
                if i == 0: initial_loss = loss.item()
                loss.backward()
                optimizer.step()
            final_loss = loss.item()
            
        self.model.eval()
        for p in trainable_params:
            p.requires_grad = False
            
        # Only print periodically to avoid spam
        rank = int(os.environ.get("RANK", 0))
        if rank == 0:
            # We use a simple hash of image_id to print every ~20 images
            if sum(ord(c) for c in image_id) % 20 == 0:
                print(f" [TTT] {image_id} | RotLoss: {initial_loss:.4f} -> {final_loss:.4f}")
            return final_loss
        return 0.0
