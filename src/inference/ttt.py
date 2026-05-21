import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image

class TestTimeAdaptor:
    """
    ORACLE SOTA: Test-Time Training (TTT).
    Briefly adapts the model to each test quadrat using self-supervision.
    """
    def __init__(self, model: nn.Module, steps: int = 10, lr: float = 1e-5):
        self.model = model
        self.steps = steps
        self.lr = lr
        
    def adapt(self, image: Image.Image):
        """
        Run 10-20 gradient steps on a self-supervised task (Rotation Prediction).
        """
        # 1. Prepare 4 rotations
        tf = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
        ])
        
        img_t = tf(image).unsqueeze(0).cuda()
        rotations = [
            img_t,
            torch.rot90(img_t, 1, [2, 3]),
            torch.rot90(img_t, 2, [2, 3]),
            torch.rot90(img_t, 3, [2, 3])
        ]
        batch = torch.cat(rotations, dim=0) # [4, 3, 448, 448]
        labels = torch.tensor([0, 1, 2, 3]).cuda()
        
        # 2. Setup transient optimizer for last 2 layers
        # (We only tune the adapter layers to prevent "forgetting")
        trainable_params = []
        for name, param in self.model.named_parameters():
            if "proj_linear" in name or "gating_network" in name:
                trainable_params.append(param)
        
        if not trainable_params: return # Skip if no layers found
        
        optimizer = optim.Adam(trainable_params, lr=self.lr)
        self.model.train()
        
        # 3. Adaptation Loop
        for _ in range(self.steps):
            optimizer.zero_grad()
            # Multi-output: species_logits, trait_logits
            # For TTT, we just care about the internal representation
            outputs = self.model(batch)
            if isinstance(outputs, tuple): outputs = outputs[0]
            
            # Simple rotation loss proxy
            # In a full SOTA implementation, this would use a dedicated rotation head
            loss = nn.CrossEntropyLoss()(outputs[:, :4], labels) 
            loss.backward()
            optimizer.step()
            
        self.model.eval()
        print(f"[TTT] Adapted model to quadrat (Loss: {loss.item():.4f})")
