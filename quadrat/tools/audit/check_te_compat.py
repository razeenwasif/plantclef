import torch
from src.models.ensemble import PlantEnsemble
from src import config

# Initialize model (need to mock some config variables if they aren't loaded)
model = PlantEnsemble(num_classes=7808, input_res=672).to("cuda")

print("--- Model Layer Audit ---")
for name, module in model.named_modules():
    if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.LayerNorm)):
        print(f"Candidate: {name} ({type(module).__name__})")

print("\n--- FP8 Compatibility Test ---")
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling
recipe = DelayedScaling()
dummy_input = torch.randn(1, 3, 672, 672).to("cuda")

try:
    with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
        with torch.no_grad():
            output = model(dummy_input)
    print("SUCCESS: Model ran in FP8 autocast mode.")
except Exception as e:
    print(f"FAILURE: Model not fully FP8 compatible. Error: {e}")
