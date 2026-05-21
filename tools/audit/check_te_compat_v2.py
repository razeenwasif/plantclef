import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling
import traceback
import sys
import os

# Mocking config and model
sys.path.append("/workspace/PlantCLEF2026")
from src.models.ensemble import PlantEnsemble

model = PlantEnsemble(num_classes=7808, input_res=672).to("cuda")
recipe = DelayedScaling()
dummy_input = torch.randn(1, 3, 672, 672).to("cuda")

print("Attempting context manager...")
try:
    with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
        print("Inside context...")
        output = model(dummy_input)
        print("Forward pass finished.")
except Exception as e:
    traceback.print_exc()
