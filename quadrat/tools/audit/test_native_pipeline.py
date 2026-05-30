import torch
import numpy as np
try:
    import plantclef_ext
    HAS_EXT = True
except ImportError:
    HAS_EXT = False
    print("Warning: plantclef_ext not found. Run setup.py build_ext --inplace first.")

def test_retinex_native():
    if not HAS_EXT: return
    
    # Create dummy image
    img = torch.rand((3, 1024, 1024), device='cuda', dtype=torch.float32)
    sigmas = [15.0, 80.0, 250.0]
    
    print("[Test] Running Native Retinex...")
    processed = plantclef_ext.retinex_normalize(img, sigmas)
    
    assert processed.shape == img.shape
    assert processed.device.type == 'cuda'
    print(f"  ✓ Shape: {processed.shape}")
    print(f"  ✓ Range: [{processed.min().item():.3f}, {processed.max().item():.3f}]")

def test_inference_engine():
    if not HAS_EXT: return
    
    model_path = "models/expert_bioclip_512.pt" # Example path
    try:
        engine = plantclef_ext.NativeInferenceEngine(model_path)
        print(f"[Test] NativeInferenceEngine loaded: {model_path}")
        
        # Dummy tiling params
        x_starts = torch.tensor([0, 256, 512], dtype=torch.int32, device='cuda')
        y_starts = torch.tensor([0, 256, 512], dtype=torch.int32, device='cuda')
        
        # 1. Test from GPU tensor
        img = torch.rand((3, 1024, 1024), device='cuda', dtype=torch.float32)
        logits = engine.predict(img, x_starts, y_starts, 512, [15.0, 80.0, 250.0])
        print(f"  ✓ Prediction from Tensor: {logits.shape}")

        # 2. Test from File (Rust -> C++ -> CUDA)
        # logits = engine.predict_from_file("data/test_image.jpg", x_starts, y_starts, 512, [15.0, 80.0, 250.0])
        # print(f"  ✓ Prediction from File: {logits.shape}")
        
    except Exception as e:
        print(f"  ✗ Engine Test Failed: {e}")

if __name__ == "__main__":
    if HAS_EXT:
        test_retinex_native()
        test_inference_engine()
    else:
        print("Please build the extension to run tests.")
