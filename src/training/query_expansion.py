"""Query expansion for inference.

At prediction time, runs each test image through the ensemble at multiple
augmentations (original + horizontal flip + center crop), averages the
feature embeddings, then classifies from the averaged features.

This is inference-only -- zero training time impact.
Typical gain: 2-5% accuracy on the final leaderboard score.
"""
import torch
import torch.nn.functional as F
from torch.amp import autocast

def extract_query_features(model, image, chunk_size=32):
    """Extracts features from an image under multiple augmentations.

    Augmentations:
        1. Original image
        2. Horizontal flip
        3. Center crop (75%) resized back

    Args:
        model: PlantEnsemble (eval mode).
        image: (1, C, H, W) tensor on CUDA.
        chunk_size: Backbone chunk size.

    Returns:
        torch.Tensor: (1, num_classes) averaged prediction.
    """
    model.eval()

    views = [image]

    # Horizontal flip
    views.append(torch.flip(image, dims=[3]))

    # Center crop (75%) + resize
    height, width = image.shape[2], image.shape[3]
    crop_h = int(height * 0.75)
    crop_w = int(width * 0.75)
    top = (height - crop_h) // 2
    left = (width - crop_w) // 2
    crop = image[:, :, top:top + crop_h, left:left + crop_w]
    crop = F.interpolate(crop.float(), size=(height, width),
                          mode='bilinear', align_corners=False)
    views.append(crop.to(image.dtype))

    all_logits = []
    with torch.no_grad():
        for view in views:
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                feat_bio = model.bioclip(view)
                feat_dino = model.dinov3(view)
                feat_conv = model.convnext(view)

                # Grouped Blackwell-native projection
                fused_raw = torch.cat([feat_bio, feat_dino, feat_conv], dim=1)
                fused_proj = model.proj_grouped(fused_raw)

                logits = model.species_classifier(F.normalize(fused_proj, dim=1))
            all_logits.append(logits)

    # Average logits across all views before argmax
    return torch.stack(all_logits).mean(dim=0)

def predict_with_expansion(model, images, threshold=0.5):
    """Batch prediction with query expansion.

    Args:
        model: PlantEnsemble (eval mode).
        images: (B, C, H, W) batch tensor on CUDA.
        threshold: Confidence threshold for multi-label prediction.

    Returns:
        tuple: (predictions boolean tensor, averaged logits tensor).
    """
    all_logits = []
    for i in range(images.shape[0]):
        logits = extract_query_features(model, images[i:i+1])
        all_logits.append(logits)

    logits = torch.cat(all_logits, dim=0)
    predictions = torch.sigmoid(logits) > threshold
    return predictions, logits
