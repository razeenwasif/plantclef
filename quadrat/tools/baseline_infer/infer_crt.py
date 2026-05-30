"""Tile-based inference for cRT (Classifier-Retraining) heads.

Architecture:
    PlantBioCLIP-2 ViT-B (768)  ─┐
    PlantViTBackbone DINOv3-L (1024) ┼─► concat 3328 (NO L2 normalize) ─► cRT MLP head ─► 7808 logits ─► slice to 7806
    PlantConvNeXt-V2-L (1536)   ─┘

The cRT cache stores RAW (un-L2-normalized) per-backbone features, so the head
was trained on un-normalized inputs. We replicate that here by skipping
F.normalize on the concat path.

Inputs to backbones use ImageNet normalization to match the DALI pipeline that
built the cache (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]) — the
training-time normalization mismatch (BioCLIP nominally wants CLIP stats) is
intentional because the head was fit to whatever the cache contained.

Output: aggregated softmax_mean logits per quadrat, saved in the same blob
format as `infer_tiles_adaptive.py` so `ensemble_logits.py` can mix it with
i002 logits directly.
"""
from __future__ import annotations
import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

# Make sure project root is importable for the backbone wrappers
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.bioclip import PlantBioCLIP
from src.models.vit_backbone import PlantViTBackbone
from src.models.convnext import PlantConvNeXt
from phases.crt_train.model import ClassificationHead

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("infer_crt")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
NUM_SPECIES = 7806


def grid_4x4_tiles(img: Image.Image) -> list[Image.Image]:
    w, h = img.size
    cw, ch = w / 4, h / 4
    tiles = []
    for r in range(4):
        for c in range(4):
            tiles.append(img.crop((int(c * cw), int(r * ch),
                                   int((c + 1) * cw), int((r + 1) * ch))))
    return tiles


def build_preprocess(img_size: int) -> T.Compose:
    return T.Compose([
        T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class TripleBackbone(torch.nn.Module):
    """Loads three frozen backbones and returns RAW (un-normalized) concat features.

    Differs from phases/p1_extract.FeatureExtractor by NOT applying F.normalize,
    matching the legacy cache format the cRT head was trained on.
    """

    def __init__(self, input_res: int = 224) -> None:
        super().__init__()
        self.bio = PlantBioCLIP(checkpoint="hf-hub:imageomics/bioclip-2", input_res=input_res)
        self.dino = PlantViTBackbone(model_name="vit_large_patch16_dinov3.lvd1689m", input_res=input_res)
        self.conv = PlantConvNeXt(model_name="convnextv2_large.fcmae_ft_in22k_in1k_384", input_res=input_res)
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is already ImageNet-normalized [B, 3, H, W]
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            f_bio = self.bio(x)
            f_dino = self.dino(x)
            f_conv = self.conv(x)
        f_bio = f_bio.float()
        f_dino = f_dino.float()
        f_conv = f_conv.float()
        return torch.cat([f_bio, f_dino, f_conv], dim=1).contiguous()


def load_crt_head(ckpt_path: str, device: str) -> torch.nn.Module:
    head = ClassificationHead(in_features=3328, hidden_features=2048, out_features=7808)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    head.load_state_dict(ckpt["model_state"])
    head = head.to(device).eval()
    logger.info(f"Loaded cRT head from {ckpt_path}  epoch={ckpt['epoch']}  best_val_acc={ckpt['best_val_acc']:.2f}%")
    return head


def load_idx_to_species(mapping_csv: str) -> list[str]:
    out = []
    with open(mapping_csv) as f:
        for line in f:
            sid = line.strip()
            if sid:
                out.append(sid)
    assert len(out) == NUM_SPECIES, f"Expected {NUM_SPECIES} species, got {len(out)}"
    return out


def list_test_images(image_dir: str) -> list[Path]:
    p = Path(image_dir)
    files = [f for f in p.rglob("*") if f.is_file() and f.suffix in _IMAGE_EXTS]
    return sorted(files)


@torch.no_grad()
def infer_one(img: Image.Image, backbone: TripleBackbone, head: torch.nn.Module,
              preprocess: T.Compose, device: str, batch_size: int) -> torch.Tensor:
    """Returns aggregated softmax_mean logits over the 16 tiles, shape [7806]."""
    tiles = grid_4x4_tiles(img.convert("RGB"))
    batches = torch.stack([preprocess(t) for t in tiles])  # [16, 3, 224, 224]
    all_logits = []
    for i in range(0, len(batches), batch_size):
        b = batches[i:i + batch_size].to(device)
        feats = backbone(b)
        logits = head(feats)[:, :NUM_SPECIES]  # drop padding slots 7806–7807
        all_logits.append(logits.float().cpu())
    tile_logits = torch.cat(all_logits, dim=0)  # [16, 7806]
    probs = F.softmax(tile_logits, dim=1).mean(dim=0)
    return torch.log(probs.clamp_min(1e-12))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--head", required=True, help="Path to cRT head .pth (head_a or head_b)")
    p.add_argument("--image-dir", required=True)
    p.add_argument("--mapping-csv", default="/workspace/plantclef/processed/species_ids_mapping.csv")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"device={device}  img_size={args.img_size}  batch_size={args.batch_size}")

    backbone = TripleBackbone(input_res=args.img_size).to(device)
    head = load_crt_head(args.head, device=device)
    preprocess = build_preprocess(args.img_size)
    idx_to_species = load_idx_to_species(args.mapping_csv)

    images = list_test_images(args.image_dir)
    if args.limit:
        images = images[:args.limit]
    logger.info(f"Found {len(images)} test images.")

    out_dir = Path(args.output_dir)
    (out_dir / "logits").mkdir(parents=True, exist_ok=True)

    all_logits = torch.zeros(len(images), NUM_SPECIES, dtype=torch.float32)
    quadrat_ids = []
    t0 = time.time()
    errors = 0

    for i, ipath in enumerate(images):
        try:
            img = Image.open(ipath)
            agg = infer_one(img, backbone, head, preprocess, device, args.batch_size)
            all_logits[i] = agg
            quadrat_ids.append(ipath.stem)
        except Exception as e:
            errors += 1
            logger.warning(f"  FAILED {ipath.name}: {e}")
            quadrat_ids.append(ipath.stem)

        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(images) - i - 1) / rate if rate > 0 else 0
            logger.info(f"  [{i+1:>5}/{len(images)}]  {elapsed:.0f}s  {rate:.1f} img/s  ETA {eta:.0f}s  errors={errors}")

    logger.info(f"Inference done: {len(images)} imgs in {time.time()-t0:.0f}s  errors={errors}")

    blob = {
        "logits": all_logits,
        "quadrat_ids": quadrat_ids,
        "idx_to_species": idx_to_species,
        "agg_mode": "softmax_mean",
    }
    out_path = out_dir / "logits" / "softmax_mean_logits.pt"
    torch.save(blob, out_path)
    logger.info(f"Logits saved: {out_path}  shape={tuple(all_logits.shape)}")


if __name__ == "__main__":
    main()
