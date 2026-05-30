"""
Asymmetric Dual-Teacher Distillation (AD-TD) Phase Entry Point.
Handles both teacher extraction and student distillation.
"""
import argparse
import os
import sys
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

def main():
    parser = argparse.ArgumentParser(description="Asymmetric Dual-Teacher Distillation Phase")
    parser.add_argument("--mode", choices=["extract", "train"], default="train",
                        help="Mode: 'extract' to pre-compute teacher logits, 'train' for student distillation")
    parser.add_argument("--variant", choices=["standard", "elite"], default="elite",
                        help="AD-TD variant (Standard: [CLS]+[DIST], Elite: [CLS]+[DIST_tax]+[DIST_geo])")
    parser.add_argument("--config", type=str, help="Path to YAML config")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", type=str, default="ad_td_run")
    
    # Hidden arg from launch.sh
    parser.add_argument("role", nargs="?", default="sprint")
    
    args, unknown = parser.parse_known_args()

    if args.mode == "extract":
        print(f"🚀 Initializing Teacher Extraction Phase (Variant: {args.variant})")
        from .extract_teachers import run_extraction
        # In a real scenario, we'd parse more specific paths from config
        img_dir = "data/train/images"
        output_dir = f"data/cache/teachers_{args.variant}"
        bio_path = "models/bioclip_finetuned.pth"
        dino_path = "models/dinov3_finetuned.pth"
        run_extraction(img_dir, output_dir, bio_path, dino_path)
        
    elif args.mode == "train":
        print(f"🚀 Initializing Student Distillation Phase (Variant: {args.variant})")
        # For 'train' mode, we'd typically be called via torchrun
        from .train_trinity import train_one_epoch, setup_hardware_context, TrinityLoss
        from .trinity_deit import create_trinity_student
        
        hw = setup_hardware_context()
        model = create_trinity_student(variant=args.variant, model_size="large").to(hw["device"])
        
        if hw["world_size"] > 1:
            from torch.nn.parallel import DistributedDataParallel as DDP
            model = DDP(model, device_ids=[hw["local_rank"]])
            
        criterion = TrinityLoss(variant=args.variant)
        # ... remainder of training loop setup (optimizer, dataloader) ...
        print(f"Ready for training variant: {args.variant}")

if __name__ == "__main__":
    main()
