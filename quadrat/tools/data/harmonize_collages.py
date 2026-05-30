"""PLANTCLEF: Multi-GPU Diffusion Harmonization.
Distributes 50k collages across all available GPUs.
"""

import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
import argparse

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    args = parser.parse_args()

    # 1. Configuration
    BASE_DIR = "/workspace/plantclef"
    COLLAGE_DIR = os.path.join(BASE_DIR, "processed/collages")
    OUTPUT_DIR = os.path.join(BASE_DIR, "processed/diffusion_collages")
    METADATA_PATH = os.path.join(BASE_DIR, "processed/synthetic_collages.csv")
    NAME_MAP_PATH = "/workspace/plantclef/raw/models/pretrained_models/species_id_to_name.txt"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = f"cuda:{args.rank}"
    torch.cuda.set_device(device)
    
    # 2. Load Mappings
    name_df = pd.read_csv(NAME_MAP_PATH, sep=';')
    id_to_name = dict(zip(name_df['species_id'].astype(str), name_df['species']))
    
    # 3. Load Pipeline
    print(f"[Rank {args.rank}] Loading SD1.5...")
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        safety_checker=None,
        use_safetensors=True
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    # Optimizations
    pipe.enable_xformers_memory_efficient_attention()
    # plantclef: Fast compilation
    pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead")
    
    # 4. Load & Shard Metadata
    df = pd.read_csv(METADATA_PATH, sep=';')
    # Split work
    shards = np.array_split(range(len(df)), args.world_size)
    my_indices = shards[args.rank]
    my_df = df.iloc[my_indices]
    
    print(f"[Rank {args.rank}] Handling {len(my_df)} images.")
    
    # 5. Loop
    STRENGTH = 0.25
    STEPS = 20
    
    for _, row in tqdm(my_df.iterrows(), total=len(my_df), desc=f"GPU {args.rank}"):
        img_name = row['image_name']
        save_path = os.path.join(OUTPUT_DIR, img_name)
        
        if os.path.exists(save_path):
            continue
            
        try:
            # Build Prompt
            sids = str(row['species_ids']).split(',')
            names = [id_to_name.get(sid, "plant") for sid in sids]
            prompt = f"overhead botanical view, {', '.join(names[:3])}, highly detailed, natural lighting"
            
            init_image = Image.open(os.path.join(COLLAGE_DIR, img_name)).convert("RGB")
            init_image = init_image.resize((512, 512))
            
            with torch.inference_mode(), torch.autocast("cuda"):
                image = pipe(
                    prompt=prompt,
                    image=init_image,
                    strength=STRENGTH,
                    num_inference_steps=STEPS,
                    guidance_scale=7.0
                ).images[0]
                
            image.save(save_path)
        except Exception as e:
            print(f"Error on {img_name}: {e}")

if __name__ == "__main__":
    main()
