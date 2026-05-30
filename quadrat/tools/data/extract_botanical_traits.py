"""Absolute Maximum Performance Botanical Trait Extraction.

Leverages Qwen2-VL-7B with vLLM or FlashAttention-2 for high-throughput
botanical trait extraction. Optimized for Blackwell RTX PRO 6000.
"""

import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

def main() -> None:
    """
    Main entry point for botanical trait extraction.

    Leverages Qwen2-VL-7B to extract traits (leaf shape, phyllotaxy, etc.)
    from images and saves them in JSONL format. Includes optimizations for
    Blackwell RTX PRO 6000 GPUs.

    Returns
    -------
    None
    """
    # 1. Configuration
    BASE_DIR = "/workspace/plantclef"
    TRAIN_CSV = os.path.join(BASE_DIR, "processed/student_train_final.csv")
    IMG_DIR = os.path.join(BASE_DIR, "raw/train/images_max_side_800/")
    OUTPUT_JSONL = os.path.join(BASE_DIR, "processed/botanical_traits.jsonl")
    
    DEVICE = "cuda"
    BATCH_SIZE = 4 # Reduced to prevent OOM with high-res tokens
    MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
    
    # 2. Load Model with Blackwell Optimizations
    print(f"[LVLM] Loading {MODEL_ID} with FlashAttention-2...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", # More stable than flash-attn during builds
        device_map="auto",
    )
    
    # RTX PRO 6000 Optimization: Compile the vision tower and language head
    print("[Blackwell] Compiling model for maximum throughput...")
    model = torch.compile(model, mode="max-autotune")
    
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    
    # 3. Botanical Prompt Engineering (Force Structured JSON)
    SYSTEM_PROMPT = (
        "You are an expert botanist. Analyze the plant in the image and provide its traits in valid JSON format. "
        "Fields: leaf_shape (ovate, lanceolate, etc), phyllotaxy (alternate, opposite, etc), "
        "flower_color, inflorescence_type, stem_type. If a trait is not visible, use 'null'."
    )
    
    # 4. Load Dataset
    df = pd.read_csv(TRAIN_CSV, sep=';')
    # Focus on a representative subset for distillation (e.g., 50k images)
    # or process all if time allows. For now, we take 50,000.
    sample_df = df.sample(n=min(50000, len(df)), random_state=42)
    
    # 5. Extraction Loop (Batched)
    existing_count = 0
    if os.path.exists(OUTPUT_JSONL):
        with open(OUTPUT_JSONL, 'r') as f:
            existing_count = sum(1 for _ in f)
    
    print(f"[LVLM] Starting extraction for {len(sample_df)} images (Already processed: {existing_count})...")
    
    with open(OUTPUT_JSONL, 'a') as f_out:
        batch_items = []
        for idx, row in tqdm(sample_df.iloc[existing_count:].iterrows(), total=len(sample_df) - existing_count):
            img_name = row['image_name']
            sid = str(row['species_ids']).split(',')[0] # Use primary species for path
            
            # Find correct path
            if img_name.startswith('LUCAS'):
                img_path = os.path.join(BASE_DIR, "raw/pseudo_quadrats", img_name)
            elif img_name.startswith('collage'):
                img_path = os.path.join(BASE_DIR, "processed/collages", img_name)
            else:
                img_path = os.path.join(IMG_DIR, sid, img_name)
                
            if not os.path.exists(img_path): continue
            
            # Prepare Message with explicit pixel limits
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image", 
                            "image": img_path,
                            "max_pixels": 512 * 512, # Cap at 512x512 equivalent tokens
                        },
                        {"type": "text", "text": "Extract botanical traits in JSON."},
                    ],
                }
            ]
            
            batch_items.append({"id": img_name, "messages": messages})
            
            if len(batch_items) >= BATCH_SIZE:
                # Process Batch
                # apply_chat_template is for the model, but process_vision_info needs the raw messages
                texts = [processor.apply_chat_template(m["messages"], tokenize=False, add_generation_prompt=True) for m in batch_items]
                raw_messages_batch = [m["messages"] for m in batch_items]
                
                # Cap resolution to prevent token explosion (e.g., max 512x512 equivalent)
                image_inputs, video_inputs = process_vision_info(raw_messages_batch)
                inputs = processor(
                    text=texts,
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(DEVICE)
                
                # Force dynamic resolution capping via processor kwargs if available in this version
                # Otherwise, Qwen2-VL will use the pixel values from process_vision_info
                
                # Inference
                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=128)
                    generated_ids_trimmed = [
                        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    output_texts = processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )
                
                # Save results
                for item, out_text in zip(batch_items, output_texts):
                    try:
                        # Attempt to clean up JSON if model adds markdown blocks
                        clean_json = out_text.strip().replace("```json", "").replace("```", "")
                        traits = json.loads(clean_json)
                        result = {"image_name": item["id"], "traits": traits}
                        f_out.write(json.dumps(result) + "\n")
                    except:
                        # Fallback for parsing errors
                        f_out.write(json.dumps({"image_name": item["id"], "raw_output": out_text}) + "\n")
                
                f_out.flush()
                batch_items = []

    print(f"Success! Botanical traits saved to {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()
