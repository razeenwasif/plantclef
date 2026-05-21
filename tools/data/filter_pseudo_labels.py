import pandas as pd
import os
from tqdm import tqdm

# --- CONFIG ---
SUBMISSION_CSV = "submissions/dinov3_bioclip.csv" # Your best inference result
TRAIN_CSV = "/workspace/plantclef/processed/student_train_final.csv"
OUTPUT_CSV = "/workspace/plantclef/processed/champion_train.csv"
TEST_IMG_DIR = "/workspace/plantclef/processed/test_700px/" # Assumes you also resize test set
CONFIDENCE_THRESHOLD = 0.98

def main():
    print(f"[*] Loading submission from {SUBMISSION_CSV}...")
    # submission format: image_id;species_id;confidence (usually)
    # Note: Kaggle format might be image_id;species_id1 species_id2...
    df_sub = pd.read_csv(SUBMISSION_CSV, sep=';')
    
    print(f"[*] Filtering for confidence > {CONFIDENCE_THRESHOLD}...")
    # Take only the top-tier predictions to avoid "label noise"
    df_high = df_sub[df_sub['confidence'] >= CONFIDENCE_THRESHOLD].copy()
    
    print(f"[*] Found {len(df_high):,} high-confidence pseudo-labels.")
    
    # Format for training: must match student_train_final columns
    # image_name;species_id;image_path
    df_pseudo = pd.DataFrame()
    df_pseudo['image_name'] = df_high['image_id']
    df_pseudo['species_id'] = df_high['species_id']
    # We will need to make sure the test images are also in a species-named subfolder 
    # OR we update the dataloader to handle flat folders.
    df_pseudo['is_pseudo'] = True
    
    print(f"[*] Merging with original training set...")
    df_train = pd.read_csv(TRAIN_CSV, sep=';')
    df_train['is_pseudo'] = False
    
    df_champion = pd.concat([df_train, df_pseudo], ignore_index=True)
    
    df_champion.to_csv(OUTPUT_CSV, sep=';', index=False)
    print(f"[+] SUCCESS: Champion dataset created with {len(df_champion):,} images.")
    print(f"[!] Tip: Run Phase 2b for 1 epoch on this file to adapt to the test domain.")

if __name__ == "__main__":
    main()
