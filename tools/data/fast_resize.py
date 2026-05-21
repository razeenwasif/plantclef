import os
import cv2
import glob
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from pathlib import Path

# --- CONFIGURATION ---
SRC_DIR = "/workspace/plantclef/raw/train/" 
DST_DIR = "/workspace/plantclef/processed/train_700px/"
TARGET_SIZE = 700
NUM_WORKERS = 90  # Optimized for 96-core EPYC

def process_image(img_path):
    try:
        rel_path = os.path.relpath(img_path, SRC_DIR)
        save_path = os.path.join(DST_DIR, rel_path)
        
        if os.path.exists(save_path):
            return True

        img = cv2.imread(img_path)
        if img is None:
            return False

        h, w = img.shape[:2]
        if h > w:
            new_h, new_w = TARGET_SIZE, int(TARGET_SIZE * (w / h))
        else:
            new_h, new_w = int(TARGET_SIZE * (h / w)), TARGET_SIZE

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(save_path, resized, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return True
    except Exception:
        return False

def main():
    print(f"[*] Scanning {SRC_DIR} for images...")
    image_paths = glob.glob(os.path.join(SRC_DIR, "**/*.jpg"), recursive=True)
    image_paths += glob.glob(os.path.join(SRC_DIR, "**/*.JPG"), recursive=True)
    
    print(f"[*] Found {len(image_paths):,} images.")
    
    subdirs = set(os.path.dirname(os.path.relpath(p, SRC_DIR)) for p in image_paths)
    for s in tqdm(subdirs, desc="Creating Dirs"):
        os.makedirs(os.path.join(DST_DIR, s), exist_ok=True)

    print(f"[*] Launching 90-core EPYC Swarm...")
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        list(tqdm(executor.map(process_image, image_paths), total=len(image_paths), desc="Resizing"))

if __name__ == "__main__":
    main()
