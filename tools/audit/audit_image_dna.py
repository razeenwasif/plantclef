import PIL.Image
import os
import glob
import numpy as np

def audit_image_dna(image_dir):
    print("=== Training Image DNA Audit ===")
    sample_imgs = glob.glob(os.path.join(image_dir, "**/*.jpg"), recursive=True)[:5]
    
    for img_path in sample_imgs:
        img = PIL.Image.open(img_path)
        # Check chroma subsampling (4:2:0, 4:4:4, etc)
        subsampling = img.layer_subsampling if hasattr(img, 'layer_subsampling') else "Unknown"
        print(f"Image: {os.path.basename(img_path)} | Size: {img.size} | Subsampling: {subsampling}")

if __name__ == "__main__":
    audit_image_dna("/workspace/plantclef/raw/train/images_max_side_800/")
