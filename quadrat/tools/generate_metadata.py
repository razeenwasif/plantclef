import os
import pandas as pd
from tqdm import tqdm
from config import IMG_DIR, RAW_CSV

img_dir = IMG_DIR
output_csv = RAW_CSV

data = []
# Ensure IMG_DIR exists
if not os.path.exists(img_dir):
    print(f"Error: {img_dir} does not exist.")
    exit(1)

species_folders = [d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))]

print(f"Scanning {len(species_folders)} species folders...")
for species_id in tqdm(species_folders):
    species_path = os.path.join(img_dir, species_id)
    images = [f for f in os.listdir(species_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for img in images:
        data.append({
            'image_name': img,
            'species_id': int(species_id)
        })

df = pd.DataFrame(data)
print(f"Total images found: {len(df)}")
# Check if parent directory exists for output_csv
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
df.to_csv(output_csv, sep=';', index=False)
print(f"Metadata saved to {output_csv}")
