import pandas as pd
import os
import time

def convert_to_feather():
    csv_path = "/workspace/plantclef/processed/student_train_final.csv"
    feather_path = csv_path.replace(".csv", ".feather")
    
    if not os.path.exists(csv_path):
        print(f"[Error] CSV not found: {csv_path}")
        return

    print(f"[Binary-Migrate] Loading {csv_path}...")
    start = time.time()
    # assuming semicolon separator
    df = pd.read_csv(csv_path, sep=';', low_memory=False)
    load_time = time.time() - start
    print(f"[Binary-Migrate] CSV Loaded in {load_time:.2f}s.")

    # Force string type for IDs to avoid Arrow type inference errors with mixed data
    df['species_ids'] = df['species_ids'].astype(str)

    print(f"[Binary-Migrate] Saving to {feather_path}...")
    start = time.time()
    df.to_feather(feather_path)
    save_time = time.time() - start
    print(f"[Binary-Migrate] Feather saved in {save_time:.2f}s.")
    
    # Verification
    print(f"[Binary-Migrate] Verifying load speed...")
    start = time.time()
    df_f = pd.read_feather(feather_path)
    feather_load_time = time.time() - start
    print(f"[Binary-Migrate] Feather loaded in {feather_load_time:.4f}s.")
    print(f"[Binary-Migrate] Speedup: {load_time / feather_load_time:.1f}x")

if __name__ == "__main__":
    convert_to_feather()
