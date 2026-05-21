import pandas as pd
import numpy as np

def check():
    train_csv = "/workspace/plantclef/processed/student_train_final.csv"
    val_csv = "/workspace/plantclef/raw/PlantCLEF2024_single_plant_training_metadata.csv" # Using as proxy for distribution
    
    print("=== Distribution Audit ===")
    
    df_train = pd.read_csv(train_csv, sep=';', low_memory=False)
    col = 'species_ids' if 'species_ids' in df_train.columns else 'species_id'
    train_counts = df_train[col].value_counts()
    
    print(f"Training Species: {len(train_counts)}")
    print(f"Mean samples/species: {train_counts.mean():.1f}")
    print(f"Max samples: {train_counts.max()}")
    print(f"Min samples: {train_counts.min()}")
    
    # Check for extreme skew
    top_10pct = int(len(train_counts) * 0.1)
    top_mass = train_counts.head(top_10pct).sum() / train_counts.sum()
    print(f"Top 10% species account for {top_mass:.1%} of data.")

if __name__ == "__main__":
    check()
