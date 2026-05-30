import pandas as pd
import os
import sys

# Add workspace root to path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(root_dir, 'src'))

import config

def stratify(max_samples_per_class=500):
    input_csv = config.CLEANED_CSV
    output_csv = input_csv.replace('.csv', '_stratified.csv')
    
    print(f"Loading metadata from {input_csv}...")
    df = pd.read_csv(input_csv, sep=';')
    
    total_before = len(df)
    print(f"Total samples before stratification: {total_before:,}")
    
    # Stratified Sub-sampling
    # Group by species and take up to max_samples_per_class
    df_stratified = df.groupby('species_id').apply(
        lambda x: x.sample(n=min(len(x), max_samples_per_class), random_state=42)
    ).reset_index(drop=True)
    
    total_after = len(df_stratified)
    reduction = (1 - total_after / total_before) * 100
    
    print(f"Total samples after stratification:  {total_after:,}")
    print(f"Reduction: {reduction:.1f}%")
    print(f"Unique Species: {df_stratified['species_id'].nunique():,}")
    
    # Save the new metadata
    df_stratified.to_csv(output_csv, sep=';', index=False)
    print(f"Stratified metadata saved to: {output_csv}")
    
    return output_csv

if __name__ == "__main__":
    # Cap common species at 500 images to boost Macro-F1
    new_csv = stratify(max_samples_per_class=500)
    print(f"\n[Action Required] To use this, update CLEANED_CSV in src/config.py to:\n'{new_csv}'")
