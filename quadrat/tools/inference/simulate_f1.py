import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
import ast
import argparse
import sys

def parse_species(s):
    if isinstance(s, list): return s
    try:
        # Handles both "[1, 2]" and "1;2" formats
        if '[' in str(s):
            return [str(x) for x in ast.literal_eval(s)]
        return [str(x) for x in str(s).split(';')]
    except:
        return []

def calculate_macro_f1(target_csv, baseline_csv):
    print(f"[*] Loading Target: {target_csv}")
    df_t = pd.read_csv(target_csv)
    print(f"[*] Loading Baseline (Pseudo-GT): {baseline_csv}")
    df_b = pd.read_csv(baseline_csv)

    # Merge to ensure alignment
    df = pd.merge(df_t, df_b, on='quadrat_id', suffixes=('_target', '_baseline'))
    
    if len(df) == 0:
        print("[!] Error: No overlapping quadrat_ids found.")
        return

    print(f"[*] Aligned {len(df)} samples for comparison.")

    y_target = [parse_species(x) for x in df['species_ids_target']]
    y_baseline = [parse_species(x) for x in df['species_ids_baseline']]

    # We use MultiLabelBinarizer to create the [N_samples, 7808] matrix
    mlb = MultiLabelBinarizer()
    # Fit on both to ensure the class space covers everything
    mlb.fit(y_target + y_baseline)
    
    print(f"[*] Encoding binary matrices (Classes: {len(mlb.classes_)})...")
    bin_target = mlb.transform(y_target)
    bin_baseline = mlb.transform(y_baseline)

    # Calculate Macro F1
    # average='macro' calculates F1 for each label and finds their unweighted mean.
    score = f1_score(bin_baseline, bin_target, average='macro', zero_division=0)
    
    print("\n" + "="*40)
    print(f"SIMULATED MACRO F1: {score:.4f}")
    print("="*40)
    print(f"Note: This is relative to {baseline_csv}.")
    print("If the baseline has 0.38 LB, and this score is > 0.90,")
    print("your new LB score is likely higher than 0.38.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target", help="Your new submission.csv")
    parser.add_argument("baseline", help="The high-scoring baseline to compare against")
    args = parser.parse_args()
    
    calculate_macro_f1(args.target, args.baseline)
