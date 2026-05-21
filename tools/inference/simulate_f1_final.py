import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
import ast
import argparse

def parse_species(s):
    if isinstance(s, list): return s
    try:
        s_str = str(s).strip()
        if '[' in s_str:
            return [str(x).strip() for x in ast.literal_eval(s_str)]
        return [str(x).strip() for x in s_str.split(';')]
    except:
        return []

def run_simulation(target_path=None, gt_path=None):
    # Load files - we ignore quadrat_id and pair them by row order
    t_path = target_path or "submissions/mini_val_results.csv"
    g_path = gt_path or "data/mini_val_ground_truth.csv"
    
    print(f"[*] Loading results from {t_path} and {g_path}...")
    df_t = pd.read_csv(t_path)
    df_g = pd.read_csv(g_path)
    
    # Take the minimum length to avoid index errors
    n = min(len(df_t), len(df_g))
    y_target = [parse_species(x) for x in df_t['species_ids'].iloc[:n]]
    y_true = [parse_species(x) for x in df_g['species_ids'].iloc[:n]]
    
    print(f"[*] Comparing {n} samples via row-order alignment.")
    
    mlb = MultiLabelBinarizer()
    all_labels = y_target + y_true
    mlb.fit(all_labels)
    
    bin_target = mlb.transform(y_target)
    bin_true = mlb.transform(y_true)
    
    score = f1_score(bin_true, bin_target, average='macro', zero_division=0)
    
    # Calculate simple Top-1 Accuracy as a sanity check
    top1_hits = 0
    for i in range(n):
        if y_target[i] and y_true[i]:
            if y_target[i][0] == y_true[i][0]:
                top1_hits += 1
    
    print("\n" + "="*40)
    print(f"REAL VALIDATION MACRO F1: {score:.4f}")
    print(f"REAL TOP-1 ACCURACY:     {100.0 * top1_hits / n:.2f}%")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", help="Target submission CSV")
    parser.add_argument("--gt", help="Ground Truth CSV", default="data/mini_val_ground_truth.csv")
    args = parser.parse_args()
    run_simulation(args.target, args.gt)
