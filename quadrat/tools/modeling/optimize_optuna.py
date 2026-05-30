"""Optuna Postprocessing Optimizer for PlantCLEF 2026.

Optimizes post-processing hyperparameters (threshold, phenology_beta, fw_lambda, etc.)
over cached logits to maximize Macro-F1 score without running the visual backbone repeatedly.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import optuna

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.inference.config import PostprocessConfig
from src.inference.postprocess import postprocess
from src.inference.types import ImagePrediction

def load_cached_data():
    # If the user has cached logits, load them. For now, simulate loading or instruct user to generate them.
    # We will look for a pre-saved NPZ file containing 'logits' and 'labels'
    cache_path = "/dev/shm/optuna_logits.npz"
    if not os.path.exists(cache_path):
        print(f"Cache not found at {cache_path}. Please run inference once to save logits.")
        print("Expected format: np.savez(cache_path, logits=array[N, C], labels=array[N])")
        sys.exit(1)
        
    data = np.load(cache_path, allow_pickle=True)
    return data['logits'], data['labels']

def parse_species(s):
    if isinstance(s, list) or isinstance(s, np.ndarray): 
        return [str(x) for x in s]
    try:
        s_str = str(s).strip()
        return [str(x).strip() for x in s_str.split(';')]
    except:
        return []

def objective(trial, logits, true_labels):
    # Suggest hyperparameters
    threshold = trial.suggest_float("threshold", 0.01, 0.2, log=True)
    top_k = trial.suggest_int("top_k", 1, 10)
    phenology_beta = trial.suggest_float("phenology_beta", 0.0, 2.0)
    fw_lambda = trial.suggest_float("fw_lambda", 0.01, 0.5, log=True)
    use_phenology = trial.suggest_categorical("use_phenology", [True, False])
    use_frank_wolfe = trial.suggest_categorical("use_frank_wolfe", [True, False])

    cfg = PostprocessConfig(
        global_threshold=threshold,
        top_k=top_k,
        use_phenology=use_phenology,
        phenology_beta=phenology_beta,
        use_frank_wolfe=use_frank_wolfe,
        fw_lambda=fw_lambda,
        min_predictions=1
    )
    
    y_pred_list = []
    y_true_list = []
    
    for i in range(len(logits)):
        pred = ImagePrediction(image_id=str(i), class_scores=logits[i].copy())
        
        # Apply postprocessing
        # Note: In a real scenario with phenology, we also need the month/DOY per image.
        # This script assumes 'postprocess' can run on raw logits for tuning.
        result = postprocess(pred, cfg)
        
        y_pred_list.append([str(x) for x in result.predicted_class_indices])
        
        gt = parse_species(true_labels[i])
        y_true_list.append(gt)

    mlb = MultiLabelBinarizer()
    mlb.fit(y_true_list + y_pred_list)
    
    bin_target = mlb.transform(y_pred_list)
    bin_true = mlb.transform(y_true_list)
    
    score = f1_score(bin_true, bin_target, average='macro', zero_division=0)
    return score

def main():
    print("Loading cached logits and labels...")
    logits, labels = load_cached_data()
    print(f"Loaded {len(logits)} samples.")
    
    study = optuna.create_study(direction="maximize")
    
    # We pass the pre-loaded data into the objective
    study.optimize(lambda trial: objective(trial, logits, labels), n_trials=50)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

if __name__ == "__main__":
    main()
