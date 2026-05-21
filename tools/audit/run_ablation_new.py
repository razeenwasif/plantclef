import os
import subprocess
import yaml
import pandas as pd
import time

# The NEWly trained DINOv3 Small (Aligned with current shards)
MODEL_PATH = "models/lightweight_dinov3_latest.pth"

# Reduced set for validation verification
STRATEGIES = {
    "0_baseline":   {"pav": False, "taxon": False, "fw": False, "ising": False, "phen": False, "geo": False, "retinex": False},
}

def run_ablation():
    results = []
    
    for name, flags in STRATEGIES.items():
        print(f"\n[Ablation] Testing NEW DINOv3 on Mini-Val: {name}...")
        
        config = {
            "model": {
                "checkpoints": [{"path": MODEL_PATH, "resolution": 224}],
                "num_classes": 7806,
                "resolution": 224
            },
            "dataset": {
                "test_csv": "data/mini_val_inference.csv",
                "img_dir": "data/mini_val_images"
            },
            "tiling": {"enabled": False},
            "inference": {
                "batch_size": 128, "num_workers": 8, "aggregation": "mean",
                "disable_logit_standardization": False, "use_hflip_tta": False, "use_ttt": False
            },
            "postprocess": {
                "threshold": 0.0, "top_k": 5, "min_predictions": 1, "min_vegetation_frac": 0.0,
                "use_pav_calibration": flags["pav"],
                "use_taxon_filter":    flags["taxon"],
                "use_frank_wolfe":     flags["fw"],
                "use_ising_model":     flags["ising"],
                "use_phenology":       flags["phen"],
                "use_geochem":         flags["geo"],
                "use_retinex":         flags["retinex"],
                "fw_top_k":            5
            },
            "output": {
                "species_mapping": "/workspace/plantclef/processed/species_ids_mapping.csv",
                "submission_csv": f"submissions/ablation_new_dino_{name}.csv"
            }
        }
        
        cfg_path = f"configs/ablation_new_dino_{name}.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(config, f)
            
        # Run Inference
        cmd = [
            "/workspace/pytorch_env/bin/torchrun", "--nproc_per_node=1", "--standalone",
            "-m", "phases.inference.run", "--config", cfg_path
        ]
        subprocess.run(cmd, check=True)
        
        # Run Simulator
        sim_cmd = [
            "/workspace/pytorch_env/bin/python3", "tools/inference/simulate_f1_final.py",
            "--target", f"submissions/ablation_new_dino_{name}.csv"
        ]
        sim_out = subprocess.check_output(sim_cmd).decode()
        
        f1 = 0.0
        acc = 0.0
        for line in sim_out.split("\n"):
            if "REAL VALIDATION MACRO F1:" in line:
                f1 = float(line.split(":")[-1].strip())
            if "REAL TOP-1 ACCURACY:" in line:
                acc = float(line.split(":")[-1].replace("%", "").strip())
                
        results.append({"strategy": name, "macro_f1": f1, "top1_acc": acc})
        print(f" -> Result for {name}: F1={f1}, Acc={acc}%")

if __name__ == "__main__":
    run_ablation()
