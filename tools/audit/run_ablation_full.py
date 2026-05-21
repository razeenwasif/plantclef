import os
import subprocess
import yaml
import pandas as pd

MODEL_PATH = "models/lightweight_dinov3_latest.pth"

# Base configurations
RESOLUTIONS = [224, 336]
THRESHOLDS = [0.03, 0.05]

# Post-processing parameter sweeps
TECHNIQUES = {
    "phenology": [{"phen_beta": 0.5}, {"phen_beta": 1.0}],
    "frankwolfe": [{"fw_lambda": 0.05}, {"fw_lambda": 0.1}],
    "ising": [{"ising_temp": 1.0}, {"ising_temp": 2.0}],
    "geochem": [{"geo_weight": 0.5}],
    "taxonomic": [{"taxon_weight": 1.0}],
}

def generate_combinations():
    combinations = []
    
    # 1. Baseline runs across resolutions and thresholds
    for res in RESOLUTIONS:
        for thr in THRESHOLDS:
            base_config = {"name": f"baseline_r{res}_t{thr}", "res": res, "thr": thr, "techs": {}}
            combinations.append(base_config)
            
            # 2. Single techniques with parameter sweeps
            for tech, param_grid in TECHNIQUES.items():
                for params in param_grid:
                    cfg = base_config.copy()
                    cfg["name"] = f"{tech}_r{res}_t{thr}_" + "_".join([f"{k}{v}" for k, v in params.items()])
                    cfg["techs"] = {tech: params}
                    combinations.append(cfg)
            
            # 3. Combinations of two techniques
            tech_names = list(TECHNIQUES.keys())
            for i in range(len(tech_names)):
                for j in range(i + 1, len(tech_names)):
                    tech1, tech2 = tech_names[i], tech_names[j]
                    # take first param setting for simplicity to avoid combinatorial explosion
                    params1 = TECHNIQUES[tech1][0]
                    params2 = TECHNIQUES[tech2][0]
                    
                    cfg = base_config.copy()
                    cfg["name"] = f"combo_{tech1}_{tech2}_r{res}_t{thr}"
                    cfg["techs"] = {tech1: params1, tech2: params2}
                    combinations.append(cfg)
                    
    return combinations

def run_ablation():
    results = []
    combinations = generate_combinations()
    
    for combo in combinations:
        name = combo["name"]
        print(f"\n[Ablation] Testing {name}...")
        
        techs = combo["techs"]
        
        config = {
            "model": {
                "checkpoints": [{"path": MODEL_PATH, "resolution": combo["res"]}],
                "num_classes": 7806,
                "resolution": combo["res"]
            },
            "dataset": {
                "test_csv": "data/mini_val_inference.csv",
                "img_dir": "data/mini_val_images"
            },
            "tiling": {"enabled": False},
            "inference": {
                "batch_size": 64, "num_workers": 4, "aggregation": "mean",
                "disable_logit_standardization": False, "use_hflip_tta": False, "use_ttt": False
            },
            "postprocess": {
                "threshold": combo["thr"], 
                "top_k": 5, 
                "min_predictions": 1,
                "use_phenology": "phenology" in techs,
                "phenology_beta": techs.get("phenology", {}).get("phen_beta", 1.0),
                "use_frank_wolfe": "frankwolfe" in techs,
                "fw_lambda": techs.get("frankwolfe", {}).get("fw_lambda", 0.05),
                "use_ising_model": "ising" in techs,
                "ising_temp": techs.get("ising", {}).get("ising_temp", 1.0),
                "use_geochem": "geochem" in techs,
                "use_taxon_filter": "taxonomic" in techs
            },
            "output": {
                "species_mapping": "/workspace/plantclef/processed/species_ids_mapping.csv",
                "submission_csv": f"submissions/{name}.csv"
            }
        }
        
        cfg_path = f"configs/experimental/ablation_new_{name}.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(config, f)
            
        cmd = ["torchrun", "--nproc_per_node=1", "--standalone", "-m", "phases.inference.run", "--config", cfg_path]
        try:
            subprocess.run(cmd, check=True)
            
            sim_cmd = ["python", "tools/inference/simulate_f1_final.py", "--target", f"submissions/{name}.csv"]
            sim_out = subprocess.check_output(sim_cmd).decode()
            
            f1 = 0.0
            for line in sim_out.split("\n"):
                if "REAL VALIDATION MACRO F1:" in line:
                    f1 = float(line.split(":")[-1].strip())
            results.append({"strategy": name, "macro_f1": f1})
            print(f" -> Result for {name}: {f1}")
        except subprocess.CalledProcessError as e:
            print(f" -> Failed for {name}: {e}")

    df = pd.DataFrame(results)
    print("\n" + "="*40)
    print(" FINAL ABLATION RESULTS")
    print(df)
    print("="*40)
    df.to_csv("ablation_results.csv", index=False)

if __name__ == "__main__":
    run_ablation()
