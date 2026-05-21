"""Universal Launcher and Orchestrator for PlantCLEF 2026 Dual-Hardware Pipeline.

This script coordinates the final student run by:
1. Performing prerequisite checks for Phase 6 research assets.
2. Handling environment setup for both NVIDIA CUDA and Google TPU.
3. Patching the feature cache for harmonized images.
4. Launching the hardware-agnostic training orchestrator.

Usage:
    # Launch on CUDA (Standard)
    python main/launch_final_student_run.py --num_gpus=X

    # Launch on TPU
    python main/launch_final_student_run.py --tpu
"""

from __future__ import annotations

import os
import sys
import argparse
import subprocess
from typing import Any

import pandas as pd

# Add src directory to sys.path so config.py can be found
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Also add root for src.* imports
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the universal launcher."""
    parser = argparse.ArgumentParser(description="PlantCLEF 2026 Universal Launcher")
    parser.add_argument("--tpu", action="store_true", help="Launch on Google TPU")
    parser.add_argument("--extreme", action="store_true", help="Enable Extreme Blackwell Optimizations")
    parser.add_argument("--dual", action="store_true", help="Enable Dual-GPU Mode (Single Node)")
    parser.add_argument("--multi_node", action="store_true", help="Enable 4-GPU Multi-Node Mode (Two Pods)")
    parser.add_argument("--num_gpus", type=int, default=2, help="GPUs per node")
    parser.add_argument("--phase", type=str, default="all", choices=["all", "p1", "p2a", "p2b"])
    parser.add_argument("--master_addr", type=str, default="127.0.0.1")
    parser.add_argument("--master_port", type=int, default=29500)
    return parser.parse_args()


def check_prerequisites() -> bool:
    """Verifies that all Phase 6 research assets are present.

    Returns
    -------
    bool
        True if checks are completed (always returns True, but logs warnings).
    """
    print("[Orchestrator] Checking Phase 6 Assets...")
    
    paths = {
        "Ecological Traits": "data/ecological_traits.npy",
        "Harmonized Images": "/workspace/plantclef/processed/diffusion_collages",
        "Botanical Traits": "/workspace/plantclef/processed/botanical_traits.jsonl"
    }
    
    missing = []
    for name, path in paths.items():
        if not os.path.exists(path):
            missing.append(name)
        elif name == "Harmonized Images" and len(os.listdir(path)) < 50000:
            missing.append(f"{name} (Incomplete count)")
            
    if missing:
        print(f"WARNING: Missing or incomplete assets: {', '.join(missing)}")
        # We don't return False here to allow standard runs without all research assets
    else:
        print("[Orchestrator] All assets verified. System ready.")
    return True


def patch_feature_cache() -> None:
    """Checks and provides info about the feature cache."""
    cache_path = config.FEATURE_CACHE_PATH
    if not os.path.exists(cache_path):
        print("[Orchestrator] No existing cache found. Full extraction will occur in trainer.py.")
        return

    print(f"[Orchestrator] Cache located: {cache_path}")
    
    # Identify which entries are 'collages' requiring potential patching
    try:
        csv_path = config.CLEANED_CSV if os.path.exists(config.CLEANED_CSV) else config.RAW_CSV
        df = pd.read_csv(csv_path, sep=';')
        col_name = 'species_ids' if 'species_ids' in df.columns else 'species_id'
        df[col_name] = df[col_name].astype(str)
        
        collage_indices = df[df['image_name'].str.startswith('collage')].index.tolist()
        if collage_indices:
            print(f"[Orchestrator] Found {len(collage_indices)} harmonized images requiring potential feature re-extraction.")
            print(f"[Orchestrator] trainer.py will handle any required feature extraction.")
    except Exception as e:
        print(f"[Orchestrator] Cache analysis skipped: {e}")


def main() -> None:
    """Main entry point for the universal launcher."""
    args = parse_args()
    
    # 1. Prerequisite Checks
    check_prerequisites()
    patch_feature_cache()

    if args.multi_node:
        print("\n" + "!"*60)
        print(" LAUNCHING EXTREME MODE: MANUAL 4-GPU MULTI-NODE CLUSTER")
        print("!"*60 + "\n")
        
        # ORACLE: Configuration for the 4-GPU cluster
        POD_A_IP = "213.173.111.137"
        POD_A_PORT = "44355"
        POD_B_IP = "213.173.105.149"
        POD_B_PORT = "31157"
        WORLD_SIZE = "4"
        MASTER_ADDR = POD_A_IP 
        # TACTICAL: Use the pod's public SSH port for DDP to bypass firewalls
        MASTER_PORT = POD_A_PORT 
        
        processes = []
        import time
        
        # 1. Local Ranks (0 and 1) on Pod A
        for rank in [0, 1]:
            env = os.environ.copy()
            env.update({
                "CUDA_VISIBLE_DEVICES": str(rank),
                "LOCAL_RANK": str(rank),
                "RANK": str(rank),
                "WORLD_SIZE": WORLD_SIZE,
                "MASTER_ADDR": MASTER_ADDR,
                "MASTER_PORT": MASTER_PORT,
                "PYTHONUNBUFFERED": "1"
            })
            cmd = [sys.executable, "-u", "src/training/trainer.py", "--phase", args.phase, "--local_rank", str(rank)]
            log_f = open(f"models/cuda_deep_sat/rank_{rank}.log", "wb", buffering=0)
            p = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=log_f)
            processes.append((p, log_f))
            print(f"[Launcher] ORACLE: Started Local Rank {rank} on GPU {rank}")

        # 2. Remote Ranks (2 and 3) on Pod B via SSH
        for rank in [2, 3]:
            # Map global ranks 2,3 to local ranks 0,1 on the remote machine
            local_r = rank - 2 
            ssh_cmd = [
                "ssh", "-p", POD_B_PORT, POD_B_IP,
                f"cd {os.getcwd()} && "
                f"RANK={rank} LOCAL_RANK={local_r} WORLD_SIZE={WORLD_SIZE} "
                f"MASTER_ADDR={MASTER_ADDR} MASTER_PORT={MASTER_PORT} "
                f"PYTHONUNBUFFERED=1 "
                f"/workspace/pytorch_env/bin/python3 -u src/training/trainer.py "
                f"--phase {args.phase} --local_rank {local_r}"
            ]
            log_f = open(f"models/cuda_deep_sat/rank_{rank}.log", "wb", buffering=0)
            p = subprocess.Popen(ssh_cmd, stdout=log_f, stderr=log_f)
            processes.append((p, log_f))
            print(f"[Launcher] ORACLE: Started Remote Rank {rank} on Pod B GPU {local_r}")

        try:
            for p, log in processes:
                p.wait()
                log.close()
            print("EXTREME RUN COMPLETE")
        except:
            for p, _ in processes: p.terminate()
        return

    print("\n" + "="*60)
    print(" STARTING FINAL PLANTCLEF 2026 STUDENT RUN")
    print("="*60 + "\n")

    # 2. Set Hardware Environment Variables
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_SILENT"] = "True"
    
    if args.extreme:
        print("[Launcher] Enabling EXTREME Blackwell Optimizations...")
        os.environ["PLANTCLEF_EXTREME_MODE"] = "True"
    
    if args.dual:
        print("[Launcher] Enabling DUAL-GPU Optimizations (2x RTX PRO 6000)...")
        os.environ["PLANTCLEF_DUAL_MODE"] = "True"
        # Auto-detect dual-GPU count if not explicitly set
        if args.num_gpus == 1:
            args.num_gpus = 2

    if args.tpu:
        print("[Launcher] Targeting Google TPU (XLA)...")
        os.environ["PLANTCLEF_HARDWARE"] = "TPU"
        cmd = [sys.executable, "src/training/trainer.py"]
        subprocess.run(cmd, check=True)
    else:
        print(f"[Launcher] Targeting NVIDIA CUDA ({args.num_gpus} GPUs - ORACLE Isolated Mode)")
        os.environ["PLANTCLEF_HARDWARE"] = "CUDA"
        
        # ORACLE: Absolute Process-Level VRAM Isolation
        # Launching separate processes with unique CUDA_VISIBLE_DEVICES
        processes = []
        import time
        for rank in range(args.num_gpus):
            # ORACLE: Hard-Staggered Launch (30s head start for Rank 0)
            # This allows the master to finish the Disk Scan and create 
            # the integrity cache before Rank 1 tries to read it.
            if rank > 0:
                print(f"[Launcher] ORACLE: Rank 0 head-start... Rank {rank} waiting 30s.")
                time.sleep(30) 
                
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(rank)
            env["LOCAL_RANK"] = str(rank) 
            env["RANK"] = str(rank)
            
            # ORACLE: Inherit world size and offset from terminal if present
            env["WORLD_SIZE"] = os.environ.get("WORLD_SIZE", str(args.num_gpus))
            if "RANK_OFFSET" in os.environ:
                env["RANK_OFFSET"] = os.environ["RANK_OFFSET"]
                
            env["MASTER_ADDR"] = args.master_addr
            env["MASTER_PORT"] = str(args.master_port)
            env["PYTHONUNBUFFERED"] = "1" # ORACLE: Force immediate log flushing
            
            # Ensure Python path includes src
            env["PYTHONPATH"] = f"{root_dir}:{src_dir}"
            
            cmd = [sys.executable, "-u", "src/training/trainer.py", "--phase", args.phase, "--local_rank", str(rank)]
            
            # ORACLE: Force unbuffered binary stream for real-time logging
            log_file = open(f"models/cuda_deep_sat/rank_{rank}.log", "wb", buffering=0)
            p = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=log_file)
            processes.append((p, log_file))
            print(f"[Launcher] ORACLE: Started Rank {rank} (Unbuffered Log: models/cuda_deep_sat/rank_{rank}.log)")
            
        try:
            print(f"[Launcher] ORACLE: Cluster monitoring active. Watch logs for progress.")
            active_p = processes.copy()
            while active_p:
                for item in active_p[:]:
                    p, log = item
                    if p.poll() is not None: # Process finished or crashed
                        log.close()
                        if p.returncode != 0:
                            print(f"\n" + "!"*60)
                            print(f" CRITICAL FAILURE: Rank {processes.index(item)} crashed!")
                            print("!"*60)
                            # Print last 20 lines of the specific log
                            log_path = f"models/cuda_deep_sat/rank_{processes.index(item)}.log"
                            with open(log_path, "r") as f:
                                lines = f.readlines()
                                print("".join(lines[-20:]))
                            raise subprocess.CalledProcessError(p.returncode, p.args)
                        active_p.remove(item)
                time.sleep(1)

            print("EXTREME RUN COMPLETE")

            print(" FINAL RUN COMPLETED SUCCESSFULLY")
            print("="*60)
        except Exception as e:
            print(f"\n[Launcher CRITICAL] Process failure: {e}")
            for p, log in processes: 
                try: p.terminate()
                except: pass
            sys.exit(1)


if __name__ == "__main__":
    main()
