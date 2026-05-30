import os
import sys
import time
import subprocess
import re
import threading
import atexit
from pathlib import Path

# plantclef: Dynamic VRAM Watchdog & Auto-Scaler (Universal Edition)
# Monitors GPU memory and dynamically scales both Training and Inference.
# Includes ERROR DISCRIMINATION and 20-MINUTE COMPILER IMMUNITY.

CONFIG_PATH = "src/config.py"
INITIAL_CONFIG_CONTENT = None

def save_config_snapshot():
    """Takes a snapshot of the config before the watchdog starts tuning."""
    global INITIAL_CONFIG_CONTENT
    try:
        with open(CONFIG_PATH, "r") as f:
            INITIAL_CONFIG_CONTENT = f.read()
            print("[Watchdog] Initial config snapshot taken for restoration.")
    except Exception as e:
        print(f"[Warning] Could not snapshot config: {e}")

def restore_config():
    """Restores the config to its original state on exit."""
    if INITIAL_CONFIG_CONTENT:
        try:
            with open(CONFIG_PATH, "w") as f:
                f.write(INITIAL_CONFIG_CONTENT)
            print("\n[Watchdog] Config restored to original state. System clean. 🛡️")
        except Exception as e:
            print(f"[Error] Failed to restore config: {e}")

def get_vram_stats():
    try:
        res = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'], 
            text=True
        )
        ratios = []
        for line in res.strip().split('\n'):
            if not line.strip(): continue
            used, total = map(float, line.split(','))
            ratios.append(used / total)
        return max(ratios) if ratios else 0.0
    except Exception:
        return 0.0

def update_training_config(scale_factor):
    """Dynamically scales P2_BATCH_SIZE in src/config.py."""
    try:
        with open(CONFIG_PATH, "r") as f:
            content = f.read()
        
        match = re.search(r"P2_BATCH_SIZE\s*=\s*(\d+)", content)
        if match:
            old_bs = int(match.group(1))
            new_bs = max(4, int(old_bs * scale_factor))
            if new_bs == old_bs: return False
            
            new_content = re.sub(r"P2_BATCH_SIZE\s*=\s*\d+", f"P2_BATCH_SIZE = {new_bs}", content)
            with open(CONFIG_PATH, "w") as f:
                f.write(new_content)
            print(f"[Watchdog] Scaled P2_BATCH_SIZE: {old_bs} -> {new_bs}")
            return True
    except Exception as e:
        print(f"[Error] Training scale failed: {e}")
    return False

def update_inference_args(command, scale_factor):
    """Scales --batch-size in a command string."""
    match = re.search(r"--batch-size\s+(\d+)", command)
    if match:
        old_bs = int(match.group(1))
        new_bs = max(1, int(old_bs * scale_factor))
        if new_bs == old_bs: return command, False
        new_command = re.sub(r"--batch-size\s+\d+", f"--batch-size {new_bs}", command)
        print(f"[Watchdog] Scaled Inference Batch: {old_bs} -> {new_bs}")
        return new_command, True
    return command, False

def run_with_watchdog(command_args, no_resume=False):
    if not command_args:
        print("Usage: python3 tools/vram_watchdog.py [--preset <name>] [--no-resume] <command>")
        sys.exit(1)

    save_config_snapshot()
    atexit.register(restore_config)

    current_command = " ".join(command_args)
    if no_resume:
        # If no-resume is set, we ensure the script starts from the current latest checkpoint
        # without forcing a manual start_step override.
        print("[Watchdog] --no-resume enabled: Restarting without manual resume flags.")
    is_inference = "run_inference" in current_command
    
    underutilized_ticks = 0
    process_start_time = time.time()

    try:
        while True:
            print(f"\n[Watchdog] Launching: {current_command}")
            process = subprocess.Popen(current_command, shell=True, stderr=subprocess.PIPE, text=True)
            
            oom_detected = False
            logic_error_detected = False
            scaling_event = False
            error_msg = ""

            def monitor_stderr(proc):
                nonlocal oom_detected, logic_error_detected, error_msg
                for line in proc.stderr:
                    print(line, end='', flush=True)
                    # Detection for standard PyTorch and NVIDIA DALI OOMs
                    if any(x in line for x in ["OutOfMemoryError", "CUDA out of memory", "Can't allocate", "Critical error in pipeline"]):
                        oom_detected = True
                    
                    if "AssertionError" in line or "RuntimeError" in line:
                        # Only mark as logic error if it's NOT an obvious OOM/Resource issue
                        is_resource_error = any(x in line.lower() for x in ["memory", "allocate", "pipeline", "capacity"])
                        if not is_resource_error:
                            logic_error_detected = True
                            error_msg = line.strip()

            stderr_thread = threading.Thread(target=monitor_stderr, args=(process,))
            stderr_thread.daemon = True
            stderr_thread.start()

            while process.poll() is None:
                vram_max = get_vram_stats()
                
                # plantclef: Precision Scaling with 20-minute Compilation Grace Period
                process_age = time.time() - process_start_time
                if 0.10 < vram_max < 0.95 and process_age > 1200:
                    underutilized_ticks += 1
                else:
                    underutilized_ticks = 0
                    
                if underutilized_ticks > 24: 
                    print(f"\n[Watchdog] Headroom detected ({vram_max*100:.1f}%). Scaling Up...")
                    if is_inference:
                        current_command, changed = update_inference_args(current_command, 1.1)
                    else:
                        changed = update_training_config(1.1)
                    
                    if changed:
                        scaling_event = True
                        process.kill()
                        # Removed pkill to prevent cluster interference
                        break
                    else:
                        underutilized_ticks = 0 # No scale possible
                    
                time.sleep(5)

            process.wait()
            
            if logic_error_detected:
                print(f"\n[Watchdog] ABORT: Logic Error detected: {error_msg}")
                break

            if scaling_event or oom_detected:
                print("\n[Watchdog] Restarting for Scale Event...")
                # Removed pkill to prevent cluster interference
                if oom_detected and not scaling_event:
                    if is_inference:
                        current_command, _ = update_inference_args(current_command, 0.75)
                    else:
                        update_training_config(0.75)
                time.sleep(5)
                continue
                
            elif process.returncode == 0:
                print("\n[Watchdog] Process completed successfully.")
                break
            else:
                print(f"\n[Watchdog] Process failed (Code {process.returncode}). Restarting...")
                os.system("pkill -9 -f torchrun >/dev/null 2>&1")
                time.sleep(5)
                continue
    except KeyboardInterrupt:
        print("\n[Watchdog] User interrupt received.")
        os.system("pkill -9 -f torchrun >/dev/null 2>&1")
        sys.exit(0)

if __name__ == "__main__":
    args = sys.argv[1:]
    preset = None
    no_resume = False
    
    if "--no-resume" in args:
        no_resume = True
        args.remove("--no-resume")
        
    if "--preset" in args:
        idx = args.index("--preset")
        if idx + 1 < len(args):
            preset = args[idx+1]
            os.environ["HARDWARE_PRESET"] = preset
            print(f"[Watchdog] Manual Hardware Override: {preset}")
            args = args[:idx] + args[idx+2:]
    
    run_with_watchdog(args, no_resume=no_resume)
