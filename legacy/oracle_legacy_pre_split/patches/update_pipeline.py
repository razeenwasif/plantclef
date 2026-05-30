import os
import csv
import torch
import torch.distributed as dist

path = 'phases/inference/pipeline.py'
with open(path, 'r') as f:
    content = f.read()

# 1. Update run_inference signature and distributed init
old_func_start = 'def run_inference(cfg: InferenceConfig) -> None:'
new_func_start = 'def run_inference(cfg: InferenceConfig) -> None:\n    if cfg.world_size > 1 and not dist.is_initialized():\n        dist.init_process_group(backend="nccl")'
content = content.replace(old_func_start, new_func_start)

# 2. Update Retinex log to Rank 0 only
content = content.replace('if cfg.use_retinex:', 'if cfg.use_retinex and cfg.rank == 0:')

# 3. Distributed Sharding logic
old_loop_start = '    for img_id, img_path in _iter_test_images(cfg):'
new_loop_start = """
    # plantclef: Distributed sharding of test images
    all_test_images = list(_iter_test_images(cfg))
    if cfg.world_size > 1:
        my_images = all_test_images[cfg.rank::cfg.world_size]
        if cfg.rank == 0:
            print(f"[Inference] Sharding {len(all_test_images)} total images -> ~{len(my_images)} per GPU.")
    else:
        my_images = all_test_images

    for img_id, img_path in my_images:"""
content = content.replace(old_loop_start, new_loop_start)

# 4. Update progress log to include GPU rank
content = content.replace('print(f"[Inference] {n_done} images processed")', 
                          'print(f"[Inference] GPU {cfg.rank}: {n_done} images processed")')

# 5. Distributed Gathering and conditional Rank 0 write
old_csv_write = """    # Submission CSV
    os.makedirs(os.path.dirname(os.path.abspath(cfg.submission_csv)) or ".", exist_ok=True)
    with open(cfg.submission_csv, "w", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL, lineterminator="\\n")
        writer.writerow(["quadrat_id", "species_ids"])
        for img_id, preds in results.items():
            writer.writerow([img_id, f"[{', '.join(preds)}]"])

    print(f"[Inference] Written {len(results)} predictions → {cfg.submission_csv}")"""

new_csv_write = """    # plantclef: Gather all results across GPUs
    if cfg.world_size > 1:
        dist.barrier()
        gathered_list = [None] * cfg.world_size
        dist.all_gather_object(gathered_list, results)
        
        # Merge all dictionaries into one on Rank 0
        if cfg.rank == 0:
            merged_results = {}
            for r_dict in gathered_list:
                merged_results.update(r_dict)
            results = merged_results

    # Submission CSV — only Rank 0 writes
    if cfg.rank == 0:
        os.makedirs(os.path.dirname(os.path.abspath(cfg.submission_csv)) or ".", exist_ok=True)
        with open(cfg.submission_csv, "w", newline="") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL, lineterminator="\\n")
            writer.writerow(["quadrat_id", "species_ids"])
            for img_id, preds in results.items():
                writer.writerow([img_id, f"[{', '.join(preds)}]"])

        print(f"[Inference] Written {len(results)} predictions → {cfg.submission_csv}")
    
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()"""

content = content.replace(old_csv_write, new_csv_write)

with open(path, 'w') as f:
    f.write(content)
