path = 'phases/inference/pipeline.py'
with open(path, 'r') as f:
    content = f.read()

old_block = """    for img_id, img_path in my_images:
        try:
            img_pil = Image.open(img_path).convert("RGB")
            arr = np.array(img_pil, dtype=np.float32) / 255.0
            if cfg.use_retinex and cfg.rank == 0:
                arr = _retinex_normalize(arr)
            img = torch.from_numpy(arr).permute(2, 0, 1)
        except Exception as e:
            print(f"[Inference] Skipping {img_path}: {e}")
            continue
        agg_logits = _infer_one_image(img, img_pil=img_pil)
        results[img_id] = _postprocess(
            agg_logits.cpu(), cfg, species_ids,
            per_class_thr=per_class_thr, q_hat=q_hat,
            pav_solver=pav_solver, taxon_filter=taxon_filter, fw_solver=fw_solver,
        )
        n_done += 1
        if n_done % 100 == 0:
            print(f"[Inference] GPU {cfg.rank}: {n_done} images processed")"""

new_block = """    pbar = None
    if cfg.rank == 0:
        pbar = tqdm(total=len(my_images), desc=f"[Inference] Rank {cfg.rank}")

    for img_id, img_path in my_images:
        try:
            img_pil = Image.open(img_path).convert("RGB")
            arr = np.array(img_pil, dtype=np.float32) / 255.0
            if cfg.use_retinex and cfg.rank == 0:
                arr = _retinex_normalize(arr)
            img = torch.from_numpy(arr).permute(2, 0, 1)
        except Exception as e:
            print(f"[Inference] Skipping {img_path}: {e}")
            continue
        agg_logits = _infer_one_image(img, img_pil=img_pil)
        results[img_id] = _postprocess(
            agg_logits.cpu(), cfg, species_ids,
            per_class_thr=per_class_thr, q_hat=q_hat,
            pav_solver=pav_solver, taxon_filter=taxon_filter, fw_solver=fw_solver,
        )
        n_done += 1
        if pbar is not None:
            pbar.update(1)
        elif n_done % 100 == 0:
            print(f"[Inference] GPU {cfg.rank}: {n_done} images processed")
    
    if pbar is not None:
        pbar.close()"""

content = content.replace(old_block, new_block)
with open(path, 'w') as f:
    f.write(content)
