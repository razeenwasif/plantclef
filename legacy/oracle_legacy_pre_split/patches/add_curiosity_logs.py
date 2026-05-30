path = 'phases/inference/pipeline.py'
with open(path, 'r') as f:
    content = f.read()

# 1. Update _postprocess signature and logic
old_post_sig = "def _postprocess(logits: torch.Tensor, cfg: InferenceConfig,"
new_post_sig = "def _postprocess(logits: torch.Tensor, cfg: InferenceConfig, return_stats=False,"

content = content.replace(old_post_sig, new_post_sig)

# We need to capture the end of _postprocess
old_post_end = "return [species_ids[i] for i in above[:cfg.top_k]]"
new_post_end = """    preds = [species_ids[i] for i in above[:cfg.top_k]]
    if return_stats:
        return preds, stats
    return preds"""
content = content.replace(old_post_end, new_post_end)

# Insert stats gathering into _postprocess
stats_init = """    stats = {"pav": 0, "ac3": 0, "fw": 0}
    probs = torch.sigmoid(logits)"""
content = content.replace("probs = torch.sigmoid(logits)", stats_init)

# PAV stats
pav_logic = """        try:
            old_p = probs.clone()
            probs = pav_solver.solve(probs)
            stats["pav"] = int((probs != old_p).sum().item())"""
content = content.replace("        try:\n            probs = pav_solver.solve(probs)", pav_logic)

# AC3 stats
ac3_logic = """        try:
            scores_np = probs.detach().cpu().numpy().astype(np.float32)
            filtered  = taxon_filter.filter_predictions(scores_np)
            new_p     = torch.from_numpy(np.asarray(filtered)).to(probs.device).float()
            stats["ac3"] = int((probs > 0).sum().item() - (new_p > 0).sum().item())
            probs     = new_p"""
content = content.replace("""        try:
            scores_np = probs.detach().cpu().numpy().astype(np.float32)
            filtered  = taxon_filter.filter_predictions(scores_np)
            probs     = torch.from_numpy(np.asarray(filtered)).to(probs.device).float()""", ac3_logic)

# FW stats
fw_logic = """            sel = fw_solver.solve(probs.float(), max_iters=5, sparsity_k=cfg.fw_top_k)
            stats["fw"] = int((sel > 1e-3).sum().item())
            above = torch.nonzero(sel > 1e-3).squeeze(-1).tolist()"""
content = content.replace("""            sel = fw_solver.solve(probs.float(), max_iters=5, sparsity_k=cfg.fw_top_k)
            above = torch.nonzero(sel > 1e-3).squeeze(-1).tolist()""", fw_logic)

# 2. Update _infer_one_image to handle Retinex stats
content = content.replace("agg_logits, ttt_loss", "agg_logits, ttt_loss, retinex_shift")
content = content.replace("retinex_shift = 0.0", "retinex_shift = 0.0\n        if cfg.use_retinex: pass # placeholder") # will handle in loop

# 3. Update main loop to display everything
old_loop_post = """        agg_logits, current_loss = _infer_one_image(img, img_pil=img_pil)
        results[img_id] = _postprocess(
            agg_logits.cpu(), cfg, species_ids,
            per_class_thr=per_class_thr, q_hat=q_hat,
            pav_solver=pav_solver, taxon_filter=taxon_filter, fw_solver=fw_solver,
        )"""

new_loop_post = """        agg_logits, current_loss, retinex_shift = _infer_one_image(img, img_pil=img_pil)
        preds, pp_stats = _postprocess(
            agg_logits.cpu(), cfg, species_ids,
            per_class_thr=per_class_thr, q_hat=q_hat,
            pav_solver=pav_solver, taxon_filter=taxon_filter, fw_solver=fw_solver,
            return_stats=True
        )
        results[img_id] = preds"""
content = content.replace(old_loop_post, new_loop_post)

old_pbar = 'pbar.set_postfix(loss=f"{current_loss:.4f}")'
new_pbar = 'pbar.set_postfix(loss=f"{current_loss:.2f}", retx=f"{retinex_shift:.3f}", pav=pp_stats["pav"], ac3=pp_stats["ac3"], fw=pp_stats["fw"])'
content = content.replace(old_pbar, new_pbar)

# Retinex shift calculation
ret_old = """            if cfg.use_retinex:
                arr = _retinex_normalize(arr)"""
ret_new = """            retinex_shift = 0.0
            if cfg.use_retinex:
                old_arr = arr.copy()
                arr = _retinex_normalize(arr)
                retinex_shift = float(np.abs(arr - old_arr).mean())"""
content = content.replace(ret_old, ret_new)

# Fix _infer_one_image return inside the function too
content = content.replace("return agg_logits, ttt_loss, retinex_shift", "return agg_logits, ttt_loss, 0.0") # pass 0.0 as retinex handled outside

with open(path, 'w') as f:
    f.write(content)
