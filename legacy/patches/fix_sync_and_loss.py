path = 'phases/inference/pipeline.py'
with open(path, 'r') as f:
    content = f.read()

# 1. Fix Retinex rank gating
content = content.replace('if cfg.use_retinex and cfg.rank == 0:', 'if cfg.use_retinex:')

# 2. Update _infer_one_image to return loss
content = content.replace('return agg_logits', 'return agg_logits, ttt_loss')
content = content.replace('agg_logits = _infer_one_image(img, img_pil=img_pil)', 'agg_logits, current_loss = _infer_one_image(img, img_pil=img_pil)')

# 3. Add loss capture logic in _infer_one_image
# We need to capture the loss from the TTT adaptor
old_ttt_call = """        # --- TTT Adaptation ---
        if ttt_adaptor is not None and img_pil is not None:
            ttt_adaptor.adapt(img_pil)"""

new_ttt_call = """        # --- TTT Adaptation ---
        ttt_loss = 0.0
        if ttt_adaptor is not None and img_pil is not None:
            ttt_loss = ttt_adaptor.adapt(img_pil)"""
content = content.replace(old_ttt_call, new_ttt_call)

# 4. Update pbar postfix
old_pbar_update = """        n_done += 1
        if pbar is not None:
            pbar.update(1)"""

new_pbar_update = """        n_done += 1
        if pbar is not None:
            pbar.set_postfix(loss=f"{current_loss:.4f}")
            pbar.update(1)"""
content = content.replace(old_pbar_update, new_pbar_update)

with open(path, 'w') as f:
    f.write(content)

# 5. Update ttt.py to return the loss
ttt_path = 'phases/inference/ttt.py'
with open(ttt_path, 'r') as f:
    ttt_content = f.read()
ttt_content = ttt_content.replace('print(f"[TTT] Adapted model to quadrat (Loss: {loss.item():.4f}, Res: {res}px)")', 'return loss.item()')
# Handle case where return might be missing if trainable_params is empty
ttt_content = ttt_content.replace('if not trainable_params: return', 'if not trainable_params: return 0.0')

with open(ttt_path, 'w') as f:
    f.write(ttt_content)
