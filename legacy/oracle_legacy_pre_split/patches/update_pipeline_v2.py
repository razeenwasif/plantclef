path = 'phases/inference/pipeline.py'
with open(path, 'r') as f:
    lines = f.readlines()

# 1. Add imports and vegetation filter functions
header_end = 0
for i, line in enumerate(lines):
    if line.startswith("def _generate_tiles"):
        header_end = i
        break

veg_filter_code = """
# ── Vegetation Filter (Winner Recipe) ──────────────────────────────────────────

def _exg_vegetation_mask(arr: np.ndarray, exg_thresh: float = 20.0) -> np.ndarray:
    \"\"\"ExG = 2G - R - B. Works on [0, 1] float32 as well (scaled internally).\"\"\"
    # Scale to [0, 255] for standard ExG tuning
    r = arr[0] * 255.0
    g = arr[1] * 255.0
    b = arr[2] * 255.0
    exg = 2 * g - r - b
    return (exg > exg_thresh) & (g > r) & (g > b)

def _filter_tiles(tiles: List[torch.Tensor], min_frac: float = 0.15) -> List[torch.Tensor]:
    if min_frac <= 0 or not tiles:
        return tiles
    
    fracs = []
    for t in tiles:
        mask = _exg_vegetation_mask(t.numpy())
        fracs.append(mask.mean())
    
    kept = [t for t, f in zip(tiles, fracs) if f >= min_frac]
    if not kept:
        # Fallback: take top 25% most green tiles
        idx = np.argsort(fracs)[-max(1, len(tiles)//4):]
        kept = [tiles[i] for i in idx]
    return kept

"""

lines.insert(header_end, veg_filter_code)

# 2. Update _infer_one_image to use filter and HFlip TTA
content = "".join(lines)

old_infer_block = """        # Tile generation: SAM-based if enabled, else multi-scale grid.
        if cfg.use_sam_tiling and sam_generator is not None and img_pil is not None:
            tiles = _sam_generate_tiles(img_pil, sam_generator, cfg)
        elif cfg.tiling_enabled:
            tiles = _generate_tiles(img, cfg.tile_size, cfg.tile_overlap, cfg.scales)
        else:
            tiles = [img]

        per_model_logits: List[torch.Tensor] = []"""

new_infer_block = """        # Tile generation: SAM-based if enabled, else multi-scale grid.
        if cfg.use_sam_tiling and sam_generator is not None and img_pil is not None:
            tiles = _sam_generate_tiles(img_pil, sam_generator, cfg)
        elif cfg.tiling_enabled:
            tiles = _generate_tiles(img, cfg.tile_size, cfg.tile_overlap, cfg.scales)
        else:
            tiles = [img]
        
        # plantclef: Apply Vegetation Filter
        if cfg.min_vegetation_frac > 0:
            tiles = _filter_tiles(tiles, cfg.min_vegetation_frac)

        per_model_logits: List[torch.Tensor] = []"""

content = content.replace(old_infer_block, new_infer_block)

# 3. Add HFlip TTA logic
old_batch_block = """                with torch.inference_mode():
                    if faiss_index is not None:
                        l_t, f_t = model(resized, return_features=True)
                        all_logits_T.append(l_t.float().cpu())
                        all_feats_T.append(f_t.float().cpu())
                    else:
                        l_t = model(resized)
                        all_logits_T.append(l_t.float().cpu())"""

new_batch_block = """                with torch.inference_mode():
                    if faiss_index is not None:
                        l_t, f_t = model(resized, return_features=True)
                        if cfg.use_hflip_tta:
                            l_flipped = model(torch.flip(resized, dims=[3]))
                            l_t = (l_t + l_flipped) / 2.0
                        all_logits_T.append(l_t.float().cpu())
                        all_feats_T.append(f_t.float().cpu())
                    else:
                        l_t = model(resized)
                        if cfg.use_hflip_tta:
                            l_flipped = model(torch.flip(resized, dims=[3]))
                            l_t = (l_t + l_flipped) / 2.0
                        all_logits_T.append(l_t.float().cpu())"""

content = content.replace(old_batch_block, new_batch_block)

with open(path, 'w') as f:
    f.write(content)
