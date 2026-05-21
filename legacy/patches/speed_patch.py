import os

path = 'phases/inference/pipeline.py'
with open(path, 'r') as f:
    content = f.read()

# Implement tile mini-batching inside _infer_one_image to saturate B200
old_forward = """            resized = torch.stack([
                F.interpolate(t.unsqueeze(0), size=(native_res, native_res),
                              mode="bilinear", align_corners=False).squeeze(0)
                for t in tiles
            ])
            with torch.inference_mode():
                if faiss_index is not None:
                    logits_T, feats_T = model(resized.to(device), return_features=True)
                    # Aggregate features across tiles (mean) for the FAISS query.
                    all_query_features.append(feats_T.mean(dim=0))   # [feat_dim]
                else:
                    logits_T = model(resized.to(device))             # [T, C]
            per_model_logits.append(_aggregate(logits_T.float(), cfg.aggregation))"""

new_forward = """            # ORACLE: B200 Optimized Tile Batching
            all_logits_T = []
            all_feats_T  = []
            for i in range(0, len(tiles), cfg.batch_size):
                batch_tiles = tiles[i : i + cfg.batch_size]
                resized = torch.stack([
                    F.interpolate(t.unsqueeze(0), size=(native_res, native_res),
                                  mode="bilinear", align_corners=False).squeeze(0)
                    for t in batch_tiles
                ]).to(device)
                
                with torch.inference_mode():
                    if faiss_index is not None:
                        l_t, f_t = model(resized, return_features=True)
                        all_logits_T.append(l_t.float().cpu())
                        all_feats_T.append(f_t.float().cpu())
                    else:
                        l_t = model(resized)
                        all_logits_T.append(l_t.float().cpu())
            
            logits_T = torch.cat(all_logits_T, dim=0)
            if faiss_index is not None:
                all_query_features.append(torch.cat(all_feats_T, dim=0).mean(dim=0).to(device))
            
            per_model_logits.append(_aggregate(logits_T.to(device), cfg.aggregation))"""

content = content.replace(old_forward, new_forward)

with open(path, 'w') as f:
    f.write(content)
