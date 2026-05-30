import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


PROMPT_TEMPLATES = [
    "a photo of a {}",
    "a close-up photo of a {}",
    "a photo of the {}",
]


def load_species(mapping_path):
    df = pd.read_csv(mapping_path, sep=";", quotechar='"', dtype=str)
    species_ids = df["species_id"].tolist()
    species_names = df["species"].tolist()
    id_to_name = dict(zip(species_ids, species_names))
    return species_ids, species_names, id_to_name


def get_tiles(image, tile_size, stride):
    w, h = image.size
    tiles = []
    coords = []
    y = 0
    while y + tile_size <= h or y == 0:
        x = 0
        while x + tile_size <= w or x == 0:
            x2 = min(x + tile_size, w)
            y2 = min(y + tile_size, h)
            x1_actual = max(0, x2 - tile_size)
            y1_actual = max(0, y2 - tile_size)
            tile = image.crop((x1_actual, y1_actual, x2, y2))
            if tile.size[0] < tile_size or tile.size[1] < tile_size:
                tile = tile.resize((tile_size, tile_size), Image.BILINEAR)
            tiles.append(tile)
            coords.append((x1_actual, y1_actual, x2, y2))
            if x + stride >= w:
                break
            x += stride
        if y + stride >= h:
            break
        y += stride
    return tiles, coords


def encode_text_features(model, tokenizer, species_names, device, batch_size=256):
    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(species_names), batch_size):
            batch_names = species_names[i:i + batch_size]
            text_list = []
            for name in batch_names:
                for tmpl in PROMPT_TEMPLATES:
                    text_list.append(tmpl.format(name))
            tokens = tokenizer(text_list).to(device)
            emb = model.encode_text(tokens)
            emb = F.normalize(emb, dim=-1)
            emb = emb.view(len(batch_names), len(PROMPT_TEMPLATES), -1)
            emb = emb.mean(dim=1)
            emb = F.normalize(emb, dim=-1)
            all_embeddings.append(emb.cpu())
    return torch.cat(all_embeddings, dim=0)  # (n_species, dim)


def encode_image_tiles(model, transform, tiles, device, batch_size=64):
    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(tiles), batch_size):
            batch = torch.stack([transform(t) for t in tiles[i:i + batch_size]]).to(device)
            emb = model.encode_image(batch)
            emb = F.normalize(emb, dim=-1)
            all_embeddings.append(emb.cpu())
    return torch.cat(all_embeddings, dim=0)  # (n_tiles, dim)


def compute_tile_logits(image_feats, text_feats, logit_scale):
    # Returns raw logits: (n_tiles, n_species)
    return logit_scale * image_feats @ text_feats.T


def aggregate_tile_logits(tile_logits):
    # SAHI-style max pooling over tiles -> (n_species,)
    return tile_logits.max(dim=0).values


def tile_top_k(tile_logits, k=3):
    # Per-tile top-k using softmax probabilities for the debug CSV scores
    probs = tile_logits.softmax(dim=-1)
    top = probs.topk(k, dim=-1)
    return top.indices.numpy(), top.values.numpy()


def image_top_k(image_logits, species_ids, k=5):
    # Top-k species ids from aggregated image logits
    top = image_logits.topk(k)
    top_species_ids = [species_ids[i] for i in top.indices.tolist()]
    top_scores = top.values.tolist()
    return top_species_ids, top_scores
