import pandas as pd
import re
from pathlib import Path

input_path = Path("./data/species_lookup_with_gbif.csv")
output_path = Path("./data/species_lookup_with_gbif_cleaned_names.csv")

df = pd.read_csv(input_path)

def normalize_spaces(text):
    if pd.isna(text):
        return pd.NA
    text = str(text).replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else pd.NA

def clean_scientific_name(raw, canonical=None):
    canonical = normalize_spaces(canonical)
    if canonical is not pd.NA and canonical is not None and str(canonical) != "<NA>":
        return canonical

    raw = normalize_spaces(raw)
    if raw is pd.NA or raw is None or str(raw) == "<NA>":
        return pd.NA

    text = str(raw)

    # Mild authorship stripping fallback:
    # keep canonical taxon part; remove trailing botanical author citation.
    patterns = [
        # species + optional infra rank/epithet
        r"^([A-Z][a-zA-Z-]+(?:\s+[a-z][a-zA-Z-]+){1,2}(?:\s+(?:subsp\.|var\.|f\.)\s+[a-z][a-zA-Z-]+)?)\b.*$",
    ]
    for pat in patterns:
        m = re.match(pat, text)
        if m:
            cleaned = normalize_spaces(m.group(1))
            if cleaned is not pd.NA and str(cleaned) != "<NA>":
                return cleaned

    return text

def sentence_case_phrase(name):
    name = normalize_spaces(name)
    if name is pd.NA or name is None or str(name) == "<NA>":
        return None
    text = str(name)
    # keep acronyms as-is, otherwise use sentence case
    if text.isupper() and len(text) <= 5:
        return text
    return text[:1].upper() + text[1:].lower()

def clean_pipe_list(text, sentence_case=False):
    text = normalize_spaces(text)
    if text is pd.NA or text is None or str(text) == "<NA>":
        return pd.NA

    seen = set()
    cleaned = []
    for part in str(text).split("|"):
        item = normalize_spaces(part)
        if item is pd.NA or item is None or str(item) == "<NA>":
            continue
        item = str(item)
        if sentence_case:
            item = sentence_case_phrase(item)
        key = item.casefold()
        if key not in seen:
            seen.add(key)
            cleaned.append(item)

    return " | ".join(cleaned) if cleaned else pd.NA

# Add cleaned columns without overwriting originals
df["species_name_clean"] = [
    clean_scientific_name(raw, canonical)
    for raw, canonical in zip(df.get("species"), df.get("gbif_canonical_name"))
]

df["gbif_scientific_name_clean"] = [
    clean_scientific_name(raw, canonical)
    for raw, canonical in zip(df.get("gbif_scientific_name"), df.get("gbif_canonical_name"))
]

df["gbif_synonyms_clean"] = df.get("gbif_synonyms", pd.Series([pd.NA] * len(df))).apply(
    lambda x: clean_pipe_list(x, sentence_case=False)
)

df["gbif_common_names_en_clean"] = df.get("gbif_common_names_en", pd.Series([pd.NA] * len(df))).apply(
    lambda x: clean_pipe_list(x, sentence_case=True)
)

df["gbif_common_names_all_clean"] = df.get("gbif_common_names_all", pd.Series([pd.NA] * len(df))).apply(
    lambda x: clean_pipe_list(x, sentence_case=False)
)

def first_pipe_item(text):
    text = normalize_spaces(text)
    if text is pd.NA or text is None or str(text) == "<NA>":
        return pd.NA
    first = str(text).split("|")[0].strip()
    return first if first else pd.NA

df["primary_common_name_en"] = df["gbif_common_names_en_clean"].apply(first_pipe_item)
df["primary_common_name_any"] = df["gbif_common_names_all_clean"].apply(first_pipe_item)

# Put cleaned columns near the original naming fields
preferred_order = [
    "species_id",
    "gbif_species_id",
    "species",
    "species_name_clean",
    "gbif_scientific_name",
    "gbif_scientific_name_clean",
    "gbif_canonical_name",
    "gbif_synonyms",
    "gbif_synonyms_clean",
    "gbif_common_names_en",
    "gbif_common_names_en_clean",
    "primary_common_name_en",
    "gbif_common_names_all",
    "gbif_common_names_all_clean",
    "primary_common_name_any",
]
other_cols = [c for c in df.columns if c not in preferred_order]
df = df[[c for c in preferred_order if c in df.columns] + other_cols]

df.to_csv(output_path, index=False)

sample_cols = [
    c for c in [
        "species_id",
        "species",
        "species_name_clean",
        "gbif_common_names_en",
        "gbif_common_names_en_clean",
        "primary_common_name_en",
    ] if c in df.columns
]
sample = df[sample_cols].head(8)

print(f"Saved cleaned file to: {output_path}")
print(sample.to_string(index=False))
