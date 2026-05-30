import json
import os
from typing import Dict, Any, List

def audit_traits() -> None:
    """
    Performs an audit of the botanical traits JSONL file.

    Analyzes the quality and distribution of extracted botanical traits,
    checking for JSON corruption, empty outputs, and presence of
    predefined vocabulary keywords across multiple categories.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    trait_json = "/workspace/plantclef/processed/botanical_traits.jsonl"
    if not os.path.exists(trait_json):
        print(f"Error: {trait_json} not found.")
        return

    stats = {
        "total_lines": 0,
        "json_corrupt": 0,
        "empty_output": 0,
        "valid_traits": 0,
        "category_counts": {
            "leaf_shape": 0,
            "phyllotaxy": 0,
            "flower_color": 0,
            "inflorescence": 0,
            "stem_type": 0
        }
    }

    # Vocabulary keywords to look for
    vocabs = {
        "leaf_shape": ["ovate", "lanceolate", "cordate", "elliptic", "linear", "oblong", "peltate", "palmate", "sagittate", "spatulate", "deltoid", "needle-like"],
        "phyllotaxy": ["alternate", "opposite", "whorled", "basal"],
        "flower_color": ["yellow", "white", "blue", "red", "purple", "pink", "orange", "green"],
        "inflorescence": ["spike", "umbel", "cyme", "raceme", "panicle", "solitary"],
        "stem_type": ["woody", "herbaceous", "succulent", "vine", "climbing"]
    }

    print(f"Auditing {trait_json}...")
    
    with open(trait_json, 'r') as f:
        for line in f:
            stats["total_lines"] += 1
            try:
                item = json.loads(line)
                raw_txt = item.get('raw_output', '').lower()
                
                if not raw_txt or len(raw_txt) < 10:
                    stats["empty_output"] += 1
                    continue

                found_any = False
                for cat, words in vocabs.items():
                    for word in words:
                        if word in raw_txt:
                            stats["category_counts"][cat] += 1
                            found_any = True
                            break # Found one word for this category
                
                if found_any:
                    stats["valid_traits"] += 1
                else:
                    stats["empty_output"] += 1

            except Exception:
                stats["json_corrupt"] += 1

    print("\n--- Audit Report ---")
    print(f"Total Entries: {stats['total_lines']}")
    print(f"Corrupt JSON:  {stats['json_corrupt']}")
    print(f"Valid Images:  {stats['valid_traits']} ({(stats['valid_traits']/stats['total_lines'])*100:.1f}%)")
    print(f"Empty/Null:    {stats['empty_output']}")
    print("\nTrait Density per Category:")
    for cat, count in stats["category_counts"].items():
        print(f"  - {cat:15}: {count:5} ({(count/stats['total_lines'])*100:.1f}%)")

if __name__ == "__main__":
    audit_traits()
