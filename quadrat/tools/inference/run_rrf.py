import argparse
import pandas as pd
from collections import defaultdict
from pathlib import Path

def parse_predictions(csv_path):
    """Parses a submission CSV into a dictionary of {image_id: [species_ids]}."""
    df = pd.read_csv(csv_path, sep=';')
    preds = {}
    for _, row in df.iterrows():
        # Handle cases where species_ids might be empty or formatted differently
        try:
            species_str = str(row['species_ids']).strip()
            if species_str and species_str != 'nan':
                # Assuming space-separated or comma-separated. PlantCLEF often uses space.
                species_list = [int(float(x)) for x in species_str.replace(',', ' ').split()]
            else:
                species_list = []
        except ValueError:
            species_list = []
        preds[str(row['image_id'])] = species_list
    return preds

def reciprocal_rank_fusion(predictions_list, k=60):
    """
    Applies Reciprocal Rank Fusion to a list of prediction dictionaries.
    
    RRF Score = sum(1 / (k + rank)) for each model
    """
    fused_scores = defaultdict(lambda: defaultdict(float))
    all_image_ids = set()
    
    # Collect all unique image IDs across all submissions
    for preds in predictions_list:
        all_image_ids.update(preds.keys())

    for image_id in all_image_ids:
        for preds in predictions_list:
            if image_id in preds:
                # The species are assumed to be sorted by confidence in the CSV
                for rank, species_id in enumerate(preds[image_id]):
                    fused_scores[image_id][species_id] += 1.0 / (k + rank + 1) # rank is 0-indexed

    # Sort species by fused score for each image
    fused_predictions = {}
    for image_id, scores in fused_scores.items():
        sorted_species = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        fused_predictions[image_id] = [species for species, score in sorted_species]
        
    return fused_predictions

def main():
    parser = argparse.ArgumentParser(description="Reciprocal Rank Fusion for PlantCLEF 2026")
    parser.add_argument("--inputs", nargs='+', required=True, help="Paths to input submission CSV files")
    parser.add_argument("--output", type=str, default="rrf_submission.csv", help="Path to output fused CSV")
    parser.add_argument("--k", type=int, default=60, help="RRF constant k (default: 60)")
    parser.add_argument("--top-n", type=int, default=100, help="Maximum number of species to keep per image")
    args = parser.parse_args()

    print(f"Loading {len(args.inputs)} submission files...")
    predictions_list = []
    for csv_path in args.inputs:
        if not Path(csv_path).exists():
            print(f"Error: File not found: {csv_path}")
            return
        preds = parse_predictions(csv_path)
        predictions_list.append(preds)
        print(f"  Loaded {len(preds)} predictions from {csv_path}")

    print(f"Applying Reciprocal Rank Fusion (k={args.k})...")
    fused_preds = reciprocal_rank_fusion(predictions_list, k=args.k)

    print(f"Writing fused predictions to {args.output}...")
    with open(args.output, 'w') as f:
        f.write("image_id;species_ids\n")
        for image_id, species_list in fused_preds.items():
            # Keep only the top-N predictions to match competition constraints
            top_species = species_list[:args.top_n]
            species_str = " ".join(map(str, top_species))
            f.write(f"{image_id};{species_str}\n")
            
    print("Done!")

if __name__ == "__main__":
    main()
