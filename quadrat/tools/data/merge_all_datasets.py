"""Merges verified training data, pseudo-labels, and synthetic collages."""

import os

import pandas as pd


def main() -> None:
    """
    Main function to merge all datasets into a final training metadata file.

    Loads the verified training data, pseudo-labels, and synthetic collages,
    standardizes the column names, and concatenates them into a single CSV.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    base_dir = "/workspace/plantclef/processed"

    # 1. Load Verified + Pseudo (Already merged by filter_pseudo_labels.py)
    # Format: image_name;species_id
    print("Loading Verified + Pseudo data...")
    base_csv_path = os.path.join(base_dir, "train_plus_pseudo.csv")
    if not os.path.exists(base_csv_path):
        print(f"Error: {base_csv_path} not found.")
        return

    df_base = pd.read_csv(base_csv_path, sep=';')
    # Rename column to plural to match multi-label schema
    df_base = df_base.rename(columns={'species_id': 'species_ids'})
    # Ensure species_ids is a string for compatibility
    df_base['species_ids'] = df_base['species_ids'].astype(str)

    # 2. Load Synthetic Collages
    # Format: image_name;species_ids (comma separated)
    print("Loading Synthetic Collages...")
    collages_csv_path = os.path.join(base_dir, "synthetic_collages.csv")
    if not os.path.exists(collages_csv_path):
        print(f"Warning: {collages_csv_path} not found. Skipping collages.")
        df_collages = pd.DataFrame(columns=['image_name', 'species_ids'])
    else:
        df_collages = pd.read_csv(collages_csv_path, sep=';')
        # Ensure species_ids is a string
        df_collages['species_ids'] = df_collages['species_ids'].astype(str)

    # 3. Concatenate
    df_final = pd.concat([df_base, df_collages], ignore_index=True)

    # 4. Save Final Metadata
    output_path = os.path.join(base_dir, "student_train_final.csv")
    df_final.to_csv(output_path, sep=';', index=False)

    print("\n" + "=" * 40)
    print("      MASTER MERGE COMPLETE")
    print("=" * 40)
    print(f"Total Base Samples:    {len(df_base):,}")
    print(f"Total Synthetic:       {len(df_collages):,}")
    print(f"Final Student Dataset: {len(df_final):,}")
    print(f"Saved to: {output_path}")
    print("=" * 40)


if __name__ == "__main__":
    main()
