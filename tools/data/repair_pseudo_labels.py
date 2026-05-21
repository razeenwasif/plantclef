"""Script to repair pseudo-labels CSV files.

This script cleans the 'uncertainty' column in a pseudo-labels CSV,
extracting numerical values from tensor-like strings and handling missing data.
"""

import os
import re

import pandas as pd


def repair_csv(file_path: str) -> None:
    """
    Repair the uncertainty column in a pseudo-labels CSV file.

    Extracts numerical values from tensor-like strings and handles missing
    data for consistency across inference outputs.

    Parameters
    ----------
    file_path : str
        The path to the CSV file to be repaired.

    Returns
    -------
    None
    """
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    print(f"Repairing {file_path}...")
    df = pd.read_csv(file_path)

    def clean_uncertainty(val: str) -> float:
        """
        Extract numerical value from a string containing 'tensor'.

        Parameters
        ----------
        val : str
            The raw uncertainty value, potentially as a string.

        Returns
        -------
        float
            The cleaned numerical uncertainty value.
        """
        if isinstance(val, str) and 'tensor' in val:
            # Extract the first number found in the string
            match = re.search(r"(\d+\.?\d*)", val)
            if match:
                return float(match.group(1))
        return float(val) if isinstance(val, (int, float, str)) else 0.0

    df['uncertainty'] = df['uncertainty'].apply(clean_uncertainty)
    df['uncertainty'] = pd.to_numeric(
        df['uncertainty'], errors='coerce').fillna(0.0)

    # Save back
    df.to_csv(file_path, index=False)
    print("Repair complete.")


if __name__ == "__main__":
    repair_csv('pseudo_labels.csv')
