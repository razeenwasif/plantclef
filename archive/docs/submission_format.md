# Submission Format

## File Format

The submission file must be a **CSV** with:

- Comma separators
- All values quoted
- Two columns: `quadrat_id` and `species_ids`
- The `species_ids` column must be enclosed in **double square brackets** `[]`
- Species IDs within the brackets must be separated by a comma and a space
- Single species predictions must also be enclosed in double square brackets

## Example

```csv
"quadrat_id","species_ids"
"CBN-Pla-B1-20130724","[1395806]"
"CBN-PdlC-A1-20130807","[1351284, 1494911, 1381367, 1396535, 1412857, 1295807]"
```

## Generating the Submission File (Python)

Given a run with predictions stored in a pandas DataFrame `df_run`:

```python
import pandas as pd
import csv

df_run.to_csv("my_run.csv", sep=',', index=False, quoting=csv.QUOTE_ALL)
```
