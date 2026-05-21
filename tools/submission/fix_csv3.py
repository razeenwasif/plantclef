import csv
import pandas as pd

df = pd.read_csv("submissions/submission.csv")
with open("submissions/submission_obs_id.csv", "w", newline="") as f:
    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
    writer.writerow(["observation_id", "species_ids"])
    for _, row in df.iterrows():
        writer.writerow([row['quadrat_id'], row['species_ids']])
