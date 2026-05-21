import csv
import sys

input_file = "submissions/submission.csv"
output_file = "submissions/submission_fixed.csv"

try:
    with open(input_file, "r", newline="") as infile, open(output_file, "w", newline="") as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile, quoting=csv.QUOTE_ALL)
        
        header = next(reader)
        writer.writerow(["quadrat_id", "species_ids"])
        
        for row in reader:
            if len(row) < 2:
                continue
            observation_id = row[0]
            species = row[1].replace('[', '').replace(']', '').replace(',', ' ').split()
            species_str = "[" + ", ".join(species) + "]"
            writer.writerow([observation_id, species_str])
    print("Fixed CSV successfully")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
