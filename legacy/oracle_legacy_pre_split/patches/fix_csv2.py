path = 'phases/inference/pipeline.py'
with open(path, 'r') as f:
    content = f.read()

# Ensure we use CSV.QUOTE_ALL for the final output
old_write = """    with open(cfg.submission_csv, "w") as f:
        f.write("quadrat_id;species_ids\\n")
        for qid, preds in results.items():
            f.write(f'"{qid}";"[{", ".join(preds)}]"\n')"""

new_write = """    import csv
    with open(cfg.submission_csv, "w", newline="\\n") as f:
        writer = csv.writer(f, delimiter=";", quoting=csv.QUOTE_ALL, lineterminator="\\n")
        writer.writerow(["quadrat_id", "species_ids"])
        for qid in sorted(results.keys()):
            preds = results[qid]
            formatted_preds = f"[{', '.join(preds)}]"
            writer.writerow([qid, formatted_preds])"""

content = content.replace(old_write, new_write)
with open(path, 'w') as f:
    f.write(content)
