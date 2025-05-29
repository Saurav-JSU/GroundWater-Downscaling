# scripts/validate_feature_timestamps.py

import os
import re
import yaml
from glob import glob
from datetime import datetime

def get_timestamp(f, dataset_type, idx):
    fname = os.path.basename(f)
    if dataset_type == "chirps" and re.match(r"^\d+\.tif$", fname):
        year = 2003 + idx // 12
        month = 1 + (idx % 12)
        return f"{year:04d}-{month:02d}"
    elif re.search(r"(\d{6})", fname):
        raw = re.search(r"(\d{6})", fname).group(1)
        return f"{raw[:4]}-{raw[4:]}"
    elif re.search(r"(\d{4}_\d{2})", fname):
        raw = re.search(r"(\d{4}_\d{2})", fname).group(1)
        return raw.replace("_", "-")
    elif re.match(r"\d{4}_\d{2}_\d{2}\.tif", fname):
        parts = fname.split("_")
        return f"{parts[0]}-{parts[1]}"
    else:
        return None

def validate():
    with open("src/config.yaml") as f:
        config = yaml.safe_load(f)

    input_dirs = config["input_dirs"]
    os.makedirs("results", exist_ok=True)
    log_path = "results/feature_timestamp_log.txt"

    all_logs = []

    for path in input_dirs:
        dataset = os.path.basename(path).lower()
        tif_files = sorted(glob(os.path.join(path, "*.tif")))
        parsed = []

        for i, f in enumerate(tif_files):
            ts = get_timestamp(f, dataset, i)
            parsed.append((os.path.basename(f), ts))

        all_logs.append(f"\n--- {dataset.upper()} ---")
        all_logs += [f"  {fname} → {ts if ts else '❌ NO TIMESTAMP'}" for fname, ts in parsed]

        total = len(parsed)
        valid = sum(1 for _, ts in parsed if ts)
        unique_months = sorted(set(ts for _, ts in parsed if ts))
        all_logs.append(f"✅ Valid timestamps: {valid}/{total}")
        all_logs.append(f"📆 Unique months: {len(unique_months)}")
        all_logs.append(f"🧾 Range: {unique_months[0] if unique_months else 'N/A'} to {unique_months[-1] if unique_months else 'N/A'}")

    with open(log_path, "w") as f:
        f.write("\n".join(all_logs))

    print(f"✅ Timestamp validation completed. See {log_path}")

if __name__ == "__main__":
    validate()
