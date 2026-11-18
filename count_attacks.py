from pathlib import Path
from collections import Counter
import pandas as pd

dataset_folder = Path(
    r"C:\Users\admin\IDS_Datasets\MachineLearningCVE"
)

csv_files = sorted(dataset_folder.glob("*.csv"))
attack_counts = Counter()
total_records = 0

print("CICIDS2017 - COMPLETE TRAFFIC LABEL ANALYSIS")
print("--------------------------------------------")

for file in csv_files:
    print(f"\nAnalyzing: {file.name}")
    file_records = 0

    for chunk in pd.read_csv(
        file,
        chunksize=50000,
        low_memory=False
    ):
        chunk.columns = chunk.columns.str.strip()

        if "Label" not in chunk.columns:
            raise ValueError(f"Label column missing: {file.name}")

        labels = chunk["Label"].astype("string").str.strip()
        attack_counts.update(labels.dropna().value_counts().to_dict())

        file_records += len(chunk)

    total_records += file_records
    print(f"Records processed: {file_records:,}")

print("\nTOTAL NETWORK RECORDS:", f"{total_records:,}")

print("\nTRAFFIC CLASS DISTRIBUTION")
print("--------------------------------------------")

for label, count in attack_counts.most_common():
    print(f"{label}: {count:,}")

benign = attack_counts.get("BENIGN", 0)
malicious = sum(
    count for label, count in attack_counts.items()
    if label != "BENIGN"
)

print("\nBENIGN RECORDS:", f"{benign:,}")
print("ATTACK RECORDS:", f"{malicious:,}")
print("\nComplete label analysis finished!")