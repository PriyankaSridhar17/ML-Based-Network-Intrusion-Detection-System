from pathlib import Path
import numpy as np
import pandas as pd

dataset_folder = Path(
    r"C:\Users\admin\IDS_Datasets\MachineLearningCVE"
)

csv_files = sorted(dataset_folder.glob("*.csv"))

total_rows = 0
missing_rows = 0
infinite_rows = 0

print("CICIDS2017 - DATA QUALITY CHECK")
print("--------------------------------")

for file in csv_files:
    print(f"\nChecking: {file.name}", flush=True)

    for chunk in pd.read_csv(file, chunksize=20000):
        chunk.columns = chunk.columns.str.strip()

        total_rows += len(chunk)

        missing_mask = chunk.isna().any(axis=1)
        missing_rows += int(missing_mask.sum())

        numeric_data = chunk.drop(
            columns=["Label"], errors="ignore"
        ).select_dtypes(include="number")

        infinite_mask = np.isinf(
            numeric_data.to_numpy(dtype="float64")
        ).any(axis=1)

        infinite_rows += int(infinite_mask.sum())

print("\nDATA QUALITY SUMMARY")
print("--------------------")
print("Total rows:", f"{total_rows:,}")
print("Rows with missing values:", f"{missing_rows:,}")
print("Rows with infinite values:", f"{infinite_rows:,}")
print("\nData quality check completed!")