from pathlib import Path
import numpy as np
import pandas as pd

# Original CICIDS2017 dataset location
dataset_folder = Path(
    r"C:\Users\admin\IDS_Datasets\MachineLearningCVE"
)

# Folder for cleaned output
output_folder = Path("data")
output_folder.mkdir(exist_ok=True)

csv_files = sorted(dataset_folder.glob("*.csv"))
samples = []

print("DAY 3 - DATA PREPROCESSING")
print("--------------------------")

# Sample 5% of each CSV file
for file in csv_files:
    print(f"Processing: {file.name}", flush=True)
    pieces = []

    for chunk in pd.read_csv(
        file, chunksize=20000, low_memory=False
    ):
        chunk.columns = chunk.columns.str.strip()

        sample = chunk.sample(frac=0.05, random_state=42)
        pieces.append(sample)

    if pieces:
        file_sample = pd.concat(pieces, ignore_index=True)
        samples.append(file_sample)
        print(f"Sampled rows: {len(file_sample):,}")

if not samples:
    raise RuntimeError("No CSV data found. Check the dataset path.")

# Combine all sampled records
data = pd.concat(samples, ignore_index=True)
data.columns = data.columns.str.strip()

if "Label" not in data.columns:
    raise ValueError("Label column not found.")

data["Label"] = data["Label"].astype("string").str.strip()

# Convert network features to numeric values
features = data.drop(columns=["Label"]).copy()
features = features.apply(pd.to_numeric, errors="coerce")

# Replace infinite values with missing values
features = features.replace([np.inf, -np.inf], np.nan)

# Recombine features and labels
cleaned = pd.concat([features, data[["Label"]]], axis=1)

# Remove rows with invalid or missing values
before = len(cleaned)
cleaned = cleaned.dropna().reset_index(drop=True)

# Save the cleaned dataset
output_path = output_folder / "cleaned_sample.csv"
cleaned.to_csv(output_path, index=False)

print("\nPREPROCESSING SUMMARY")
print("Sampled rows:", f"{before:,}")
print("Invalid rows removed:", f"{before - len(cleaned):,}")
print("Cleaned rows:", f"{len(cleaned):,}")
print("Saved to:", output_path.resolve())

print("\nClass distribution:")
print(cleaned["Label"].value_counts())

print("\nPreprocessing completed!")