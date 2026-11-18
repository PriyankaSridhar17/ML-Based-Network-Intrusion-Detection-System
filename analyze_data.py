from pathlib import Path
import pandas as pd

dataset_folder = Path(
    r"C:\Users\admin\IDS_Datasets\MachineLearningCVE"
)

csv_files = sorted(dataset_folder.glob("*.csv"))

print("DAY 2 - CICIDS2017 DATA ANALYSIS")
print("--------------------------------")
print("Total CSV files:", len(csv_files))

for file in csv_files:
    print("\nFile:", file.name)

    # Read a small sample to conserve memory
    data = pd.read_csv(file, nrows=1000, low_memory=False)

    # Remove extra spaces from column names
    data.columns = data.columns.str.strip()

    print("Sample rows:", len(data))
    print("Columns:", len(data.columns))

    if "Label" in data.columns:
        print("Traffic labels in sample:")
        print(data["Label"].astype(str).str.strip().value_counts())
    else:
        print("Label column not found")

    print("Missing values in sample:", int(data.isna().sum().sum()))

print("\nInitial dataset analysis completed!")