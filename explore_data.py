from pathlib import Path
import pandas as pd

# Location of your CICIDS2017 dataset
dataset_folder = Path(
    r"C:\Users\admin\IDS_Datasets\MachineLearningCVE"
)

# Find all CSV files
csv_files = sorted(dataset_folder.glob("*.csv"))

print("ML-Based Network Intrusion Detection System")
print("-------------------------------------------")
print("Number of CSV files found:", len(csv_files))

for file in csv_files:
    print(file.name)

# Read the first five network traffic records
if csv_files:
    data = pd.read_csv(csv_files[0], nrows=5)

    print("\nFirst 5 network traffic records:")
    print(data.head().to_string(index=False))

    print("\nNumber of columns:", len(data.columns))

    label_column = next(
        (col for col in data.columns if col.strip() == "Label"),
        None
    )

    if label_column:
        print("\nTraffic labels in this sample:")
        print(data[label_column].unique())
    else:
        print("Label column not found.")
else:
    print("No CSV files found. Check the dataset path.")