from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

data_path = Path("data/cleaned_sample.csv")
output_folder = Path("data")
output_folder.mkdir(exist_ok=True)

print("DAY 3 - TRAIN TEST SPLIT")

data = pd.read_csv(data_path)
data.columns = data.columns.str.strip()

# Convert attack labels into binary classes
data["BinaryLabel"] = (
    data["Label"].astype(str).str.strip() != "BENIGN"
).astype(int)

# Remove exact duplicate feature records to reduce leakage
feature_columns = [
    col for col in data.columns
    if col not in ["Label", "BinaryLabel"]
]

before = len(data)
data = data.drop_duplicates(subset=feature_columns)
print("Duplicates removed:", before - len(data))

# Separate features and target
X = data[feature_columns]
y = data["BinaryLabel"]

# Check that both classes are present
print("\nBinary class distribution:")
print(y.value_counts().sort_index())

if y.nunique() != 2:
    raise ValueError("Both BENIGN and ATTACK classes are required.")

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Save datasets
X_train.to_csv(output_folder / "X_train.csv", index=False)
X_test.to_csv(output_folder / "X_test.csv", index=False)
y_train.to_csv(output_folder / "y_train.csv", index=False)
y_test.to_csv(output_folder / "y_test.csv", index=False)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))
print("\nTrain/test split completed!")