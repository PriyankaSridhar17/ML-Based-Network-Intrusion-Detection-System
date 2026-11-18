
# ============================================================
# DAY 11 - DATA LEAKAGE DETECTION
# ML-Based Network Intrusion Detection System
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np
import joblib

# ------------------------------------------------------------
# STEP 1: Define project paths
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(exist_ok=True)

X_TRAIN_PATH = DATA_DIR / "X_train.csv"
X_TEST_PATH = DATA_DIR / "X_test.csv"
Y_TRAIN_PATH = DATA_DIR / "y_train.csv"
Y_TEST_PATH = DATA_DIR / "y_test.csv"

MODEL_PATH = MODEL_DIR / "xgboost_model.joblib"

print("=" * 60)
print("DAY 11 - DATA LEAKAGE DETECTION")
print("=" * 60)

# ------------------------------------------------------------
# STEP 2: Load training and testing datasets
# ------------------------------------------------------------

print("\nSTEP 2: Loading datasets...")

X_train = pd.read_csv(X_TRAIN_PATH)
X_test = pd.read_csv(X_TEST_PATH)

y_train = pd.read_csv(Y_TRAIN_PATH).iloc[:, 0]
y_test = pd.read_csv(Y_TEST_PATH).iloc[:, 0]

X_train.columns = X_train.columns.str.strip()
X_test.columns = X_test.columns.str.strip()

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

print("Training features:", X_train.shape[1])
print("Testing features:", X_test.shape[1])

if len(X_train) != len(y_train):
    raise ValueError("Training features and labels do not match.")

if len(X_test) != len(y_test):
    raise ValueError("Testing features and labels do not match.")

if not X_train.columns.equals(X_test.columns):
    raise ValueError(
        "Training and testing feature columns are different."
    )

print("Dataset loading completed!")

# ------------------------------------------------------------
# STEP 3: Check missing values
# ------------------------------------------------------------

print("\nSTEP 3: Checking missing values...")

train_missing = int(X_train.isna().sum().sum())
test_missing = int(X_test.isna().sum().sum())

print("Missing training values:", train_missing)
print("Missing testing values:", test_missing)

# ------------------------------------------------------------
# STEP 4: Check duplicate rows inside each dataset
# ------------------------------------------------------------

print("\nSTEP 4: Checking duplicate rows...")

train_duplicates = int(X_train.duplicated().sum())
test_duplicates = int(X_test.duplicated().sum())

print("Duplicate training rows:", train_duplicates)
print("Duplicate testing rows:", test_duplicates)

# ------------------------------------------------------------
# STEP 5: Check exact train/test feature overlap
# ------------------------------------------------------------

print("\nSTEP 5: Checking train/test overlap...")

train_hashes = pd.util.hash_pandas_object(
    X_train,
    index=False
)

test_hashes = pd.util.hash_pandas_object(
    X_test,
    index=False
)

train_hash_set = set(train_hashes.to_numpy())

overlap_count = int(
    test_hashes.isin(train_hash_set).sum()
)

overlap_percent = (
    overlap_count / len(X_test) * 100
    if len(X_test) > 0 else 0
)

print("Matching test feature rows:", overlap_count)

print(
    f"Exact feature overlap percentage: "
    f"{overlap_percent:.4f}%"
)

# ------------------------------------------------------------
# STEP 6: Check suspicious label-related columns
# ------------------------------------------------------------

print("\nSTEP 6: Checking suspicious feature names...")

suspicious_keywords = [
    "label",
    "attack",
    "target",
    "class",
    "outcome",
    "prediction",
    "is_malicious"
]

suspicious_columns = []

for column in X_train.columns:

    column_lower = column.lower()

    if any(
        keyword in column_lower
        for keyword in suspicious_keywords
    ):
        suspicious_columns.append(column)

print("Potential label-related columns:")

if suspicious_columns:
    for column in suspicious_columns:
        print("-", column)
else:
    print("No obvious label-related column names found.")

# ------------------------------------------------------------
# STEP 7: Check near-duplicate feature patterns
# ------------------------------------------------------------

print("\nSTEP 7: Checking rounded feature overlap...")

numeric_columns = X_train.select_dtypes(
    include=[np.number]
).columns.tolist()

rounded_overlap = 0

if numeric_columns:

    train_rounded = X_train[numeric_columns].round(2)
    test_rounded = X_test[numeric_columns].round(2)

    rounded_train_hashes = pd.util.hash_pandas_object(
        train_rounded,
        index=False
    )

    rounded_test_hashes = pd.util.hash_pandas_object(
        test_rounded,
        index=False
    )

    rounded_train_set = set(
        rounded_train_hashes.to_numpy()
    )

    rounded_overlap = int(
        rounded_test_hashes.isin(
            rounded_train_set
        ).sum()
    )

print(
    "Matching rounded test rows:",
    rounded_overlap
)

print(
    "Note: Rounded overlap is only a limited similarity "
    "check, not a complete near-duplicate analysis."
)

# ------------------------------------------------------------
# STEP 8: Check model feature consistency
# ------------------------------------------------------------

print("\nSTEP 8: Checking model features...")

saved_model = joblib.load(MODEL_PATH)

model = saved_model["model"]
feature_names = list(saved_model["features"])

missing_model_features = [
    feature
    for feature in feature_names
    if feature not in X_train.columns
]

print("Saved model type:", type(model).__name__)
print("Required model features:", len(feature_names))

print(
    "Missing required model features:",
    len(missing_model_features)
)

if missing_model_features:
    print("Missing:", missing_model_features)

# ------------------------------------------------------------
# STEP 9: Generate leakage investigation report
# ------------------------------------------------------------

print("\nSTEP 9: Saving investigation report...")

report = pd.DataFrame([{
    "Training_Rows": len(X_train),
    "Testing_Rows": len(X_test),
    "Training_Duplicates": train_duplicates,
    "Testing_Duplicates": test_duplicates,
    "Exact_Overlap_Rows": overlap_count,
    "Exact_Overlap_Percent": overlap_percent,
    "Rounded_Overlap_Rows": rounded_overlap,
    "Missing_Training_Values": train_missing,
    "Missing_Testing_Values": test_missing,
    "Suspicious_Columns": ", ".join(suspicious_columns),
    "Model_Feature_Count": len(feature_names),
    "Missing_Model_Features": len(missing_model_features)
}])

report_path = (
    REPORT_DIR / "day11_data_leakage_report.csv"
)

report.to_csv(
    report_path,
    index=False
)

print("Report saved successfully!")
print("Location:", report_path)

# ------------------------------------------------------------
# STEP 10: Final interpretation
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("DATA LEAKAGE INVESTIGATION SUMMARY")
print("=" * 60)

if overlap_count > 0:
    print(
        "WARNING: Exact feature rows overlap "
        "between training and testing."
    )
else:
    print(
        "No exact train/test feature overlap detected."
    )

if suspicious_columns:
    print(
        "WARNING: Review potentially label-related columns."
    )
else:
    print(
        "No obvious label-related feature names detected."
    )

if rounded_overlap > overlap_count:
    print(
        "Additional rounded feature matches were found."
    )

print(
    "\nIMPORTANT: These checks cannot rule out "
    "session-level, temporal, preprocessing, "
    "or model-selection leakage."
)

print(
    "Independent capture/session-based validation "
    "is still recommended."
)

print("\nDAY 11 DATA LEAKAGE CHECK COMPLETED!")
