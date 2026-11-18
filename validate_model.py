
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# STEP 1: Locate project files
BASE_DIR = Path(__file__).resolve().parent

X_TEST_PATH = BASE_DIR / "data" / "X_test.csv"
Y_TEST_PATH = BASE_DIR / "data" / "y_test.csv"
MODEL_PATH = BASE_DIR / "models" / "xgboost_model.joblib"

REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)

print("\nDAY 11 - XGBOOST MODEL VALIDATION")
print("--------------------------------")

# STEP 2: Load saved model
saved_model = joblib.load(MODEL_PATH)

model = saved_model["model"]
feature_names = list(saved_model["features"])

print("Model loaded successfully!")
print("Required features:", len(feature_names))

# STEP 3: Load testing data
X_test = pd.read_csv(X_TEST_PATH)
y_test_df = pd.read_csv(Y_TEST_PATH)

X_test.columns = X_test.columns.str.strip()

if y_test_df.shape[1] != 1:
    raise ValueError("y_test.csv must contain one label column.")

y_test = y_test_df.iloc[:, 0]

if len(X_test) != len(y_test):
    raise ValueError("Feature and label row counts differ.")

print("Testing records:", len(X_test))

# STEP 4: Prepare features
missing = [
    name for name in feature_names
    if name not in X_test.columns
]

if missing:
    raise ValueError(f"Missing model features: {missing}")

X_test = X_test[feature_names].copy()

for column in feature_names:
    X_test[column] = pd.to_numeric(
        X_test[column],
        errors="raise"
    )

if not np.isfinite(X_test.to_numpy(dtype=float)).all():
    raise ValueError("Test features contain NaN or infinity.")

# STEP 5: Prepare actual labels
y_test = pd.to_numeric(y_test, errors="raise").astype(int)

if not y_test.isin([0, 1]).all():
    raise ValueError("Expected labels 0 and 1.")

# STEP 6: Make predictions
print("\nGenerating predictions...")

y_pred = np.asarray(model.predict(X_test)).astype(int)

# STEP 7: Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
).ravel()

false_positive_rate = fp / (fp + tn) if fp + tn else 0
false_negative_rate = fn / (fn + tp) if fn + tp else 0

# STEP 8: Display results
print("\nMODEL PERFORMANCE")
print("-----------------")
print(f"Accuracy:  {accuracy:.4%}")
print(f"Precision: {precision:.4%}")
print(f"Recall:    {recall:.4%}")
print(f"F1-score:  {f1:.4%}")

print("\nCONFUSION MATRIX")
print("----------------")
print("True Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives:", tp)

print(f"\nFalse Positive Rate: {false_positive_rate:.4%}")
print(f"False Negative Rate: {false_negative_rate:.4%}")

print("\nCLASSIFICATION REPORT")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["BENIGN", "ATTACK"],
        zero_division=0
    )
)

# STEP 9: Save validation metrics
metrics = pd.DataFrame([{
    "Model": "XGBoost",
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1": f1,
    "True_Negatives": int(tn),
    "False_Positives": int(fp),
    "False_Negatives": int(fn),
    "True_Positives": int(tp),
    "False_Positive_Rate": false_positive_rate,
    "False_Negative_Rate": false_negative_rate
}])

metrics.to_csv(
    REPORT_DIR / "day11_validation_metrics.csv",
    index=False
)

print("\nValidation metrics saved successfully!")
print("DAY 11 MODEL VALIDATION COMPLETED!")
