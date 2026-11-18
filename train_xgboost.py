
from pathlib import Path
import time

import pandas as pd
import numpy as np
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# STEP 1: Set up folders
data_folder = Path("data")
model_folder = Path("models")
model_folder.mkdir(exist_ok=True)

print("DAY 6 - XGBOOST MODEL TRAINING")
print("--------------------------------", flush=True)

# STEP 2: Load training and testing datasets
print("Loading datasets...", flush=True)

X_train = pd.read_csv(data_folder / "X_train.csv")
X_test = pd.read_csv(data_folder / "X_test.csv")

y_train = pd.read_csv(
    data_folder / "y_train.csv"
).iloc[:, 0]

y_test = pd.read_csv(
    data_folder / "y_test.csv"
).iloc[:, 0]

# STEP 3: Verify the datasets
assert list(X_train.columns) == list(X_test.columns)
assert len(X_train) == len(y_train)
assert len(X_test) == len(y_test)
assert set(y_train.unique()) == {0, 1}

assert all(
    pd.api.types.is_numeric_dtype(t)
    for t in X_train.dtypes
)

assert np.isfinite(X_train.to_numpy()).all()
assert np.isfinite(X_test.to_numpy()).all()

print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Number of features:", len(X_train.columns))

# STEP 4: Create the XGBoost model
model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
    n_jobs=2
)

# STEP 5: Train the model
print("\nTraining XGBoost...", flush=True)

start_time = time.time()

model.fit(X_train, y_train)

training_time = time.time() - start_time

print("Training completed!", flush=True)
print(f"Training time: {training_time:.2f} seconds")

# STEP 6: Make predictions
predictions = model.predict(X_test)

# STEP 7: Calculate performance metrics
accuracy = accuracy_score(y_test, predictions)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

print("\nXGBOOST MODEL EVALUATION")
print("--------------------------------")

print(f"Accuracy:  {accuracy:.6f} ({accuracy * 100:.2f}%)")
print(f"Precision: {precision:.6f} ({precision * 100:.2f}%)")
print(f"Recall:    {recall:.6f} ({recall * 100:.2f}%)")
print(f"F1-score:  {f1:.6f} ({f1 * 100:.2f}%)")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["BENIGN", "ATTACK"],
        zero_division=0
    )
)

# STEP 8: Generate confusion matrix
cm = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

print("\nCONFUSION MATRIX RESULTS")
print("True Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives:", tp)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["BENIGN", "ATTACK"]
)

display.plot(
    cmap="Blues",
    values_format="d"
)

plt.title("XGBoost - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    model_folder / "xgboost_confusion_matrix.png",
    dpi=150
)

plt.close()

# STEP 9: Save trained XGBoost model
joblib.dump(
    {
        "model": model,
        "features": X_train.columns.tolist()
    },
    model_folder / "xgboost_model.joblib"
)

# STEP 10: Save evaluation metrics
metrics = pd.DataFrame([{
    "Model": "XGBoost",
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1": f1,
    "TN": tn,
    "FP": fp,
    "FN": fn,
    "TP": tp,
    "TrainingTimeSeconds": training_time
}])

metrics.to_csv(
    model_folder / "xgboost_metrics.csv",
    index=False
)

print("\nXGBoost model saved successfully!")
print("Confusion matrix saved successfully!")
print("Evaluation metrics saved successfully!")
print("DAY 6 TRAINING COMPLETED!")
