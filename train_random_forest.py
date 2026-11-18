
from pathlib import Path
import pandas as pd
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
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

print("DAY 5 - RANDOM FOREST MODEL TRAINING")
print("------------------------------------", flush=True)

# STEP 2: Load the training and testing data
print("Loading datasets...", flush=True)

X_train = pd.read_csv(data_folder / "X_train.csv")
X_test = pd.read_csv(data_folder / "X_test.csv")

y_train = pd.read_csv(
    data_folder / "y_train.csv"
).iloc[:, 0]

y_test = pd.read_csv(
    data_folder / "y_test.csv"
).iloc[:, 0]

# Verify the datasets
assert list(X_train.columns) == list(X_test.columns)
assert len(X_train) == len(y_train)
assert len(X_test) == len(y_test)
assert set(y_train.unique()) == {0, 1}

print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Number of features:", len(X_train.columns))

# STEP 3: Create the Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

# STEP 4: Train the model
print("\nTraining Random Forest...", flush=True)

model.fit(X_train, y_train)

print("Training completed!", flush=True)

# STEP 5: Predict test traffic
predictions = model.predict(X_test)

# STEP 6: Calculate performance metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(
    y_test, predictions, zero_division=0
)
recall = recall_score(
    y_test, predictions, zero_division=0
)
f1 = f1_score(
    y_test, predictions, zero_division=0
)

print("\nRANDOM FOREST MODEL EVALUATION")
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

# STEP 7: Generate the confusion matrix
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

display.plot(cmap="Blues", values_format="d")
plt.title("Random Forest - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    model_folder / "random_forest_confusion_matrix.png",
    dpi=150
)
plt.close()

# STEP 8: Save the trained model
joblib.dump(
    {
        "model": model,
        "features": X_train.columns.tolist()
    },
    model_folder / "random_forest_model.joblib"
)

# STEP 9: Save metrics for later comparison
metrics = pd.DataFrame([{
    "Model": "Random Forest",
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1": f1,
    "TN": tn,
    "FP": fp,
    "FN": fn,
    "TP": tp
}])

metrics.to_csv(
    model_folder / "random_forest_metrics.csv",
    index=False
)

print("\nRandom Forest model saved successfully!")
print("Confusion matrix saved successfully!")
print("Evaluation metrics saved successfully!")
print("DAY 5 TRAINING COMPLETED!")
