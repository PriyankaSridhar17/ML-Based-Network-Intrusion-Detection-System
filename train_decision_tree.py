
from pathlib import Path
import pandas as pd
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
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

print("DAY 4 - DECISION TREE MODEL TRAINING")
print("------------------------------------")

# STEP 2: Load training and testing datasets
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

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# STEP 3: Create the Decision Tree model
model = DecisionTreeClassifier(
    max_depth=12,
    min_samples_leaf=5,
    random_state=42
)

# STEP 4: Train the model
print("\nTraining model...")

model.fit(X_train, y_train)

print("Training completed!")

# STEP 5: Make predictions
predictions = model.predict(X_test)

# STEP 6: Evaluate the model
print("\nMODEL EVALUATION")
print("----------------")

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

print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1-score:", round(f1, 4))

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

# STEP 7: Generate confusion matrix
cm = confusion_matrix(
    y_test, predictions, labels=[0, 1]
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["BENIGN", "ATTACK"]
)

display.plot(cmap="Blues", values_format="d")

plt.title("Decision Tree - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    model_folder / "decision_tree_confusion_matrix.png",
    dpi=150
)

plt.close()

# STEP 8: Save trained model
joblib.dump(
    {
        "model": model,
        "features": X_train.columns.tolist()
    },
    model_folder / "decision_tree_model.joblib"
)

print("\nModel saved successfully!")
print("Confusion matrix saved successfully!")
print("DAY 4 TRAINING COMPLETED!")
