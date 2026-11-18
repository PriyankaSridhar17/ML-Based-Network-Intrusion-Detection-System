
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# STEP 1: Set up folders
data_folder = Path("data")
model_folder = Path("models")
report_folder = Path("reports")
report_folder.mkdir(exist_ok=True)

print("DAY 7 - MODEL COMPARISON AND VALIDATION")
print("---------------------------------------", flush=True)

# STEP 2: Load existing datasets
print("Loading datasets...", flush=True)

X_train = pd.read_csv(data_folder / "X_train.csv")
X_test = pd.read_csv(data_folder / "X_test.csv")

y_test = pd.read_csv(
    data_folder / "y_test.csv"
).iloc[:, 0]

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# STEP 3: Load all three saved models
model_files = {
    "Decision Tree": "decision_tree_model.joblib",
    "Random Forest": "random_forest_model.joblib",
    "XGBoost": "xgboost_model.joblib"
}

results = []
loaded_models = {}

for model_name, filename in model_files.items():

    print(f"\nEvaluating {model_name}...", flush=True)

    saved = joblib.load(model_folder / filename)
    model = saved["model"]
    feature_names = saved["features"]

    if list(X_test.columns) != list(feature_names):
        raise ValueError(
            f"Feature mismatch for {model_name}"
        )

    predictions = model.predict(X_test)

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    results.append({
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "False Positives": int(fp),
        "False Negatives": int(fn)
    })

    loaded_models[model_name] = model

# STEP 4: Save comparison table
comparison = pd.DataFrame(results)

comparison.to_csv(
    report_folder / "model_comparison.csv",
    index=False
)

print("\nMODEL COMPARISON")
print("----------------")
print(comparison.to_string(index=False))

# STEP 5: Generate comparison chart
chart_data = comparison.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1"]
]

ax = chart_data.plot(
    kind="bar",
    figsize=(11, 6),
    ylim=(0.95, 1.005)
)

ax.set_title("ML Intrusion Detection - Model Comparison")
ax.set_ylabel("Score")
ax.set_xlabel("Machine Learning Model")
ax.legend(loc="lower right")

plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    report_folder / "model_comparison.png",
    dpi=150
)

plt.close()

print("\nComparison chart saved successfully!")

# STEP 6: Analyze XGBoost feature importance
xgb_model = loaded_models["XGBoost"]

importance = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance": xgb_model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

importance.to_csv(
    report_folder / "xgboost_feature_importance.csv",
    index=False
)

print("\nTOP 15 XGBOOST FEATURES")
print("----------------------")
print(importance.head(15).to_string(index=False))

# STEP 7: Generate feature importance chart
top_features = importance.head(15).iloc[::-1]

plt.figure(figsize=(11, 7))
plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.title("XGBoost - Top 15 Feature Importances")
plt.xlabel("Feature Importance")
plt.tight_layout()

plt.savefig(
    report_folder / "xgboost_feature_importance.png",
    dpi=150
)

plt.close()

# STEP 8: Check for suspicious feature names
suspicious_terms = [
    "label",
    "attack",
    "target",
    "class",
    "timestamp",
    "flow id",
    "source ip",
    "destination ip"
]

suspicious_columns = [
    column for column in X_train.columns
    if any(
        term in column.lower()
        for term in suspicious_terms
    )
]

print("\nPOTENTIALLY SUSPICIOUS FEATURE NAMES")
print("-----------------------------------")
print(suspicious_columns)

# STEP 9: Check exact feature-row overlap
print("\nCHECKING TRAIN/TEST FEATURE OVERLAP...")

train_hashes = pd.util.hash_pandas_object(
    X_train,
    index=False
)

test_hashes = pd.util.hash_pandas_object(
    X_test,
    index=False
)

overlap = int(
    test_hashes.isin(set(train_hashes)).sum()
)

print("Matching feature rows:", overlap)
print(
    "Note: This checks exact row matches, "
    "not related traffic flows or capture-session leakage."
)

# STEP 10: Identify strongest baseline
best = comparison.sort_values(
    ["F1", "Recall"],
    ascending=False
).iloc[0]

print("\nBEST BASELINE BY F1-SCORE")
print("-------------------------")
print("Model:", best["Model"])
print("F1-score:", round(best["F1"], 6))
print("Recall:", round(best["Recall"], 6))

print("\nReports saved to:", report_folder.resolve())
print("DAY 7 INITIAL ANALYSIS COMPLETED!")
