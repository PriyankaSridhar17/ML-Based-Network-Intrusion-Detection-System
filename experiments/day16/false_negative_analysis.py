
"""
DAY 16 - FALSE NEGATIVE INVESTIGATION

Reproduce Day 15 held-out DDoS predictions and
investigate missed attacks without retraining or
changing the classification threshold.
"""

from pathlib import Path
import importlib.util
import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DAY15 = PROJECT_ROOT / "experiments" / "day15"
OUTPUT = PROJECT_ROOT / "experiments" / "day16"
OUTPUT.mkdir(parents=True, exist_ok=True)

MODEL_PATH = DAY15 / "experimental_xgboost.json"
SUMMARY_PATH = DAY15 / "experiment_summary.json"

THRESHOLD = 0.50


def load_day15_module():
    """Reuse the exact Day 15 preprocessing functions."""
    script = DAY15 / "independent_validation.py"

    spec = importlib.util.spec_from_file_location(
        "day15_validation", script
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def main():
    print("=" * 60)
    print("DAY 16 - FALSE NEGATIVE INVESTIGATION")
    print("=" * 60)

    day15 = load_day15_module()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    feature_names = day15.load_feature_names()

    holdout_path = (
        day15.DATASET_FOLDER / day15.HOLDOUT_FILE
    )

    print("\nLoading Day 15 held-out DDoS sample...")

    test_data = day15.load_clean_sample(
        holdout_path, feature_names
    )

    X_test = test_data[feature_names].copy()
    y_test = (
        test_data["BinaryLabel"]
        .astype(int)
        .to_numpy()
    )

    print("\nLoading experimental XGBoost model...")

    model = XGBClassifier()
    model.load_model(str(MODEL_PATH))

    if model.n_features_in_ != len(feature_names):
        raise ValueError("Feature count mismatch.")

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    cm = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    print("\nRECONSTRUCTED CONFUSION MATRIX")
    print(cm)

    # Stop if the reconstructed sample does not
    # reproduce the original Day 15 evaluation.
    expected = np.array([
        [4882, 2],
        [2447, 3956]
    ])

    if not np.array_equal(cm, expected):
        raise RuntimeError(
            "Day 16 results do not match Day 15. "
            "Check preprocessing, model, and sampling."
        )

    print("\nDay 15 results reproduced successfully!")

    results = pd.DataFrame({
        "ActualLabel": y_test,
        "PredictedLabel": predictions,
        "AttackScore": probabilities
    })

    results["Outcome"] = np.select(
        [
            (y_test == 0) & (predictions == 0),
            (y_test == 0) & (predictions == 1),
            (y_test == 1) & (predictions == 0),
            (y_test == 1) & (predictions == 1)
        ],
        ["TN", "FP", "FN", "TP"],
        default="UNKNOWN"
    )

    missed = results["Outcome"] == "FN"
    detected = results["Outcome"] == "TP"

    print("\nFALSE NEGATIVE ANALYSIS")
    print("Missed DDoS records:", int(missed.sum()))
    print("Detected DDoS records:", int(detected.sum()))

    # Aggregate score statistics, without exporting
    # individual test records.
    score_summary = (
        results.groupby("Outcome")["AttackScore"]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
    )

    score_summary.to_csv(
        OUTPUT / "score_summary.csv",
        index=False
    )

    print("\nATTACK SCORE SUMMARY")
    print(score_summary.to_string(index=False))

    # Compare selected features across missed and
    # correctly detected DDoS records.
    selected_features = [
        "Flow Duration",
        "Flow Bytes/s",
        "Flow Packets/s",
        "Total Fwd Packets",
        "Total Backward Packets",
        "Average Packet Size",
        "Bwd Packet Length Std"
    ]

    comparison = []

    for feature in selected_features:
        if feature not in X_test.columns:
            continue

        comparison.append({
            "Feature": feature,
            "FalseNegativeMedian": float(
                X_test.loc[missed, feature].median()
            ),
            "TruePositiveMedian": float(
                X_test.loc[detected, feature].median()
            )
        })

    pd.DataFrame(comparison).to_csv(
        OUTPUT / "feature_comparison.csv",
        index=False
    )

    # Visualize scores for detected and missed attacks.
    plt.figure(figsize=(9, 5))

    plt.hist(
        probabilities[missed],
        bins=30,
        alpha=0.65,
        label="Missed DDoS (FN)"
    )

    plt.hist(
        probabilities[detected],
        bins=30,
        alpha=0.65,
        label="Detected DDoS (TP)"
    )

    plt.axvline(
        THRESHOLD,
        linestyle="--",
        label="Threshold = 0.50"
    )

    plt.xlabel("Predicted ATTACK score")
    plt.ylabel("Number of records")
    plt.title("Day 16 - DDoS Attack Score Distribution")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT / "attack_score_distribution.png",
        dpi=150
    )
    plt.close()

    summary = {
        "experiment": "Day 16 false-negative analysis",
        "threshold": THRESHOLD,
        "test_records": int(len(y_test)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "attack_recall": float(
            tp / (tp + fn)
        ),
        "day15_reproduced": True,
        "note": (
            "Held-out test data used for diagnostic "
            "analysis only, not threshold selection."
        )
    }

    (OUTPUT / "analysis_summary.json").write_text(
        json.dumps(summary, indent=4),
        encoding="utf-8"
    )

    print("\nDAY 16 ANALYSIS COMPLETED")
    print("Results saved to:", OUTPUT)


if __name__ == "__main__":
    main()
