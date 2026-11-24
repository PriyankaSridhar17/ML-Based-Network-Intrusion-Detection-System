
"""
DAY 17 - SOURCE-AWARE XGBOOST VALIDATION

Compare class weights and decision thresholds using
development-side CICIDS2017 captures only.

The Friday DDoS holdout is never loaded.
"""

from pathlib import Path
import importlib.util
import json

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


ROOT = Path(__file__).resolve().parents[2]
DAY15 = ROOT / "experiments" / "day15"
OUTPUT = ROOT / "experiments" / "day17"
OUTPUT.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [0.30, 0.50, 0.70]
RANDOM_STATE = 42

# Use two attack-containing captures for validation.
VALIDATION_FILES = {
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
}


def load_day15():
    script = DAY15 / "independent_validation.py"

    spec = importlib.util.spec_from_file_location(
        "day15_validation", script
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def main():
    print("=" * 60)
    print("DAY 17 - XGBOOST VALIDATION OPTIMIZATION")
    print("=" * 60)

    day15 = load_day15()
    features = day15.load_feature_names()

    source_files = sorted(
        day15.DATASET_FOLDER.glob("*.csv")
    )

    train_parts = []
    validation_parts = []

    for file in source_files:
        # Preserve the Friday DDoS holdout.
        if file.name == day15.HOLDOUT_FILE:
            continue

        sample = day15.load_clean_sample(file, features)

        if file.name in VALIDATION_FILES:
            validation_parts.append(sample)
        else:
            train_parts.append(sample)

    if not train_parts or not validation_parts:
        raise RuntimeError(
            "Training or validation data is missing."
        )

    train = pd.concat(train_parts, ignore_index=True)
    validation = pd.concat(
        validation_parts, ignore_index=True
    )

    # Remove duplicate training feature records.
    train = train.drop_duplicates(
        subset=features
    ).reset_index(drop=True)

    # Remove validation rows with identical feature
    # records appearing in training.
    train_hashes = set(
        pd.util.hash_pandas_object(
            train[features], index=False
        ).to_numpy()
    )

    validation_hashes = pd.util.hash_pandas_object(
        validation[features], index=False
    )

    validation = validation.loc[
        ~validation_hashes.isin(train_hashes)
    ].reset_index(drop=True)

    X_train = train[features]
    y_train = train["BinaryLabel"].astype(int)

    X_val = validation[features]
    y_val = validation["BinaryLabel"].astype(int)

    if y_train.nunique() != 2 or y_val.nunique() != 2:
        raise RuntimeError(
            "Both datasets must contain BENIGN and ATTACK."
        )

    ratio = (
        int((y_train == 0).sum())
        / int((y_train == 1).sum())
    )

    print("\nTraining records:", len(X_train))
    print("Validation records:", len(X_val))
    print("Training class ratio:", round(ratio, 4))
    print("\nTraining distribution:")
    print(y_train.value_counts().sort_index())
    print("\nValidation distribution:")
    print(y_val.value_counts().sort_index())

    configurations = {
        "standard": 1.0,
        "balanced": ratio,
    }

    results = []

    for name, weight in configurations.items():
        print(f"\nTraining configuration: {name}")

        model = XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=4,
            scale_pos_weight=weight,
        )

        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_val)[:, 1]

        for threshold in THRESHOLDS:
            predictions = (
                probabilities >= threshold
            ).astype(int)

            tn, fp, fn, tp = confusion_matrix(
                y_val,
                predictions,
                labels=[0, 1]
            ).ravel()

            results.append({
                "configuration": name,
                "class_weight": weight,
                "threshold": threshold,
                "accuracy": accuracy_score(
                    y_val, predictions
                ),
                "precision": precision_score(
                    y_val, predictions,
                    zero_division=0
                ),
                "recall": recall_score(
                    y_val, predictions,
                    zero_division=0
                ),
                "f1_score": f1_score(
                    y_val, predictions,
                    zero_division=0
                ),
                "false_positive_rate": (
                    fp / (fp + tn)
                    if fp + tn else 0
                ),
                "false_positives": int(fp),
                "false_negatives": int(fn),
            })

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT / "validation_comparison.csv",
        index=False
    )

    print("\nVALIDATION RESULTS")
    print(results_df.to_string(index=False))

    # Example selection criterion:
    # Highest F1, with lower false-positive rate
    # used to break ties.
    ranked = results_df.sort_values(
        ["f1_score", "false_positive_rate"],
        ascending=[False, True]
    )

    best = ranked.iloc[0]

    summary = {
        "experiment": "Day 17 validation optimization",
        "training_sources": sorted(
            train["SourceFile"].unique().tolist()
        ),
        "validation_sources": sorted(
            validation["SourceFile"].unique().tolist()
        ),
        "excluded_holdout": day15.HOLDOUT_FILE,
        "training_records": int(len(train)),
        "validation_records": int(len(validation)),
        "class_weight_ratio": float(ratio),
        "selection_criterion": "validation F1-score",
        "selected_configuration": str(
            best["configuration"]
        ),
        "selected_threshold": float(
            best["threshold"]
        ),
        "selected_validation_f1": float(
            best["f1_score"]
        ),
        "limitations": [
            "Validation includes known development attack types.",
            "Validation is based on two source captures.",
            "Results do not establish DDoS generalization.",
            "The held-out DDoS capture was not used for tuning."
        ]
    }

    (OUTPUT / "validation_summary.json").write_text(
        json.dumps(summary, indent=4),
        encoding="utf-8"
    )

    print("\nDAY 17 EXPERIMENT COMPLETED")
    print("Results saved to:", OUTPUT)


if __name__ == "__main__":
    main()
