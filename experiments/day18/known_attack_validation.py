
"""
DAY 18 - KNOWN-ATTACK VALIDATION

ML-Based Network Intrusion Detection System

Objectives:
1. Load Wednesday CICIDS2017 traffic.
2. Clean invalid network features.
3. Remove exact duplicate feature records.
4. Exclude categories with insufficient samples.
5. Create an 80/20 stratified train-validation split.
6. Train standard and balanced XGBoost models.
7. Compare classification thresholds.
8. Save evaluation results.

IMPORTANT:
This is within-capture validation.
It does not establish independent-session
or unseen-attack generalization.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ==========================================================
# PROJECT CONFIGURATION
# ==========================================================

ROOT = Path(__file__).resolve().parents[2]

OUTPUT = ROOT / "experiments" / "day18"
OUTPUT.mkdir(parents=True, exist_ok=True)

DATASET = Path(
    r"C:\Users\admin\IDS_Datasets\MachineLearningCVE"
) / "Wednesday-workingHours.pcap_ISCX.csv"

FEATURE_FILE = ROOT / "data" / "X_train.csv"

SAMPLE_FRACTION = 0.05
CHUNK_SIZE = 20000
RANDOM_STATE = 42

TEST_SIZE = 0.20

MIN_CLASS_RECORDS = 5

THRESHOLDS = [0.30, 0.50, 0.70]


# ==========================================================
# LOAD AND CLEAN DATA
# ==========================================================

def load_data(feature_names):

    pieces = []

    print("\nLoading Wednesday CICIDS2017 dataset...")

    for chunk in pd.read_csv(
        DATASET,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):

        chunk.columns = chunk.columns.str.strip()

        sample = chunk.sample(
            frac=SAMPLE_FRACTION,
            random_state=RANDOM_STATE
        )

        pieces.append(sample)

    if not pieces:
        raise RuntimeError("No records were loaded.")

    data = pd.concat(
        pieces,
        ignore_index=True
    )

    if "Label" not in data.columns:
        raise ValueError("Label column not found.")

    data["Label"] = (
        data["Label"]
        .astype("string")
        .str.strip()
    )

    # Convert features to numeric values.

    X = data[feature_names].apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Replace infinite values.

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    cleaned = pd.concat(
        [X, data[["Label"]]],
        axis=1
    )

    # Remove invalid rows.

    before_cleaning = len(cleaned)

    cleaned = cleaned.dropna().reset_index(
        drop=True
    )

    print(
        "Invalid records removed:",
        before_cleaning - len(cleaned)
    )

    # Convert labels to binary classes.

    cleaned["BinaryLabel"] = (
        cleaned["Label"] != "BENIGN"
    ).astype(int)

    # Remove exact duplicate feature records.

    before_duplicates = len(cleaned)

    cleaned = cleaned.drop_duplicates(
        subset=feature_names
    ).reset_index(drop=True)

    print(
        "Duplicate records removed:",
        before_duplicates - len(cleaned)
    )

    return cleaned


# ==========================================================
# REMOVE RARE ATTACK CATEGORIES
# ==========================================================

def remove_rare_categories(data):

    print("\nORIGINAL ATTACK DISTRIBUTION")
    print(data["Label"].value_counts())

    label_counts = data["Label"].value_counts()

    rare_labels = label_counts[
        label_counts < MIN_CLASS_RECORDS
    ].index.tolist()

    print("\nRare categories excluded:")

    if rare_labels:
        for label in rare_labels:
            print(
                f" - {label}: "
                f"{label_counts[label]} record(s)"
            )
    else:
        print("None")

    filtered = data[
        ~data["Label"].isin(rare_labels)
    ].copy().reset_index(drop=True)

    if filtered.empty:
        raise RuntimeError(
            "No records remain after filtering."
        )

    if filtered["BinaryLabel"].nunique() != 2:
        raise RuntimeError(
            "Both BENIGN and ATTACK classes are required."
        )

    print("\nFILTERED ATTACK DISTRIBUTION")
    print(filtered["Label"].value_counts())

    return filtered, rare_labels


# ==========================================================
# EVALUATION FUNCTION
# ==========================================================

def evaluate_model(
    y_true,
    probabilities,
    threshold
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    return {
        "threshold": float(threshold),

        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),

        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),

        "f1_score": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),

        "false_positive_rate": float(
            fp / (fp + tn)
            if fp + tn > 0 else 0
        ),

        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp)
    }


# ==========================================================
# MAIN EXPERIMENT
# ==========================================================

def main():

    print("=" * 60)
    print("DAY 18 - KNOWN-ATTACK VALIDATION")
    print("=" * 60)

    # ------------------------------------------------------
    # STEP 1 - Verify required files
    # ------------------------------------------------------

    if not DATASET.exists():
        raise FileNotFoundError(DATASET)

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(FEATURE_FILE)

    # ------------------------------------------------------
    # STEP 2 - Load feature names
    # ------------------------------------------------------

    feature_names = pd.read_csv(
        FEATURE_FILE,
        nrows=0
    ).columns.str.strip().tolist()

    print("\nExpected features:", len(feature_names))

    if len(feature_names) != 78:
        raise ValueError(
            "Expected exactly 78 network features."
        )

    # ------------------------------------------------------
    # STEP 3 - Load and clean dataset
    # ------------------------------------------------------

    data = load_data(feature_names)

    print("\nCleaned records:", len(data))

    # ------------------------------------------------------
    # STEP 4 - Remove rare categories
    # ------------------------------------------------------

    data, rare_labels = remove_rare_categories(
        data
    )

    print(
        "\nRecords after rare-category filtering:",
        len(data)
    )

    # ------------------------------------------------------
    # STEP 5 - Separate features and labels
    # ------------------------------------------------------

    X = data[feature_names]

    y = data["BinaryLabel"].astype(int)

    original_labels = data["Label"]

    # ------------------------------------------------------
    # STEP 6 - Stratified train-validation split
    # ------------------------------------------------------

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=original_labels
    )

    print("\nTRAINING AND VALIDATION SUMMARY")
    print("--------------------------------")

    print("Training records:", len(X_train))
    print("Validation records:", len(X_val))

    print("\nTraining class distribution:")
    print(y_train.value_counts().sort_index())

    print("\nValidation class distribution:")
    print(y_val.value_counts().sort_index())

    # ------------------------------------------------------
    # STEP 7 - Calculate class imbalance ratio
    # ------------------------------------------------------

    benign_count = int((y_train == 0).sum())
    attack_count = int((y_train == 1).sum())

    if attack_count == 0:
        raise RuntimeError(
            "No ATTACK records found in training."
        )

    class_ratio = benign_count / attack_count

    print(
        "\nCalculated class-weight ratio:",
        round(class_ratio, 4)
    )

    # ------------------------------------------------------
    # STEP 8 - Define experimental configurations
    # ------------------------------------------------------

    configurations = {
        "standard": 1.0,
        "balanced": class_ratio
    }

    results = []

    # ------------------------------------------------------
    # STEP 9 - Train XGBoost models
    # ------------------------------------------------------

    for name, weight in configurations.items():

        print("\n" + "-" * 60)
        print("Training configuration:", name)
        print("Class weight:", round(weight, 4))

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
            scale_pos_weight=weight
        )

        model.fit(
            X_train,
            y_train
        )

        probabilities = model.predict_proba(
            X_val
        )[:, 1]

        # --------------------------------------------------
        # STEP 10 - Evaluate classification thresholds
        # --------------------------------------------------

        for threshold in THRESHOLDS:

            metrics = evaluate_model(
                y_val,
                probabilities,
                threshold
            )

            metrics["configuration"] = name
            metrics["class_weight"] = float(weight)

            results.append(metrics)

            print(
                f"\nThreshold: {threshold}"
            )

            print(
                "Accuracy:",
                f"{metrics['accuracy']:.4%}"
            )

            print(
                "Precision:",
                f"{metrics['precision']:.4%}"
            )

            print(
                "Recall:",
                f"{metrics['recall']:.4%}"
            )

            print(
                "F1-score:",
                f"{metrics['f1_score']:.4%}"
            )

            print(
                "False positives:",
                metrics["false_positives"]
            )

            print(
                "False negatives:",
                metrics["false_negatives"]
            )

    # ------------------------------------------------------
    # STEP 11 - Save comparison results
    # ------------------------------------------------------

    comparison = pd.DataFrame(results)

    comparison_path = (
        OUTPUT / "known_attack_comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False
    )

    # ------------------------------------------------------
    # STEP 12 - Select best validation configuration
    # ------------------------------------------------------

    ranked = comparison.sort_values(
        ["f1_score", "false_positive_rate"],
        ascending=[False, True]
    )

    best = ranked.iloc[0]

    print("\n" + "=" * 60)
    print("BEST VALIDATION CONFIGURATION")
    print("=" * 60)

    print(
        "Configuration:",
        best["configuration"]
    )

    print(
        "Threshold:",
        best["threshold"]
    )

    print(
        "Validation F1-score:",
        f"{best['f1_score']:.4%}"
    )

    # ------------------------------------------------------
    # STEP 13 - Save experiment summary
    # ------------------------------------------------------

    summary = {
        "experiment": "Day 18 known-attack validation",

        "dataset": DATASET.name,

        "validation_type": (
            "Within-capture stratified random split"
        ),

        "sample_fraction": SAMPLE_FRACTION,

        "random_state": RANDOM_STATE,

        "total_records_after_filtering": int(
            len(data)
        ),

        "training_records": int(
            len(X_train)
        ),

        "validation_records": int(
            len(X_val)
        ),

        "excluded_rare_categories": [
            str(label) for label in rare_labels
        ],

        "class_weight_ratio": float(
            class_ratio
        ),

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
            "Training and validation use the same capture.",
            "Related network flows may appear in both subsets.",
            "Exact duplicate feature records were removed.",
            "Rare categories with fewer than five sampled records were excluded.",
            "Results do not establish independent-session generalization.",
            "The Friday DDoS holdout was not used."
        ]
    }

    summary_path = (
        OUTPUT / "known_attack_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=4
        ),
        encoding="utf-8"
    )

    # ------------------------------------------------------
    # STEP 14 - Completion
    # ------------------------------------------------------

    print("\n" + "=" * 60)
    print("DAY 18 EXPERIMENT COMPLETED!")
    print("=" * 60)

    print("\nResults saved to:")
    print(OUTPUT)

    print("\nGenerated files:")
    print(comparison_path.name)
    print(summary_path.name)


# ==========================================================
# PROGRAM ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()
