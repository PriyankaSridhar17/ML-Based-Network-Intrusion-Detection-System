
"""
DAY 15 - INDEPENDENT XGBOOST VALIDATION

ML-Based Network Intrusion Detection System

Author: Priyanka Sridhar

Experiment:
- Train using seven CICIDS2017 source files.
- Hold out the Friday DDoS source file.
- Evaluate detection of an unseen attack category.
- Save metrics, confusion matrix, and model separately.

IMPORTANT:
This experiment does not modify the existing
XGBoost baseline or Flask application.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ==================================================
# STEP 1: CONFIGURATION
# ==================================================

DATASET_FOLDER = Path(
    r"C:\Users\admin\IDS_Datasets\MachineLearningCVE"
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_FOLDER = (
    PROJECT_ROOT / "experiments" / "day15"
)

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

HOLDOUT_FILE = (
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)

SAMPLE_FRACTION = 0.05
CHUNK_SIZE = 20000
RANDOM_STATE = 42

# Keep experimental artifacts separate.
MODEL_OUTPUT = OUTPUT_FOLDER / "experimental_xgboost.json"
METRICS_OUTPUT = OUTPUT_FOLDER / "validation_metrics.csv"
CONFUSION_OUTPUT = OUTPUT_FOLDER / "confusion_matrix.png"
REPORT_OUTPUT = OUTPUT_FOLDER / "classification_report.txt"
SUMMARY_OUTPUT = OUTPUT_FOLDER / "experiment_summary.json"

# Existing baseline feature names.
BASELINE_FEATURE_FILE = (
    PROJECT_ROOT / "data" / "X_train.csv"
)


# ==================================================
# STEP 2: LOAD EXPECTED FEATURE NAMES
# ==================================================

def load_feature_names():

    if not BASELINE_FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Baseline feature file not found: "
            f"{BASELINE_FEATURE_FILE}"
        )

    columns = pd.read_csv(
        BASELINE_FEATURE_FILE,
        nrows=0
    ).columns.tolist()

    if len(columns) != 78:
        raise ValueError(
            f"Expected 78 features, found {len(columns)}."
        )

    if len(set(columns)) != len(columns):
        raise ValueError(
            "Duplicate feature names detected."
        )

    return columns


# ==================================================
# STEP 3: LOAD AND CLEAN SOURCE CSV
# ==================================================

def load_clean_sample(file_path, feature_names):

    print(f"\nProcessing: {file_path.name}", flush=True)

    samples = []
    sampled_rows = 0
    cleaned_rows = 0

    for chunk in pd.read_csv(
        file_path,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):

        chunk.columns = chunk.columns.str.strip()

        # Pandas may add suffixes to duplicate source columns.
        # Match the saved baseline feature schema exactly.
        required_columns = feature_names + ["Label"]

        missing = [
            col for col in required_columns
            if col not in chunk.columns
        ]

        if missing:
            raise ValueError(
                f"Missing columns in {file_path.name}: "
                f"{missing}"
            )

        sample = chunk.sample(
            frac=SAMPLE_FRACTION,
            random_state=RANDOM_STATE
        )

        sampled_rows += len(sample)

        if sample.empty:
            continue

        labels = (
            sample["Label"]
            .astype("string")
            .str.strip()
        )

        features = sample[
            feature_names
        ].apply(
            pd.to_numeric,
            errors="coerce"
        )

        features = features.replace(
            [np.inf, -np.inf],
            np.nan
        )

        valid_mask = (
            features.notna().all(axis=1)
            & labels.notna()
            & labels.ne("")
        )

        features = features.loc[valid_mask]
        labels = labels.loc[valid_mask]

        if features.empty:
            continue

        cleaned = features.copy()

        cleaned["BinaryLabel"] = (
            labels.ne("BENIGN")
            .astype(int)
            .to_numpy()
        )

        cleaned["SourceFile"] = file_path.name

        cleaned_rows += len(cleaned)
        samples.append(cleaned)

    if not samples:
        raise ValueError(
            f"No usable records in {file_path.name}"
        )

    result = pd.concat(
        samples,
        ignore_index=True
    )

    print(f"Sampled rows: {sampled_rows:,}")
    print(f"Cleaned rows: {cleaned_rows:,}")

    return result


# ==================================================
# STEP 4: LOAD TRAINING AND HOLDOUT DATA
# ==================================================

def prepare_datasets(feature_names):

    csv_files = sorted(
        DATASET_FOLDER.glob("*.csv")
    )

    if len(csv_files) != 8:
        raise RuntimeError(
            f"Expected 8 original CSV files, "
            f"found {len(csv_files)}."
        )

    holdout_path = (
        DATASET_FOLDER / HOLDOUT_FILE
    )

    if not holdout_path.exists():
        raise FileNotFoundError(
            f"Holdout file not found: {holdout_path}"
        )

    training_files = [
        file for file in csv_files
        if file.name != HOLDOUT_FILE
    ]

    print("\nTRAINING SOURCE FILES:")

    for file in training_files:
        print(" -", file.name)

    print("\nHELD-OUT TEST SOURCE:")
    print(" -", holdout_path.name)

    training_parts = []

    for file in training_files:
        part = load_clean_sample(
            file,
            feature_names
        )
        training_parts.append(part)

    training_data = pd.concat(
        training_parts,
        ignore_index=True
    )

    test_data = load_clean_sample(
        holdout_path,
        feature_names
    )

    return training_data, test_data, training_files


# ==================================================
# STEP 5: CHECK DATASET SEPARATION
# ==================================================

def verify_separation(train_data, test_data):

    train_sources = set(
        train_data["SourceFile"].unique()
    )

    test_sources = set(
        test_data["SourceFile"].unique()
    )

    overlap = train_sources.intersection(
        test_sources
    )

    if overlap:
        raise RuntimeError(
            f"Source-file overlap detected: {overlap}"
        )

    print("\nSOURCE SEPARATION CHECK")
    print("Training sources:", len(train_sources))
    print("Testing sources:", len(test_sources))
    print("Overlapping source files:", len(overlap))

    assert len(overlap) == 0


# ==================================================
# STEP 6: PREPARE FEATURE MATRICES
# ==================================================

def prepare_features(
    training_data,
    test_data,
    feature_names
):

    # Remove duplicate training feature records.
    before = len(training_data)

    training_data = training_data.drop_duplicates(
        subset=feature_names
    ).reset_index(drop=True)

    removed = before - len(training_data)

    print("\nDuplicate training rows removed:", removed)

    X_train = training_data[feature_names].copy()
    y_train = training_data["BinaryLabel"].astype(int)

    X_test = test_data[feature_names].copy()
    y_test = test_data["BinaryLabel"].astype(int)

    if y_train.nunique() != 2:
        raise ValueError(
            "Training data must contain both classes."
        )

    print("\nTRAINING CLASS DISTRIBUTION")
    print(y_train.value_counts().sort_index())

    print("\nTEST CLASS DISTRIBUTION")
    print(y_test.value_counts().sort_index())

    print("\nTraining records:", len(X_train))
    print("Testing records:", len(X_test))

    return X_train, X_test, y_train, y_test


# ==================================================
# STEP 7: TRAIN EXPERIMENTAL XGBOOST
# ==================================================

def train_experimental_model(X_train, y_train):

    print("\nTRAINING EXPERIMENTAL XGBOOST MODEL")

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
        n_jobs=4
    )

    model.fit(
        X_train,
        y_train
    )

    model.save_model(
        str(MODEL_OUTPUT)
    )

    print("Experimental model saved:")
    print(MODEL_OUTPUT)

    return model


# ==================================================
# STEP 8: EVALUATE HOLDOUT PERFORMANCE
# ==================================================

def evaluate_model(model, X_test, y_test):

    print("\nEVALUATING HELD-OUT DDoS CAPTURE")

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

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

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "True_Negatives": int(tn),
        "False_Positives": int(fp),
        "False_Negatives": int(fn),
        "True_Positives": int(tp),
        "Test_Records": len(y_test)
    }

    print("\nINDEPENDENT VALIDATION RESULTS")
    print("-" * 45)

    print(f"Accuracy : {accuracy:.4%}")
    print(f"Precision: {precision:.4%}")
    print(f"Recall   : {recall:.4%}")
    print(f"F1-score : {f1:.4%}")

    print("\nCONFUSION MATRIX")
    print(cm)

    print("\nFalse positives:", fp)
    print("False negatives:", fn)

    pd.DataFrame(
        [metrics]
    ).to_csv(
        METRICS_OUTPUT,
        index=False
    )

    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["BENIGN", "ATTACK"],
        zero_division=0
    )

    REPORT_OUTPUT.write_text(
        report,
        encoding="utf-8"
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["BENIGN", "ATTACK"]
    )

    display.plot(
        cmap="Blues",
        values_format="d"
    )

    plt.title(
        "Day 15 - Held-Out DDoS Validation"
    )

    plt.tight_layout()

    plt.savefig(
        CONFUSION_OUTPUT,
        dpi=150
    )

    plt.close()

    return metrics


# ==================================================
# STEP 9: SAVE EXPERIMENT SUMMARY
# ==================================================

def save_summary(
    training_files,
    X_train,
    X_test,
    metrics
):

    summary = {
        "experiment": "Day 15 source-file holdout",
        "dataset": "CICIDS2017",
        "training_source_files": [
            file.name for file in training_files
        ],
        "held_out_source_file": HOLDOUT_FILE,
        "sample_fraction": SAMPLE_FRACTION,
        "random_state": RANDOM_STATE,
        "training_records": len(X_train),
        "testing_records": len(X_test),
        "feature_count": X_train.shape[1],
        "evaluation_type": (
            "Unseen DDoS attack category evaluation"
        ),
        "metrics": metrics,
        "limitations": [
            "Source-file holdout is not a guarantee "
            "of independent network sessions.",
            "DDoS is absent from the training labels.",
            "Results are based on sampled CICIDS2017 "
            "network traffic.",
            "This is not live-network validation."
        ]
    }

    SUMMARY_OUTPUT.write_text(
        json.dumps(
            summary,
            indent=4
        ),
        encoding="utf-8"
    )


# ==================================================
# STEP 10: MAIN EXPERIMENT
# ==================================================

def main():

    print("=" * 60)
    print("DAY 15 - INDEPENDENT XGBOOST VALIDATION")
    print("=" * 60)

    if not DATASET_FOLDER.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATASET_FOLDER}"
        )

    feature_names = load_feature_names()

    print("\nExpected network features:", len(feature_names))

    train_data, test_data, training_files = (
        prepare_datasets(feature_names)
    )

    verify_separation(
        train_data,
        test_data
    )

    X_train, X_test, y_train, y_test = (
        prepare_features(
            train_data,
            test_data,
            feature_names
        )
    )

    model = train_experimental_model(
        X_train,
        y_train
    )

    metrics = evaluate_model(
        model,
        X_test,
        y_test
    )

    save_summary(
        training_files,
        X_train,
        X_test,
        metrics
    )

    print("\n" + "=" * 60)
    print("DAY 15 VALIDATION EXPERIMENT COMPLETED")
    print("=" * 60)

    print("\nResults saved to:")
    print(OUTPUT_FOLDER)


if __name__ == "__main__":
    main()
