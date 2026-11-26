
"""
DAY 19 - CONTROLLED XGBOOST MODEL COMPARISON

ML-Based Network Intrusion Detection System

Compare four XGBoost configurations using a
60/20/20 within-capture development split.

IMPORTANT:
- No existing trained models are overwritten.
- The Friday DDoS capture is not loaded.
- Evaluation remains within the Wednesday capture.
- Related network flows may cross split boundaries.
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

# ----------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "experiments" / "day19"
OUTPUT.mkdir(parents=True, exist_ok=True)

DATASET = Path(
    r"C:\Users\admin\IDS_Datasets\MachineLearningCVE"
) / "Wednesday-workingHours.pcap_ISCX.csv"

FEATURE_FILE = ROOT / "data" / "X_train.csv"

SAMPLE_FRACTION = 0.05
CHUNK_SIZE = 20000
RANDOM_STATE = 42
MIN_CLASS_RECORDS = 5
THRESHOLD = 0.50

CONFIGURATIONS = {
    "baseline": {"max_depth": 5, "n_estimators": 150},
    "shallow": {"max_depth": 3, "n_estimators": 150},
    "deep": {"max_depth": 7, "n_estimators": 150},
    "more_trees": {"max_depth": 5, "n_estimators": 250},
}


def load_and_clean(feature_names):
    pieces = []

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
        raise RuntimeError("No dataset records loaded.")

    data = pd.concat(pieces, ignore_index=True)

    if "Label" not in data.columns:
        raise ValueError("Label column is missing.")

    data["Label"] = (
        data["Label"].astype("string").str.strip()
    )

    features = data[feature_names].apply(
        pd.to_numeric, errors="coerce"
    )
    features = features.replace(
        [np.inf, -np.inf], np.nan
    )

    cleaned = pd.concat(
        [features, data[["Label"]]], axis=1
    ).dropna().reset_index(drop=True)

    before = len(cleaned)
    cleaned = cleaned.drop_duplicates(
        subset=feature_names
    ).reset_index(drop=True)

    print("Duplicate records removed:", before - len(cleaned))

    counts = cleaned["Label"].value_counts()
    rare = counts[counts < MIN_CLASS_RECORDS].index.tolist()

    cleaned = cleaned[
        ~cleaned["Label"].isin(rare)
    ].copy().reset_index(drop=True)

    cleaned["BinaryLabel"] = (
        cleaned["Label"] != "BENIGN"
    ).astype(int)

    print("Excluded rare categories:", rare)
    print("Final cleaned records:", len(cleaned))

    return cleaned, rare


def calculate_metrics(y_true, probabilities):
    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
        "false_positive_rate": float(
            fp / (fp + tn) if fp + tn else 0
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def main():
    print("=" * 60)
    print("DAY 19 - CONTROLLED XGBOOST MODEL COMPARISON")
    print("=" * 60)

    if not DATASET.exists():
        raise FileNotFoundError(DATASET)

    feature_names = pd.read_csv(
        FEATURE_FILE, nrows=0
    ).columns.str.strip().tolist()

    if len(feature_names) != 78:
        raise ValueError("Expected 78 input features.")

    data, rare_labels = load_and_clean(feature_names)

    # First reserve 20% for supplementary evaluation.
    development, evaluation = train_test_split(
        data,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=data["Label"],
    )

    # Split the remaining 80% into 60% training
    # and 20% validation of the original dataset.
    training, validation = train_test_split(
        development,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=development["Label"],
    )

    X_train = training[feature_names]
    y_train = training["BinaryLabel"].astype(int)

    X_val = validation[feature_names]
    y_val = validation["BinaryLabel"].astype(int)

    X_eval = evaluation[feature_names]
    y_eval = evaluation["BinaryLabel"].astype(int)

    print("\nDATASET SPLIT")
    print("Training:", len(training))
    print("Validation:", len(validation))
    print("Supplementary evaluation:", len(evaluation))

    if any(
        subset["BinaryLabel"].nunique() != 2
        for subset in (training, validation, evaluation)
    ):
        raise RuntimeError("Both binary classes required.")

    validation_results = []

    for name, params in CONFIGURATIONS.items():
        print("\nTraining:", name, params)

        model = XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=4,
            scale_pos_weight=1.0,
        )

        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_val)[:, 1]
        metrics = calculate_metrics(y_val, probabilities)

        metrics.update({
            "configuration": name,
            "max_depth": params["max_depth"],
            "n_estimators": params["n_estimators"],
            "dataset": "validation",
        })

        validation_results.append(metrics)

        print(
            f"Validation F1: {metrics['f1_score']:.4%}, "
            f"Recall: {metrics['recall']:.4%}, "
            f"False positives: {metrics['false_positives']}"
        )

    comparison = pd.DataFrame(validation_results)

    comparison.to_csv(
        OUTPUT / "model_comparison.csv", index=False
    )

    ranked = comparison.sort_values(
        ["f1_score", "false_positive_rate", "configuration"],
        ascending=[False, True, True],
    )

    selected = ranked.iloc[0]
    selected_name = str(selected["configuration"])

    print("\nSELECTED CONFIGURATION:", selected_name)

    # Refit the selected configuration using the original
    # training subset only. Evaluation remains separate.
    params = CONFIGURATIONS[selected_name]

    selected_model = XGBClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=4,
        scale_pos_weight=1.0,
    )

    selected_model.fit(X_train, y_train)

    eval_probabilities = selected_model.predict_proba(
        X_eval
    )[:, 1]

    evaluation_metrics = calculate_metrics(
        y_eval, eval_probabilities
    )

    print("\nSUPPLEMENTARY EVALUATION RESULTS")
    for key, value in evaluation_metrics.items():
        print(f"{key}: {value}")

    pd.DataFrame([{
        "configuration": selected_name,
        **evaluation_metrics
    }]).to_csv(
        OUTPUT / "selected_model_evaluation.csv",
        index=False
    )

    summary = {
        "experiment": "Day 19 controlled XGBoost comparison",
        "dataset": DATASET.name,
        "validation_type": "within-capture stratified 60/20/20",
        "training_records": int(len(training)),
        "validation_records": int(len(validation)),
        "evaluation_records": int(len(evaluation)),
        "excluded_rare_categories": [
            str(label) for label in rare_labels
        ],
        "classification_threshold": THRESHOLD,
        "selection_criterion": "validation F1-score",
        "selected_configuration": selected_name,
        "selected_parameters": params,
        "selected_validation_f1": float(
            selected["f1_score"]
        ),
        "supplementary_evaluation": evaluation_metrics,
        "limitations": [
            "All subsets originate from one Wednesday capture.",
            "Related network sessions may cross split boundaries.",
            "The Wednesday capture was used in prior experiments.",
            "Supplementary evaluation is not a fully untouched final test.",
            "No independent-capture generalization claim is made.",
            "The Friday DDoS holdout was not used."
        ],
    }

    (OUTPUT / "model_comparison_summary.json").write_text(
        json.dumps(summary, indent=4),
        encoding="utf-8"
    )

    print("\nDAY 19 EXPERIMENT COMPLETED")
    print("Results saved to:", OUTPUT)


if __name__ == "__main__":
    main()
