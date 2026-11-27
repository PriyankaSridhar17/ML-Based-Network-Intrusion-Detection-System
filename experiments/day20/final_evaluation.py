
"""
DAY 20 - FINAL CONSOLIDATED MODEL EVALUATION

ML-Based Network Intrusion Detection System
Author: Priyanka Sridhar

Reads previously generated research reports.
Does not retrain models or modify original results.

Outputs:
- consolidated_metrics.csv
- evaluation_comparison.png
- final_research_summary.json
- final_evaluation_report.txt
"""

from pathlib import Path
import json

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "experiments" / "day20"
OUTPUT.mkdir(parents=True, exist_ok=True)


def read_csv_row(path):
    if not path.exists():
        raise FileNotFoundError(path)

    data = pd.read_csv(path)

    if data.empty:
        raise ValueError(f"Empty report: {path}")

    return data.iloc[0]


def read_json(path):
    if not path.exists():
        raise FileNotFoundError(path)

    return json.loads(path.read_text(encoding="utf-8-sig"))


def make_record(
    name, evaluation_type, source,
    accuracy, precision, recall, f1,
    tn, fp, fn, tp
):
    values = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    if not all(
        np.isfinite(values[k])
        for k in (
            "accuracy", "precision",
            "recall", "f1_score"
        )
    ):
        raise ValueError(f"Invalid metrics: {name}")

    total = sum(
        values[k] for k in (
            "true_negatives",
            "false_positives",
            "false_negatives",
            "true_positives"
        )
    )

    if total <= 0:
        raise ValueError(f"Invalid record count: {name}")

    calculated_accuracy = (
        values["true_negatives"]
        + values["true_positives"]
    ) / total

    if not np.isclose(
        calculated_accuracy,
        values["accuracy"],
        atol=1e-6
    ):
        raise ValueError(
            f"Accuracy mismatch for {name}"
        )

    return {
        "experiment": name,
        "evaluation_type": evaluation_type,
        "source_report": source,
        "test_records": total,
        **values
    }


def main():
    print("=" * 60)
    print("DAY 20 - CONSOLIDATED MODEL EVALUATION")
    print("=" * 60)

    records = []

    # Day 11: Original random-split baseline
    baseline_path = (
        ROOT / "reports" / "day11_validation_metrics.csv"
    )
    b = read_csv_row(baseline_path)

    records.append(make_record(
        "Day 11 - Original Baseline",
        "Random row split",
        str(baseline_path.relative_to(ROOT)),
        b["Accuracy"], b["Precision"],
        b["Recall"], b["F1"],
        b["True_Negatives"], b["False_Positives"],
        b["False_Negatives"], b["True_Positives"]
    ))

    # Day 15: Held-out DDoS
    day15_path = (
        ROOT / "experiments/day15/validation_metrics.csv"
    )
    d15 = read_csv_row(day15_path)

    records.append(make_record(
        "Day 15 - Held-out DDoS",
        "Source-separated unseen attack",
        str(day15_path.relative_to(ROOT)),
        d15["Accuracy"], d15["Precision"],
        d15["Recall"], d15["F1_Score"],
        d15["True_Negatives"], d15["False_Positives"],
        d15["False_Negatives"], d15["True_Positives"]
    ))

    # Day 17: Selected validation configuration
    day17_path = (
        ROOT / "experiments/day17/validation_comparison.csv"
    )
    day17_summary = read_json(
        ROOT / "experiments/day17/validation_summary.json"
    )

    d17_table = pd.read_csv(day17_path)

    selected_rows = d17_table.loc[
        (d17_table["configuration"]
         == day17_summary["selected_configuration"])
        &
        np.isclose(
            d17_table["threshold"].astype(float),
            float(day17_summary["selected_threshold"])
        )
    ]

    if len(selected_rows) != 1:
        raise ValueError(
            "Day 17 selected configuration is ambiguous."
        )

    d17 = selected_rows.iloc[0]

    tn17 = round(
        d17["false_positives"]
        / d17["false_positive_rate"]
        - d17["false_positives"]
    ) if d17["false_positive_rate"] > 0 else None

    if tn17 is None:
        raise ValueError(
            "Day 17 true negatives cannot be inferred."
        )

    tp17 = round(
        d17["recall"]
        * d17["false_negatives"]
        / (1 - d17["recall"])
    )

    records.append(make_record(
        "Day 17 - Unseen Attack Validation",
        "Source-separated validation",
        str(day17_path.relative_to(ROOT)),
        d17["accuracy"], d17["precision"],
        d17["recall"], d17["f1_score"],
        tn17, d17["false_positives"],
        d17["false_negatives"], tp17
    ))

    # Day 18: Selected known-attack configuration
    day18_path = (
        ROOT / "experiments/day18/known_attack_comparison.csv"
    )
    day18_summary = read_json(
        ROOT / "experiments/day18/known_attack_summary.json"
    )

    d18_table = pd.read_csv(day18_path)

    selected_rows = d18_table.loc[
        (d18_table["configuration"]
         == day18_summary["selected_configuration"])
        &
        np.isclose(
            d18_table["threshold"].astype(float),
            float(day18_summary["selected_threshold"])
        )
    ]

    if len(selected_rows) != 1:
        raise ValueError(
            "Day 18 selected configuration is ambiguous."
        )

    d18 = selected_rows.iloc[0]

    records.append(make_record(
        "Day 18 - Known Attack Validation",
        "Within-capture validation",
        str(day18_path.relative_to(ROOT)),
        d18["accuracy"], d18["precision"],
        d18["recall"], d18["f1_score"],
        d18["true_negatives"], d18["false_positives"],
        d18["false_negatives"], d18["true_positives"]
    ))

    # Day 19: Supplementary evaluation
    day19_path = (
        ROOT / "experiments/day19/selected_model_evaluation.csv"
    )
    d19 = read_csv_row(day19_path)

    records.append(make_record(
        "Day 19 - Model Comparison",
        "Supplementary within-capture evaluation",
        str(day19_path.relative_to(ROOT)),
        d19["accuracy"], d19["precision"],
        d19["recall"], d19["f1_score"],
        d19["true_negatives"], d19["false_positives"],
        d19["false_negatives"], d19["true_positives"]
    ))

    # Day 16 is diagnostic evidence, not another
    # independent performance benchmark.
    day16 = read_json(
        ROOT / "experiments/day16/analysis_summary.json"
    )

    # Save consolidated results
    comparison = pd.DataFrame(records)

    comparison.to_csv(
        OUTPUT / "consolidated_metrics.csv",
        index=False
    )

    print("\nCONSOLIDATED METRICS")
    print(
        comparison[
            [
                "experiment",
                "accuracy",
                "precision",
                "recall",
                "f1_score"
            ]
        ].to_string(index=False)
    )

    # Visualization
    labels = [
        "Baseline", "DDoS",
        "Unseen", "Known", "Model comparison"
    ]

    x = np.arange(len(labels))
    width = 0.19

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, metric in enumerate(
        ["accuracy", "precision", "recall", "f1_score"]
    ):
        ax.bar(
            x + (i - 1.5) * width,
            comparison[metric] * 100,
            width,
            label=metric.replace("_", " ").title()
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Metric (%)")
    ax.set_ylim(0, 105)
    ax.set_title(
        "CICIDS2017 - Evaluation Results Across Experiments"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.2)

    fig.text(
        0.5, 0.01,
        "Different datasets and experimental designs; "
        "not a controlled comparison.",
        ha="center", fontsize=9
    )

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(
        OUTPUT / "evaluation_comparison.png",
        dpi=160
    )
    plt.close(fig)

    findings = [
        "Within-capture known-attack evaluation "
        "produced high classification performance.",
        "Source-separated experiments exposed "
        "poor generalization to some unseen attacks.",
        "Day 16 showed many missed DDoS attacks "
        "received very low ATTACK scores.",
        "Increasing XGBoost complexity provided "
        "negligible within-capture improvement.",
        "The reported evaluations do not establish "
        "live-network or production performance."
    ]

    limitations = [
        "Experiments used different training data "
        "and model configurations.",
        "Within-capture splits may contain related flows.",
        "Source-file separation does not guarantee "
        "independent network sessions.",
        "Some configurations were selected using "
        "the reported validation datasets.",
        "The project has not been evaluated on "
        "independent live network traffic."
    ]

    summary = {
        "project": "ML-Based Network Intrusion Detection System",
        "evaluation_scope": "Days 11 and 15-19",
        "comparison_is_controlled": False,
        "experiment_count": len(records),
        "day16_false_negative_analysis": {
            "false_negatives": day16["false_negatives"],
            "attack_recall": day16["attack_recall"],
            "day15_reproduced": day16["day15_reproduced"]
        },
        "research_findings": findings,
        "limitations": limitations
    }

    (OUTPUT / "final_research_summary.json").write_text(
        json.dumps(summary, indent=4),
        encoding="utf-8"
    )

    report_lines = [
        "DAY 20 - CONSOLIDATED RESEARCH EVALUATION",
        "=" * 60,
        "",
        "This report summarizes the existing experimental",
        "results without retraining any classifiers.",
        "",
        comparison.to_string(index=False),
        "",
        "RESEARCH FINDINGS",
        "-" * 60,
    ]

    report_lines.extend(
        f"{i}. {finding}"
        for i, finding in enumerate(findings, 1)
    )

    report_lines.extend([
        "",
        "LIMITATIONS",
        "-" * 60
    ])

    report_lines.extend(
        f"{i}. {item}"
        for i, item in enumerate(limitations, 1)
    )

    report_lines.extend([
        "",
        "CONCLUSION",
        "-" * 60,
        "High within-capture performance did not guarantee",
        "strong recognition of attack categories excluded",
        "from training. Additional independent evaluation",
        "is required before production claims can be made."
    ])

    (OUTPUT / "final_evaluation_report.txt").write_text(
        "\n".join(report_lines),
        encoding="utf-8"
    )

    print("\nDAY 20 CONSOLIDATION COMPLETED")
    print("Results saved to:", OUTPUT)


if __name__ == "__main__":
    main()
