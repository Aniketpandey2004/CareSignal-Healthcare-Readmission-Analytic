
"""
CareSignal - Experiment 6
Subgroup Performance and Fairness Analysis

Uses the existing saved model and original patient-level test split.
No model retraining. No changes to the deployed model or dashboard.

Run from project root:
python experiments/subgroup_fairness_analysis.py

Subgroups:
1. Age
2. Gender
3. Race

Metrics:
Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC,
Brier Score, TPR, FPR, Specificity and predicted positive rate.
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix
)


# ============================================================
# 1. SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "raw" / "diabetic_data_original.csv"

MODEL_PATH = ROOT / "models" / "healthcare_readmission_model.joblib"

OUT_DIR = ROOT / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_GROUP_SIZE = 100

SUBGROUP_COLUMNS = [
    "age",
    "gender",
    "race"
]

print("\n" + "=" * 70)
print("CARESIGNAL - EXPERIMENT 6")
print("SUBGROUP PERFORMANCE AND FAIRNESS ANALYSIS")
print("=" * 70)


# ============================================================
# 2. LOAD DATA AND MODEL
# ============================================================

print("\nLoading dataset and saved model...")

df = pd.read_csv(DATA_PATH, low_memory=False)

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]

threshold = float(artifact["threshold"])

print("Dataset shape:", df.shape)
print("Model:", artifact.get("model_name", "Saved model"))
print("Saved threshold:", threshold)


# ============================================================
# 3. REPRODUCE ORIGINAL PREPROCESSING
# ============================================================

print("\nReproducing original preprocessing...")

df.columns = df.columns.astype(str).str.strip()

duplicate_pairs = [
    ("patient_nbr.1", "patient_nbr"),
    ("encounter_id.1", "encounter_id"),
]

for duplicate, original in duplicate_pairs:

    if duplicate in df.columns:

        if not df[duplicate].equals(df[original]):
            raise ValueError(
                f"{duplicate} does not match {original}"
            )

        df = df.drop(columns=[duplicate])


y = (
    df["readmitted"]
    .astype(str)
    .str.strip()
    .eq("<30")
    .astype("int8")
)

groups = df["patient_nbr"].copy()

excluded_columns = {
    "patient_nbr",
    "encounter_id",
    "readmitted",
    "readmitted_30",
}

X = df.drop(
    columns=[
        col for col in excluded_columns
        if col in df.columns
    ]
).copy()

X = X.replace("?", np.nan)

coded_categorical = [
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
]

for col in coded_categorical:

    if col in X.columns:
        X[col] = X[col].astype("string")


all_missing = X.columns[
    X.isna().all()
].tolist()

if all_missing:
    X = X.drop(columns=all_missing)


features = artifact["feature_columns"]

if list(X.columns) != list(features):
    raise ValueError(
        "Feature schema mismatch with saved model."
    )

X = X[features].copy()


# ============================================================
# 4. CHECK SUBGROUP COLUMNS
# ============================================================

for col in SUBGROUP_COLUMNS:

    if col not in df.columns:
        raise ValueError(
            f"Required subgroup column missing: {col}"
        )


# ============================================================
# 5. RECREATE ORIGINAL PATIENT-LEVEL TEST SPLIT
# ============================================================

print("\nRecreating original patient-level test split...")

outer = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    outer.split(X, y, groups=groups)
)

X_test = X.iloc[test_idx].copy()

y_test = y.iloc[test_idx].copy()

test_groups = groups.iloc[test_idx]

train_patients = set(
    groups.iloc[train_idx].unique()
)

test_patients = set(
    test_groups.unique()
)

if train_patients & test_patients:
    raise RuntimeError(
        "Patient overlap detected."
    )

print("Test encounters:", len(X_test))
print("Test patients:", len(test_patients))
print("Patient overlap: 0")


# ============================================================
# 6. GET MODEL PREDICTIONS
# ============================================================

print("\nGenerating predictions using saved model...")

probabilities = model.predict_proba(X_test)[:, 1]

predictions = (
    probabilities >= threshold
).astype(int)

print("Predictions generated.")
print("Evaluation threshold:", threshold)


# ============================================================
# 7. PREPARE TEST SUBGROUP DATA
# ============================================================

subgroup_data = df.iloc[test_idx][
    SUBGROUP_COLUMNS
].copy()

subgroup_data = subgroup_data.reset_index(drop=True)

y_array = y_test.to_numpy()

probabilities = np.asarray(probabilities)

predictions = np.asarray(predictions)

for col in SUBGROUP_COLUMNS:

    subgroup_data[col] = (
        subgroup_data[col]
        .astype("string")
        .fillna("Unknown/Not recorded")
        .str.strip()
        .replace({
            "?": "Unknown/Not recorded",
            "": "Unknown/Not recorded",
            "nan": "Unknown/Not recorded"
        })
    )


# ============================================================
# 8. METRIC CALCULATION FUNCTION
# ============================================================

def calculate_metrics(y_true, y_pred, y_prob):

    y_true = np.asarray(y_true)

    y_pred = np.asarray(y_pred)

    y_prob = np.asarray(y_prob)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else np.nan
    )

    if len(np.unique(y_true)) == 2:

        roc_auc = roc_auc_score(
            y_true,
            y_prob
        )

        pr_auc = average_precision_score(
            y_true,
            y_prob
        )

    else:

        roc_auc = np.nan
        pr_auc = np.nan

    return {

        "sample_size": int(len(y_true)),

        "positive_cases": int(y_true.sum()),

        "negative_cases": int(len(y_true) - y_true.sum()),

        "actual_positive_rate": float(y_true.mean()),

        "predicted_positive_rate": float(y_pred.mean()),

        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),

        "precision": float(precision),

        "recall": float(recall),

        "f1_score": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),

        "roc_auc": float(roc_auc)
        if not np.isnan(roc_auc)
        else None,

        "pr_auc": float(pr_auc)
        if not np.isnan(pr_auc)
        else None,

        "brier_score": float(
            brier_score_loss(y_true, y_prob)
        ),

        "specificity": float(specificity)
        if not np.isnan(specificity)
        else None,

        "false_positive_rate": float(fpr)
        if not np.isnan(fpr)
        else None,

        "true_positives": int(tp),

        "false_positives": int(fp),

        "true_negatives": int(tn),

        "false_negatives": int(fn)

    }


# ============================================================
# 9. OVERALL TEST PERFORMANCE
# ============================================================

print("\nCalculating overall test performance...")

overall_metrics = calculate_metrics(
    y_array,
    predictions,
    probabilities
)

overall_metrics["subgroup"] = "Overall"

overall_metrics["category"] = "All test encounters"

overall_df = pd.DataFrame([overall_metrics])


# ============================================================
# 10. SUBGROUP ANALYSIS
# ============================================================

print("\nCalculating subgroup performance...")

results = []

for column in SUBGROUP_COLUMNS:

    print(f"\nAnalysing: {column}")

    categories = subgroup_data[column].unique()

    for category in categories:

        mask = (
            subgroup_data[column] == category
        ).to_numpy()

        group_size = int(mask.sum())

        if group_size < MIN_GROUP_SIZE:

            print(
                f"Skipping {category}: "
                f"only {group_size} encounters"
            )

            continue

        group_metrics = calculate_metrics(
            y_array[mask],
            predictions[mask],
            probabilities[mask]
        )

        group_metrics["subgroup"] = column

        group_metrics["category"] = str(category)

        results.append(group_metrics)

        print(
            f"{category}: "
            f"N={group_size}, "
            f"Recall={group_metrics['recall']:.4f}, "
            f"Precision={group_metrics['precision']:.4f}, "
            f"F1={group_metrics['f1_score']:.4f}"
        )


if not results:
    raise RuntimeError(
        "No subgroup met the minimum sample size."
    )

results_df = pd.DataFrame(results)

results_df = results_df[
    [
        "subgroup",
        "category",
        "sample_size",
        "positive_cases",
        "negative_cases",
        "actual_positive_rate",
        "predicted_positive_rate",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "specificity",
        "false_positive_rate",
        "true_positives",
        "false_positives",
        "true_negatives",
        "false_negatives"
    ]
]


# ============================================================
# 11. SAVE SUBGROUP METRICS
# ============================================================

csv_path = (
    OUT_DIR / "subgroup_fairness_metrics.csv"
)

results_df.to_csv(
    csv_path,
    index=False
)

overall_path = (
    OUT_DIR / "subgroup_overall_metrics.csv"
)

overall_df.to_csv(
    overall_path,
    index=False
)


# ============================================================
# 12. CALCULATE DESCRIPTIVE GROUP GAPS
# ============================================================

gap_results = []

for column in SUBGROUP_COLUMNS:

    subset = results_df[
        results_df["subgroup"] == column
    ]

    if len(subset) < 2:
        continue

    for metric in [
        "recall",
        "precision",
        "f1_score",
        "specificity",
        "false_positive_rate",
        "predicted_positive_rate"
    ]:

        valid = subset[metric].dropna()

        if len(valid) < 2:
            continue

        gap_results.append({

            "subgroup": column,

            "metric": metric,

            "minimum": float(valid.min()),

            "maximum": float(valid.max()),

            "absolute_gap": float(
                valid.max() - valid.min()
            )

        })


gaps_df = pd.DataFrame(gap_results)

gaps_path = (
    OUT_DIR / "subgroup_metric_gaps.csv"
)

gaps_df.to_csv(
    gaps_path,
    index=False
)


# ============================================================
# 13. GENERATE SUBGROUP PERFORMANCE PLOTS
# ============================================================

print("\nGenerating subgroup performance plots...")

for column in SUBGROUP_COLUMNS:

    subset = results_df[
        results_df["subgroup"] == column
    ].copy()

    if subset.empty:
        continue

    plot_data = subset.set_index(
        "category"
    )[
        [
            "precision",
            "recall",
            "f1_score"
        ]
    ]

    ax = plot_data.plot(
        kind="bar",
        figsize=(11, 6)
    )

    ax.set_title(
        f"CareSignal Performance by {column.capitalize()}"
    )

    ax.set_ylabel("Metric score")

    ax.set_xlabel(column.capitalize())

    ax.set_ylim(0, 1)

    ax.legend(
        title="Evaluation metric"
    )

    plt.xticks(rotation=35, ha="right")

    plt.tight_layout()

    plot_path = (
        OUT_DIR / f"subgroup_{column}_performance.png"
    )

    plt.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 14. SAVE JSON SUMMARY
# ============================================================

summary = {

    "experiment": (
        "Subgroup Performance and Fairness Analysis"
    ),

    "model": artifact.get(
        "model_name",
        "Saved model"
    ),

    "threshold": threshold,

    "test_encounters": int(len(X_test)),

    "test_patients": int(len(test_patients)),

    "patient_overlap": 0,

    "minimum_subgroup_sample_size": MIN_GROUP_SIZE,

    "subgroups_analysed": SUBGROUP_COLUMNS,

    "overall_metrics": overall_metrics,

    "subgroup_results": results,

    "descriptive_metric_gaps": gap_results,

    "methodology_note": (
        "The original patient-level test split and saved "
        "model were used without retraining. The saved "
        "classification threshold was applied consistently "
        "to all subgroups. Groups below the minimum sample "
        "size were excluded. ROC-AUC and PR-AUC are not "
        "calculated for groups containing only one target "
        "class. Metric gaps are descriptive comparisons, "
        "not proof of discrimination or fairness. Results "
        "require uncertainty analysis and clinical context."
    )

}

json_path = (
    OUT_DIR / "subgroup_fairness_results.json"
)

json_path.write_text(
    json.dumps(
        summary,
        indent=2,
        allow_nan=False
    )
)


# ============================================================
# 15. PRINT FINAL RESULTS
# ============================================================

print("\n" + "=" * 75)

print("OVERALL TEST PERFORMANCE")

print("=" * 75)

print(
    pd.DataFrame([overall_metrics]).to_string(
        index=False
    )
)

print("\n" + "=" * 75)

print("SUBGROUP PERFORMANCE RESULTS")

print("=" * 75)

print(
    results_df.round(4).to_string(
        index=False
    )
)

print("\n" + "=" * 75)

print("DESCRIPTIVE SUBGROUP METRIC GAPS")

print("=" * 75)

if not gaps_df.empty:

    print(
        gaps_df.round(4).to_string(
            index=False
        )
    )

else:

    print("Insufficient subgroup categories for gap analysis.")


print("\nOUTPUT FILES")

for path in [
    csv_path,
    overall_path,
    gaps_path,
    json_path
]:

    print(path.relative_to(ROOT))


for column in SUBGROUP_COLUMNS:

    path = (
        OUT_DIR / f"subgroup_{column}_performance.png"
    )

    if path.exists():
        print(path.relative_to(ROOT))


print("\nEXPERIMENT 6 COMPLETED SUCCESSFULLY.")

print(
    "\nInterpret subgroup differences cautiously. "
    "This analysis alone does not establish fairness, "
    "bias, or clinical suitability."
)

print(
    "Original saved model and dashboard remain unchanged."
)
