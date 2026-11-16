
"""
CareSignal - Experiment 4
Validation-only calibrated threshold optimization.

Methodology:
1. Reproduce original patient-level train/test split.
2. Reproduce original patient-level subtrain/validation split.
3. Divide validation patients into two disjoint groups:
   a. Calibration fitting
   b. Threshold optimization
4. Fit the base model on subtraining patients only.
5. Fit sigmoid and isotonic calibrators on calibration-fit patients.
6. Optimize thresholds on separate threshold-validation patients.
7. Evaluate once on untouched patient-level test patients.

The saved model and Streamlit dashboard are NOT modified.

Run from project root:
python experiments/calibrated_threshold_optimization.py
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    precision_recall_curve,
)

from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

from scipy.special import logit


# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "raw" / "diabetic_data_original.csv"

MODEL_PATH = ROOT / "models" / "healthcare_readmission_model.joblib"

OUT_DIR = ROOT / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MINIMUM_RECALL = 0.60

print("\n" + "=" * 65)
print("CARESIGNAL - EXPERIMENT 4")
print("VALIDATION-ONLY CALIBRATED THRESHOLD OPTIMIZATION")
print("=" * 65)


# ============================================================
# 2. LOAD DATA AND ARTIFACT
# ============================================================

print("\nLoading dataset and saved model...")

df = pd.read_csv(DATA_PATH, low_memory=False)
artifact = joblib.load(MODEL_PATH)

print("Dataset:", df.shape)
print("Saved model loaded.")


# ============================================================
# 3. REPRODUCE ORIGINAL PREPROCESSING
# ============================================================

print("\nReproducing original preprocessing...")

df.columns = df.columns.astype(str).str.strip()

# Verify duplicate identifier columns.
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


# Target definition.
y = (
    df["readmitted"]
    .astype(str)
    .str.strip()
    .eq("<30")
    .astype("int8")
)

groups = df["patient_nbr"].copy()

# Exclude identifiers and outcome columns.
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

# Replace missing-value marker.
X = X.replace("?", np.nan)

# Match original categorical treatment.
coded_categorical = [
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
]

for col in coded_categorical:

    if col in X.columns:
        X[col] = X[col].astype("string")

# Drop only fully missing features.
all_missing = X.columns[
    X.isna().all()
].tolist()

if all_missing:
    X = X.drop(columns=all_missing)

# Verify feature schema and order.
features = artifact["feature_columns"]

if list(X.columns) != list(features):

    raise ValueError(
        "Feature schema mismatch.\n"
        f"Expected: {len(features)}\n"
        f"Actual: {len(X.columns)}"
    )

X = X[features].copy()

print("Preprocessing verified.")
print("Rows:", len(X))
print("Features:", len(features))
print("Patients:", groups.nunique())


# ============================================================
# 4. OUTER PATIENT-LEVEL TRAIN/TEST SPLIT
# ============================================================

print("\nCreating outer patient-level split...")

outer = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

outer_train_idx, test_idx = next(
    outer.split(X, y, groups=groups)
)

X_outer_train = X.iloc[outer_train_idx]
y_outer_train = y.iloc[outer_train_idx]
groups_outer_train = groups.iloc[outer_train_idx]


# ============================================================
# 5. INNER PATIENT-LEVEL SUBTRAIN/VALIDATION SPLIT
# ============================================================

print("Creating inner patient-level split...")

inner = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=21
)

subtrain_rel_idx, val_rel_idx = next(
    inner.split(
        X_outer_train,
        y_outer_train,
        groups=groups_outer_train
    )
)

subtrain_idx = outer_train_idx[subtrain_rel_idx]
val_idx = outer_train_idx[val_rel_idx]

X_subtrain = X.iloc[subtrain_idx].copy()
y_subtrain = y.iloc[subtrain_idx].copy()

X_val = X.iloc[val_idx].copy()
y_val = y.iloc[val_idx].copy()

X_test = X.iloc[test_idx].copy()
y_test = y.iloc[test_idx].copy()

groups_subtrain = groups.iloc[subtrain_idx]
groups_val = groups.iloc[val_idx]
groups_test = groups.iloc[test_idx]


# ============================================================
# 6. SPLIT VALIDATION INTO TWO DISJOINT PATIENT GROUPS
# ============================================================

print("\nSplitting validation patients...")

validation_split = GroupShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=77
)

cal_fit_rel_idx, threshold_rel_idx = next(
    validation_split.split(
        X_val,
        y_val,
        groups=groups_val
    )
)

X_cal_fit = X_val.iloc[cal_fit_rel_idx].copy()
y_cal_fit = y_val.iloc[cal_fit_rel_idx].copy()

X_threshold_val = X_val.iloc[threshold_rel_idx].copy()
y_threshold_val = y_val.iloc[threshold_rel_idx].copy()

groups_cal_fit = groups_val.iloc[cal_fit_rel_idx]
groups_threshold = groups_val.iloc[threshold_rel_idx]


# ============================================================
# 7. VERIFY NO PATIENT OVERLAP
# ============================================================

patient_sets = {
    "subtrain": set(groups_subtrain.unique()),
    "calibration_fit": set(groups_cal_fit.unique()),
    "threshold_validation": set(groups_threshold.unique()),
    "test": set(groups_test.unique()),
}

names = list(patient_sets.keys())

for i in range(len(names)):

    for j in range(i + 1, len(names)):

        overlap = (
            patient_sets[names[i]]
            & patient_sets[names[j]]
        )

        if overlap:

            raise RuntimeError(
                f"Patient overlap: {names[i]} and {names[j]}"
            )


print("\nPARTITION VERIFICATION")
print("----------------------")

print("Subtrain encounters:", len(X_subtrain))
print("Calibration-fit encounters:", len(X_cal_fit))
print("Threshold-validation encounters:", len(X_threshold_val))
print("Test encounters:", len(X_test))

print("Subtrain patients:", len(patient_sets["subtrain"]))
print("Calibration-fit patients:", len(patient_sets["calibration_fit"]))
print("Threshold-validation patients:", len(patient_sets["threshold_validation"]))
print("Test patients:", len(patient_sets["test"]))

print("Patient overlap: 0")


# Check both validation groups contain both classes.
for name, target in [
    ("Calibration-fit", y_cal_fit),
    ("Threshold-validation", y_threshold_val),
    ("Test", y_test),
]:

    if target.nunique() < 2:
        raise RuntimeError(
            f"{name} partition does not contain both classes."
        )


# ============================================================
# 8. FIT FRESH BASE MODEL ON SUBTRAIN ONLY
# ============================================================

print("\nFitting fresh copy of saved pipeline...")

base_model = clone(artifact["model"])

base_model.fit(
    X_subtrain,
    y_subtrain
)

print("Base model fitted.")


# ============================================================
# 9. GENERATE PROBABILITIES
# ============================================================

print("\nGenerating probabilities...")

p_cal_fit_raw = base_model.predict_proba(
    X_cal_fit
)[:, 1]

p_threshold_raw = base_model.predict_proba(
    X_threshold_val
)[:, 1]

p_test_raw = base_model.predict_proba(
    X_test
)[:, 1]


EPS = 1e-6

p_cal_fit_raw = np.clip(
    p_cal_fit_raw, EPS, 1 - EPS
)

p_threshold_raw = np.clip(
    p_threshold_raw, EPS, 1 - EPS
)

p_test_raw = np.clip(
    p_test_raw, EPS, 1 - EPS
)


# ============================================================
# 10. FIT SIGMOID CALIBRATOR
# ============================================================

print("\nFitting sigmoid calibrator...")

sigmoid_calibrator = LogisticRegression(
    C=1e6,
    solver="lbfgs",
    max_iter=2000
)

sigmoid_calibrator.fit(
    logit(p_cal_fit_raw).reshape(-1, 1),
    y_cal_fit
)


# ============================================================
# 11. FIT ISOTONIC CALIBRATOR
# ============================================================

print("Fitting isotonic calibrator...")

isotonic_calibrator = IsotonicRegression(
    out_of_bounds="clip"
)

isotonic_calibrator.fit(
    p_cal_fit_raw,
    y_cal_fit
)


# ============================================================
# 12. CALIBRATE THRESHOLD-VALIDATION PROBABILITIES
# ============================================================

p_threshold_sigmoid = sigmoid_calibrator.predict_proba(
    logit(p_threshold_raw).reshape(-1, 1)
)[:, 1]

p_threshold_isotonic = isotonic_calibrator.predict(
    p_threshold_raw
)


# ============================================================
# 13. CALIBRATE TEST PROBABILITIES
# ============================================================

print("\nCalibrating untouched test probabilities...")

p_test_sigmoid = sigmoid_calibrator.predict_proba(
    logit(p_test_raw).reshape(-1, 1)
)[:, 1]

p_test_isotonic = isotonic_calibrator.predict(
    p_test_raw
)


# ============================================================
# 14. VALIDATION-ONLY THRESHOLD OPTIMIZATION
# ============================================================

def optimize_threshold(y_true, probabilities, min_recall=0.60):

    probabilities = np.asarray(probabilities)

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probabilities
    )

    if len(thresholds) == 0:
        raise RuntimeError("No thresholds available.")

    precision_values = precision[:-1]
    recall_values = recall[:-1]

    f1_values = (
        2 * precision_values * recall_values
        / (
            precision_values + recall_values + 1e-12
        )
    )

    valid_indices = np.where(
        recall_values >= min_recall
    )[0]

    if len(valid_indices) == 0:

        raise RuntimeError(
            "No validation threshold satisfies "
            f"minimum recall of {min_recall:.2f}."
        )

    best_index = valid_indices[
        np.argmax(f1_values[valid_indices])
    ]

    return {
        "threshold": float(thresholds[best_index]),
        "validation_precision": float(
            precision_values[best_index]
        ),
        "validation_recall": float(
            recall_values[best_index]
        ),
        "validation_f1": float(
            f1_values[best_index]
        ),
    }


print("\nOptimizing thresholds on separate validation patients...")

threshold_results = {

    "Uncalibrated": optimize_threshold(
        y_threshold_val,
        p_threshold_raw,
        MINIMUM_RECALL
    ),

    "Sigmoid": optimize_threshold(
        y_threshold_val,
        p_threshold_sigmoid,
        MINIMUM_RECALL
    ),

    "Isotonic": optimize_threshold(
        y_threshold_val,
        p_threshold_isotonic,
        MINIMUM_RECALL
    ),

}


# ============================================================
# 15. FINAL TEST EVALUATION
# ============================================================

def evaluate_test(
    name,
    probabilities,
    threshold
):

    probabilities = np.clip(
        np.asarray(probabilities),
        1e-7,
        1 - 1e-7
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    ).ravel()

    return {

        "method": name,

        "threshold": float(threshold),

        "brier_score": float(
            brier_score_loss(
                y_test,
                probabilities
            )
        ),

        "log_loss": float(
            log_loss(
                y_test,
                probabilities,
                labels=[0, 1]
            )
        ),

        "accuracy": float(
            accuracy_score(
                y_test,
                predictions
            )
        ),

        "precision": float(
            precision_score(
                y_test,
                predictions,
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                y_test,
                predictions,
                zero_division=0
            )
        ),

        "f1": float(
            f1_score(
                y_test,
                predictions,
                zero_division=0
            )
        ),

        "roc_auc": float(
            roc_auc_score(
                y_test,
                probabilities
            )
        ),

        "pr_auc": float(
            average_precision_score(
                y_test,
                probabilities
            )
        ),

        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),

    }


print("\nEvaluating on untouched test patients...")

test_probabilities = {

    "Uncalibrated": p_test_raw,

    "Sigmoid": p_test_sigmoid,

    "Isotonic": p_test_isotonic,

}

results = []

for name, probabilities in test_probabilities.items():

    threshold = threshold_results[name]["threshold"]

    result = evaluate_test(
        name,
        probabilities,
        threshold
    )

    result["validation_precision"] = (
        threshold_results[name]["validation_precision"]
    )

    result["validation_recall"] = (
        threshold_results[name]["validation_recall"]
    )

    result["validation_f1"] = (
        threshold_results[name]["validation_f1"]
    )

    results.append(result)


results_df = pd.DataFrame(results)


# ============================================================
# 16. SAVE RESULTS
# ============================================================

csv_path = (
    OUT_DIR / "calibrated_threshold_optimization.csv"
)

results_df.to_csv(
    csv_path,
    index=False
)


# ============================================================
# 17. CONFUSION MATRIX SUMMARY
# ============================================================

confusion_summary = {}

for row in results:

    confusion_summary[row["method"]] = {

        "threshold": row["threshold"],

        "TN": row["true_negatives"],

        "FP": row["false_positives"],

        "FN": row["false_negatives"],

        "TP": row["true_positives"],

    }


# ============================================================
# 18. SAVE JSON SUMMARY
# ============================================================

summary = {

    "experiment": "Validation-only calibrated threshold optimization",

    "dataset_rows": int(len(df)),

    "feature_count": int(len(features)),

    "unique_patients": int(groups.nunique()),

    "minimum_validation_recall_constraint": MINIMUM_RECALL,

    "subtrain_encounters": int(len(X_subtrain)),

    "calibration_fit_encounters": int(len(X_cal_fit)),

    "threshold_validation_encounters": int(len(X_threshold_val)),

    "test_encounters": int(len(X_test)),

    "subtrain_patients": int(len(patient_sets["subtrain"])),

    "calibration_fit_patients": int(len(patient_sets["calibration_fit"])),

    "threshold_validation_patients": int(len(patient_sets["threshold_validation"])),

    "test_patients": int(len(patient_sets["test"])),

    "patient_overlap": 0,

    "validation_thresholds": threshold_results,

    "test_results": results,

    "confusion_matrices": confusion_summary,

    "methodology_note": (
        "The original preprocessing and patient-level split "
        "strategy were reproduced. The saved pipeline was cloned "
        "and fitted only on sub-training patients. The validation "
        "partition was divided into disjoint patient groups for "
        "calibration fitting and threshold optimization. "
        "Sigmoid and isotonic calibrators were fitted only on "
        "calibration-fit patients. Thresholds were selected on "
        "separate threshold-validation patients by maximizing "
        "F1 subject to recall >= 0.60. The test partition was "
        "used only for final evaluation. The original saved model "
        "and dashboard were not modified."
    ),

}

json_path = (
    OUT_DIR / "calibrated_threshold_optimization_results.json"
)

json_path.write_text(
    json.dumps(summary, indent=2)
)


# ============================================================
# 19. VISUALIZE TEST F1 COMPARISON
# ============================================================

plt.figure(figsize=(8, 6))

plt.bar(
    results_df["method"],
    results_df["f1"]
)

plt.ylabel("Test F1-score")

plt.xlabel("Model method")

plt.title(
    "Calibrated Threshold Optimization\n"
    "Patient-Level Test F1 Comparison"
)

plt.grid(
    axis="y",
    alpha=0.25
)

plt.tight_layout()

plot_path = (
    OUT_DIR / "calibrated_threshold_optimization.png"
)

plt.savefig(
    plot_path,
    dpi=300
)

plt.close()


# ============================================================
# 20. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 75)

print("VALIDATION-OPTIMIZED THRESHOLDS")

print("=" * 75)

for name, values in threshold_results.items():

    print(
        f"{name}: "
        f"Threshold={values['threshold']:.6f}, "
        f"Validation Precision={values['validation_precision']:.4f}, "
        f"Validation Recall={values['validation_recall']:.4f}, "
        f"Validation F1={values['validation_f1']:.4f}"
    )


print("\n" + "=" * 75)

print("FINAL PATIENT-LEVEL TEST RESULTS")

print("=" * 75)

print(
    results_df.to_string(index=False)
)

print("\nSaved files:")

print(csv_path.relative_to(ROOT))
print(plot_path.relative_to(ROOT))
print(json_path.relative_to(ROOT))

print("\nEXPERIMENT 4 COMPLETED SUCCESSFULLY.")

print(
    "\nImportant: Thresholds were selected on validation "
    "patients, not test patients."
)

print(
    "Do not choose a deployment model based on test results."
)

print(
    "Original saved model and Streamlit dashboard remain unchanged."
)
