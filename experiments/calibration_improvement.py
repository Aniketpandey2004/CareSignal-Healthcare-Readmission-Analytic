
"""
CareSignal: Validation-only probability calibration experiment.

Run from the project root:
    python experiments/calibration_improvement.py

The saved model is NOT modified.

A fresh copy of its pipeline is fitted on the grouped sub-training
partition only. Sigmoid and isotonic calibrators are fitted on the
grouped validation partition and evaluated once on the untouched
patient-level test partition.
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
)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

from scipy.special import logit


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT / "data" / "raw" / "diabetic_data_original.csv"
)

MODEL_PATH = (
    ROOT / "models" / "healthcare_readmission_model.joblib"
)

OUT_DIR = ROOT / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATA AND SAVED MODEL
# ============================================================

print("\nLoading dataset and saved model...")

df = pd.read_csv(DATA_PATH, low_memory=False)
artifact = joblib.load(MODEL_PATH)

print("Dataset loaded:", df.shape)
print("Saved model loaded successfully.")


# ============================================================
# 3. REPRODUCE ORIGINAL TRAINING PREPROCESSING
# ============================================================

print("\nReproducing original preprocessing...")

# Clean column names.
df.columns = df.columns.astype(str).str.strip()

# Verify duplicate identifier columns before dropping.
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


# Target: 30-day readmission.
y = (
    df["readmitted"]
    .astype(str)
    .str.strip()
    .eq("<30")
    .astype("int8")
)

# Patient grouping.
groups = df["patient_nbr"].copy()


# Remove identifiers and outcome columns from predictors.
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


# Replace dataset missing-value marker.
X = X.replace("?", np.nan)


# Treat coded administrative IDs as categorical variables.
coded_categorical = [
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
]

for col in coded_categorical:

    if col in X.columns:
        X[col] = X[col].astype("string")


# Remove only completely missing columns.
all_missing_columns = X.columns[
    X.isna().all()
].tolist()

if all_missing_columns:
    X = X.drop(columns=all_missing_columns)


# Verify feature names and order against saved artifact.
features = artifact["feature_columns"]

if list(X.columns) != list(features):

    raise ValueError(
        "\nFEATURE SCHEMA MISMATCH!\n"
        f"Expected features: {len(features)}\n"
        f"Actual features: {len(X.columns)}\n"
        f"Expected columns: {list(features)}\n"
        f"Actual columns: {list(X.columns)}"
    )

X = X[features].copy()

print("\nPREPROCESSING VERIFICATION")
print("--------------------------")
print("Rows:", len(X))
print("Features:", len(features))
print("Unique patients:", groups.nunique())
print("Positive rate:", round(float(y.mean()), 5))

print("Feature schema matches saved model.")


# ============================================================
# 4. OUTER PATIENT-LEVEL TRAIN/TEST SPLIT
# ============================================================

print("\nCreating patient-level train/test split...")

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
# 5. INNER PATIENT-LEVEL TRAIN/VALIDATION SPLIT
# ============================================================

print("Creating grouped train/validation split...")

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


# Create final partitions.
X_subtrain = X.iloc[subtrain_idx].copy()
y_subtrain = y.iloc[subtrain_idx].copy()

X_val = X.iloc[val_idx].copy()
y_val = y.iloc[val_idx].copy()

X_test = X.iloc[test_idx].copy()
y_test = y.iloc[test_idx].copy()


# ============================================================
# 6. VERIFY PATIENT-LEVEL SEPARATION
# ============================================================

train_patients = set(
    groups.iloc[subtrain_idx].unique()
)

val_patients = set(
    groups.iloc[val_idx].unique()
)

test_patients = set(
    groups.iloc[test_idx].unique()
)

train_val_overlap = train_patients & val_patients
train_test_overlap = train_patients & test_patients
val_test_overlap = val_patients & test_patients

if (
    train_val_overlap
    or train_test_overlap
    or val_test_overlap
):

    raise RuntimeError(
        "Patient overlap detected between partitions!"
    )


print("\nPARTITION VERIFICATION")
print("----------------------")

print("Sub-training encounters:", len(X_subtrain))
print("Validation encounters:", len(X_val))
print("Test encounters:", len(X_test))

print("Sub-training patients:", len(train_patients))
print("Validation patients:", len(val_patients))
print("Test patients:", len(test_patients))

print("Patient overlap: 0")


# ============================================================
# 7. FIT FRESH COPY OF SAVED PIPELINE
# ============================================================

print("\nFitting fresh model copy on sub-training data...")

base_model = clone(artifact["model"])

base_model.fit(
    X_subtrain,
    y_subtrain
)

print("Fresh model fitted successfully.")


# ============================================================
# 8. GENERATE VALIDATION AND TEST PROBABILITIES
# ============================================================

print("\nGenerating probability predictions...")

p_val_raw = base_model.predict_proba(
    X_val
)[:, 1]

p_test_raw = base_model.predict_proba(
    X_test
)[:, 1]

p_val_raw = np.clip(
    p_val_raw,
    1e-6,
    1 - 1e-6
)

p_test_raw = np.clip(
    p_test_raw,
    1e-6,
    1 - 1e-6
)


# ============================================================
# 9. SIGMOID / PLATT CALIBRATION
# ============================================================

print("\nFitting sigmoid calibration...")

sigmoid_calibrator = LogisticRegression(
    C=1e6,
    solver="lbfgs",
    max_iter=2000
)

sigmoid_calibrator.fit(
    logit(p_val_raw).reshape(-1, 1),
    y_val
)

p_test_sigmoid = sigmoid_calibrator.predict_proba(
    logit(p_test_raw).reshape(-1, 1)
)[:, 1]

print("Sigmoid calibration completed.")


# ============================================================
# 10. ISOTONIC CALIBRATION
# ============================================================

print("\nFitting isotonic calibration...")

isotonic_calibrator = IsotonicRegression(
    out_of_bounds="clip"
)

isotonic_calibrator.fit(
    p_val_raw,
    y_val
)

p_test_isotonic = isotonic_calibrator.predict(
    p_test_raw
)

print("Isotonic calibration completed.")


# ============================================================
# 11. EVALUATION FUNCTION
# ============================================================

def evaluate(name, probabilities):

    probabilities = np.clip(
        np.asarray(probabilities),
        1e-7,
        1 - 1e-7
    )

    # Original saved threshold used only as a reference.
    threshold = float(artifact["threshold"])

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {

        "method": name,

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

        "precision_at_saved_threshold": float(
            precision_score(
                y_test,
                predictions,
                zero_division=0
            )
        ),

        "recall_at_saved_threshold": float(
            recall_score(
                y_test,
                predictions,
                zero_division=0
            )
        ),

        "f1_at_saved_threshold": float(
            f1_score(
                y_test,
                predictions,
                zero_division=0
            )
        ),
    }


# ============================================================
# 12. COMPARE CALIBRATION METHODS
# ============================================================

print("\nEvaluating calibration methods...")

methods = {

    "Uncalibrated": p_test_raw,

    "Sigmoid": p_test_sigmoid,

    "Isotonic": p_test_isotonic,

}

results = []

for name, probabilities in methods.items():

    results.append(
        evaluate(name, probabilities)
    )


results_df = pd.DataFrame(results)

csv_path = (
    OUT_DIR / "calibration_method_comparison.csv"
)

results_df.to_csv(
    csv_path,
    index=False
)


# ============================================================
# 13. RELIABILITY CURVE
# ============================================================

print("\nGenerating reliability curve...")

plt.figure(figsize=(8, 7))

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect calibration"
)

for name, probabilities in methods.items():

    observed, predicted = calibration_curve(
        y_test,
        probabilities,
        n_bins=10,
        strategy="uniform"
    )

    plt.plot(
        predicted,
        observed,
        marker="o",
        label=name
    )


plt.xlabel("Mean predicted probability")
plt.ylabel("Observed readmission rate")

plt.title(
    "Probability Calibration Comparison\n"
    "Patient-Level Test Set"
)

plt.legend()
plt.grid(alpha=0.25)

plt.tight_layout()

plot_path = (
    OUT_DIR / "calibration_comparison.png"
)

plt.savefig(
    plot_path,
    dpi=300
)

plt.close()


# ============================================================
# 14. SAVE EXPERIMENT SUMMARY
# ============================================================

summary = {

    "experiment": "Validation-only probability calibration",

    "dataset_rows": int(len(df)),

    "feature_count": int(len(features)),

    "unique_patients": int(groups.nunique()),

    "subtrain_encounters": int(len(X_subtrain)),

    "validation_encounters": int(len(X_val)),

    "test_encounters": int(len(X_test)),

    "subtrain_patients": int(len(train_patients)),

    "validation_patients": int(len(val_patients)),

    "test_patients": int(len(test_patients)),

    "patient_overlap": 0,

    "test_positive_rate": float(y_test.mean()),

    "saved_threshold_used_for_reference": float(
        artifact["threshold"]
    ),

    "results": results,

    "methodology_note": (

        "The original preprocessing and patient-level split "
        "strategy were reproduced. A fresh copy of the saved "
        "pipeline was fitted on grouped sub-training patients "
        "only. Sigmoid and isotonic calibrators were fitted "
        "using grouped validation predictions only. The "
        "patient-level test partition was untouched until "
        "final evaluation. The saved model artifact and "
        "Streamlit dashboard were not modified. The saved "
        "threshold is used only as a reference because "
        "calibration can change probability scale."
    ),

}


json_path = (
    OUT_DIR / "calibration_improvement_results.json"
)

json_path.write_text(
    json.dumps(
        summary,
        indent=2
    )
)


# ============================================================
# 15. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 65)

print(
    "CALIBRATION METHOD COMPARISON "
    "(PATIENT-LEVEL TEST SET)"
)

print("=" * 65)

print(
    results_df.to_string(
        index=False
    )
)

print("\nSaved output files:")

print(
    "1.",
    csv_path.relative_to(ROOT)
)

print(
    "2.",
    plot_path.relative_to(ROOT)
)

print(
    "3.",
    json_path.relative_to(ROOT)
)

print("\nEXPERIMENT COMPLETED SUCCESSFULLY.")

print(
    "\nNote: Lower Brier Score and Log Loss indicate "
    "better probability calibration."
)

print(
    "Do not select a calibration method based on "
    "test results for deployment."
)

print(
    "\nOriginal saved model remains unchanged."
)
