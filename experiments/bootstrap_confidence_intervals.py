
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score


# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "raw" / "diabetic_data_original.csv"
MODEL_PATH = ROOT / "models" / "healthcare_readmission_model.joblib"
REPORT_DIR = ROOT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

N_BOOTSTRAPS = 1000
RANDOM_STATE = 42


# --------------------------------------------------
# 2. Load model and dataset
# --------------------------------------------------

artifact = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH, low_memory=False)

df.columns = df.columns.str.strip()

# Reproduce original training duplicate-ID handling
for canonical, duplicate in [
    ("patient_nbr", "patient_nbr.1"),
    ("encounter_id", "encounter_id.1")
]:
    if duplicate in df.columns:
        if not df[canonical].astype(str).equals(
            df[duplicate].astype(str)
        ):
            raise ValueError(f"{canonical} duplicate mismatch")

        df = df.drop(columns=[duplicate])


# --------------------------------------------------
# 3. EXACT original training preprocessing
# --------------------------------------------------

y = (
    df["readmitted"]
    .astype(str)
    .str.strip()
    .eq("<30")
    .astype("int8")
)

patient_groups = df["patient_nbr"].copy()

if patient_groups.isna().any():
    raise ValueError("Missing patient IDs detected.")

exclude = {
    "patient_nbr",
    "encounter_id",
    "readmitted",
    "readmitted_30"
}

X = df[
    [c for c in df.columns if c not in exclude]
].copy()

X = X.replace("?", np.nan)

coded_categorical_cols = [
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id"
]

for col in coded_categorical_cols:
    if col in X.columns:
        X[col] = X[col].astype("string")

empty_columns = [
    c for c in X.columns if X[c].isna().all()
]

if empty_columns:
    X = X.drop(columns=empty_columns)

feature_columns = list(X.columns)

if feature_columns != artifact["feature_columns"]:
    raise ValueError(
        "Feature columns do not match saved model artifact."
    )

print("Dataset rows:", len(df))
print("Feature count:", X.shape[1])
print("Unique patients:", patient_groups.nunique())


# --------------------------------------------------
# 4. Reproduce original patient-level test split
# --------------------------------------------------

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=patient_groups)
)

X_test = X.iloc[test_idx].copy()
y_test = y.iloc[test_idx].to_numpy()
test_groups = patient_groups.iloc[test_idx].to_numpy()

if len(np.unique(test_groups)) != len(test_groups):
    # Multiple encounters per patient are expected.
    pass

model = artifact["model"]

test_prob = model.predict_proba(X_test)[:, 1]

# Verify original point estimates before bootstrapping
point_roc = roc_auc_score(y_test, test_prob)
point_pr = average_precision_score(y_test, test_prob)

print("\nREPRODUCIBILITY CHECK")
print("Test encounters:", len(test_idx))
print("Test patients:", len(np.unique(test_groups)))
print("ROC-AUC:", point_roc)
print("PR-AUC:", point_pr)

expected = artifact["evaluation"]

if not np.isclose(
    point_roc, expected["ROC_AUC"], atol=1e-8
):
    raise ValueError("ROC-AUC does not match saved evaluation.")

if not np.isclose(
    point_pr, expected["PR_AUC"], atol=1e-8
):
    raise ValueError("PR-AUC does not match saved evaluation.")

print("Original evaluation reproduced successfully.")


# --------------------------------------------------
# 5. Patient-cluster bootstrap
# --------------------------------------------------

rng = np.random.default_rng(RANDOM_STATE)

unique_patients = pd.unique(test_groups)

patient_positions = {
    patient: np.flatnonzero(test_groups == patient)
    for patient in unique_patients
}

roc_scores = []
pr_scores = []

for i in range(N_BOOTSTRAPS):

    sampled_patients = rng.choice(
        unique_patients,
        size=len(unique_patients),
        replace=True
    )

    sampled_positions = np.concatenate([
        patient_positions[patient]
        for patient in sampled_patients
    ])

    y_sample = y_test[sampled_positions]
    p_sample = test_prob[sampled_positions]

    # Skip samples containing only one target class
    if len(np.unique(y_sample)) < 2:
        continue

    roc_scores.append(
        roc_auc_score(y_sample, p_sample)
    )

    pr_scores.append(
        average_precision_score(y_sample, p_sample)
    )


# --------------------------------------------------
# 6. Confidence intervals
# --------------------------------------------------

def confidence_interval(values):
    return np.percentile(values, [2.5, 97.5])


roc_ci = confidence_interval(roc_scores)
pr_ci = confidence_interval(pr_scores)

results = {
    "method": "Patient-cluster bootstrap",
    "bootstrap_replicates_requested": N_BOOTSTRAPS,
    "bootstrap_replicates_valid": len(roc_scores),
    "test_encounters": int(len(test_idx)),
    "test_patients": int(len(unique_patients)),
    "roc_auc": {
        "point_estimate": float(point_roc),
        "ci_95_lower": float(roc_ci[0]),
        "ci_95_upper": float(roc_ci[1])
    },
    "pr_auc": {
        "point_estimate": float(point_pr),
        "ci_95_lower": float(pr_ci[0]),
        "ci_95_upper": float(pr_ci[1])
    }
}


# --------------------------------------------------
# 7. Save reports
# --------------------------------------------------

pd.DataFrame({
    "ROC_AUC": roc_scores,
    "PR_AUC": pr_scores
}).to_csv(
    REPORT_DIR / "bootstrap_confidence_intervals.csv",
    index=False
)

with open(
    REPORT_DIR / "bootstrap_confidence_intervals.json",
    "w"
) as f:
    json.dump(results, f, indent=4)

print("\nFINAL BOOTSTRAP RESULTS")
print(json.dumps(results, indent=4))
print("\nReports saved successfully.")
