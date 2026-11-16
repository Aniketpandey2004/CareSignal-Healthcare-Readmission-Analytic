"""
CareSignal calibration evaluation
Uses the saved patient-level model and recreates the untouched 20% patient-level
test split. It evaluates calibration only; it does NOT fit or modify the model.
Run from the project root:
    python experiments/calibration_analysis.py
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.calibration import calibration_curve

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "diabetic_data_original.csv"
MODEL_PATH = ROOT / "models" / "healthcare_readmission_model.joblib"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

print("Loading dataset and saved model...")
df = pd.read_csv(DATA_PATH, low_memory=False)
artifact = joblib.load(MODEL_PATH)

required = {"patient_nbr", "readmitted"}
missing = required.difference(df.columns)
if missing:
    raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

feature_columns = artifact["feature_columns"]
missing_features = [c for c in feature_columns if c not in df.columns]
if missing_features:
    raise ValueError(f"Dataset is missing model features: {missing_features}")

# Match the project's target definition: positive means readmitted in <30 days.
y = (df["readmitted"].astype(str) == "<30").astype(int)
X = df[feature_columns].copy()
groups = df["patient_nbr"]

# Recreate the outer patient-level 80/20 holdout used by train_model.py.
splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=groups))

X_test = X.iloc[test_idx]
y_test = y.iloc[test_idx]

# Confirm there is no patient overlap between training and test partitions.
train_patients = set(groups.iloc[train_idx].unique())
test_patients = set(groups.iloc[test_idx].unique())
overlap = train_patients.intersection(test_patients)
if overlap:
    raise RuntimeError(f"Patient leakage detected: {len(overlap)} shared patients")

model = artifact["model"]
probabilities = model.predict_proba(X_test)[:, 1]
probabilities = np.clip(probabilities, 1e-7, 1 - 1e-7)

brier = brier_score_loss(y_test, probabilities)
logloss = log_loss(y_test, probabilities, labels=[0, 1])

# Ten equal-width probability bins. Empty bins are omitted by sklearn.
fraction_positive, mean_predicted = calibration_curve(
    y_test,
    probabilities,
    n_bins=10,
    strategy="uniform"
)

calibration_table = pd.DataFrame({
    "mean_predicted_probability": mean_predicted,
    "observed_positive_rate": fraction_positive,
    "absolute_gap": np.abs(mean_predicted - fraction_positive),
})
csv_path = REPORTS_DIR / "calibration_bins.csv"
calibration_table.to_csv(csv_path, index=False)

plt.figure(figsize=(7, 6))
plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
plt.plot(mean_predicted, fraction_positive, marker="o", label="CareSignal model")
plt.xlabel("Mean predicted probability")
plt.ylabel("Observed readmission rate")
plt.title("Calibration Curve – Patient-Level Test Set")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plot_path = REPORTS_DIR / "calibration_curve.png"
plt.savefig(plot_path, dpi=300)
plt.close()

results = {
    "test_encounters": int(len(test_idx)),
    "test_patients": int(len(test_patients)),
    "shared_patients_train_test": int(len(overlap)),
    "test_positive_rate": float(y_test.mean()),
    "brier_score": float(brier),
    "log_loss": float(logloss),
    "calibration_bins": int(len(calibration_table)),
    "model_name": artifact.get("model_name", "Saved model"),
    "saved_threshold": float(artifact["threshold"]),
    "note": (
        "Evaluation only. No recalibration or model fitting was performed. "
        "Lower Brier score and log loss are better; compare Brier scores "
        "against the test-set positive-rate baseline before drawing conclusions."
    ),
}

json_path = REPORTS_DIR / "calibration_results.json"
json_path.write_text(json.dumps(results, indent=2))

print("\nCALIBRATION RESULTS")
for key, value in results.items():
    print(f"{key}: {value}")

print("\nCalibration bins:")
print(calibration_table.to_string(index=False))
print(f"\nSaved: {csv_path.relative_to(ROOT)}")
print(f"Saved: {plot_path.relative_to(ROOT)}")
print(f"Saved: {json_path.relative_to(ROOT)}")
