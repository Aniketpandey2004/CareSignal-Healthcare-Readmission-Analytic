"""Train a patient-grouped 30-day readmission model.

Run from the repository root:
    python train_model.py
Or from notebooks/04_modeling.ipynb with %run ../train_model.py
"""
from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "raw" / "diabetic_data_original.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 72)
print("HEALTHCARE READMISSION | PATIENT-GROUPED MODEL TRAINING")
print("=" * 72)
print("Project:", ROOT)
print("scikit-learn:", sklearn.__version__)
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

# 1. Load source data and normalize duplicate ID columns created by CSV export.
df = pd.read_csv(DATA_PATH, low_memory=False)
df.columns = df.columns.str.strip()
required = {"patient_nbr", "encounter_id", "readmitted"}
missing = required.difference(df.columns)
if missing:
    raise ValueError(f"Required columns missing: {sorted(missing)}")

# Verify duplicated ID copies agree, then keep the canonical columns only.
for canonical, duplicate in [("patient_nbr", "patient_nbr.1"), ("encounter_id", "encounter_id.1")]:
    if duplicate in df.columns:
        same = df[canonical].astype(str).equals(df[duplicate].astype(str))
        if not same:
            raise ValueError(f"{canonical} and {duplicate} do not match; inspect raw export.")
        df = df.drop(columns=[duplicate])

# 2. Define the outcome: only '<30' is a positive 30-day readmission.
df["readmitted_30"] = df["readmitted"].astype(str).str.strip().eq("<30").astype("int8")
patient_groups = df["patient_nbr"].copy()
if patient_groups.isna().any():
    raise ValueError("Missing patient_nbr values prevent patient-level splitting.")

# 3. Remove identifiers and outcome columns from model inputs.
exclude = {"patient_nbr", "encounter_id", "readmitted", "readmitted_30"}
feature_columns = [c for c in df.columns if c not in exclude]
X = df[feature_columns].copy().replace("?", np.nan)

# Treat coded hospital IDs as categorical variables
coded_categorical_cols = [
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
]

for col in coded_categorical_cols:
    if col in X.columns:
        X[col] = X[col].astype("string")

y = df["readmitted_30"].copy()
# Drop only columns with no observed values at all.
empty_columns = [c for c in X.columns if X[c].isna().all()]
if empty_columns:
    X = X.drop(columns=empty_columns)
    feature_columns = [c for c in feature_columns if c not in empty_columns]

print(f"\nRows: {len(df):,} | Unique patients: {patient_groups.nunique():,}")
print(f"Features: {X.shape[1]} | 30-day positive rate: {y.mean():.2%}")
print("Positive class counts:", y.value_counts().sort_index().to_dict())
print("Removed all-missing features:", empty_columns)

# 4. Patient-level train/test split. A patient's encounters cannot cross partitions.
outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = next(outer.split(X, y, groups=patient_groups))
X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
g_train, g_test = patient_groups.iloc[train_idx], patient_groups.iloc[test_idx]
if set(g_train).intersection(set(g_test)):
    raise AssertionError("Patient overlap detected between train and test.")

# 5. Grouped validation split inside training data; test remains untouched.
inner = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=21)
sub_idx, val_idx = next(inner.split(X_train, y_train, groups=g_train))
X_sub, X_val = X_train.iloc[sub_idx].copy(), X_train.iloc[val_idx].copy()
y_sub, y_val = y_train.iloc[sub_idx].copy(), y_train.iloc[val_idx].copy()
g_sub, g_val = g_train.iloc[sub_idx], g_train.iloc[val_idx]
assert not set(g_sub).intersection(set(g_val))
assert not set(g_sub).intersection(set(g_test))
assert not set(g_val).intersection(set(g_test))

print("\nPATIENT-LEVEL PARTITIONS")
for label, xx, yy, gg in [
    ("Train", X_train, y_train, g_train),
    ("Validation", X_val, y_val, g_val),
    ("Test", X_test, y_test, g_test),
]:
    print(f"{label:10s}: {len(xx):,} encounters | {gg.nunique():,} patients | positive rate {yy.mean():.2%}")
print("Train/test patient overlap:", len(set(g_train).intersection(set(g_test))))

# 6. Fit preprocessing only on the sub-training data.
numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = [c for c in X.columns if c not in numeric_cols]
numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
    ("scaler", StandardScaler()),
])
category_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
    ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
])
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipe, numeric_cols),
    ("categorical", category_pipe, categorical_cols),
], remainder="drop")

# 7. Compare complementary algorithms on validation data.
model_specs = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced", max_iter=2000, solver="liblinear", random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=20, min_samples_split=10,
        min_samples_leaf=5, class_weight="balanced_subsample",
        n_jobs=-1, random_state=42
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=300, max_depth=20, min_samples_split=10,
        min_samples_leaf=5, class_weight="balanced",
        n_jobs=-1, random_state=42
    ),
}
validation_rows, fitted_validation_models = [], {}
for name, estimator in model_specs.items():
    print(f"\nTraining {name} ...")
    pipe = Pipeline([("preprocessor", preprocessor), ("classifier", estimator)])
    pipe.fit(X_sub, y_sub)
    prob = pipe.predict_proba(X_val)[:, 1]
    pred = (prob >= 0.50).astype(int)
    row = {
        "Model": name,
        "Precision": precision_score(y_val, pred, zero_division=0),
        "Recall": recall_score(y_val, pred, zero_division=0),
        "F1": f1_score(y_val, pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_val, prob),
        "PR_AUC": average_precision_score(y_val, prob),
    }
    validation_rows.append(row)
    fitted_validation_models[name] = pipe
    print(" | ".join(f"{k}: {v:.4f}" for k, v in row.items() if k != "Model"))

comparison = pd.DataFrame(validation_rows).sort_values("PR_AUC", ascending=False).reset_index(drop=True)
comparison.to_csv(REPORT_DIR / "patient_level_model_comparison.csv", index=False)
best_name = str(comparison.iloc[0]["Model"])
print("\nMODEL COMPARISON (validation, threshold 0.50)")
print(comparison.to_string(index=False))
print("Selected by validation PR-AUC:", best_name)

# 8. Select a threshold on validation data only. 0.60 recall is an exploratory
# project rule, not a clinically established operating requirement.
val_prob = fitted_validation_models[best_name].predict_proba(X_val)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_val, val_prob)
threshold_rows = []
for threshold in thresholds:
    pred = (val_prob >= threshold).astype(int)
    threshold_rows.append({
        "Threshold": float(threshold),
        "Precision": precision_score(y_val, pred, zero_division=0),
        "Recall": recall_score(y_val, pred, zero_division=0),
        "F1": f1_score(y_val, pred, zero_division=0),
    })
threshold_df = pd.DataFrame(threshold_rows)
eligible = threshold_df[threshold_df["Recall"] >= 0.60]
if eligible.empty:
    chosen = threshold_df.sort_values("F1", ascending=False).iloc[0]
    threshold_rule = "Maximum validation F1; no threshold met 0.60 recall"
else:
    chosen = eligible.sort_values("F1", ascending=False).iloc[0]
    threshold_rule = "Maximum validation F1 among thresholds with recall >= 0.60"
selected_threshold = float(chosen["Threshold"])
threshold_df.to_csv(REPORT_DIR / "patient_level_threshold_analysis.csv", index=False)
print(f"\nSelected threshold: {selected_threshold:.4f} | rule: {threshold_rule}")

# 9. Refit selected algorithm on all training data; evaluate once on holdout test.
final_model = Pipeline([("preprocessor", preprocessor), ("classifier", model_specs[best_name])])
print("\nRefitting selected model on all training encounters ...")
final_model.fit(X_train, y_train)
test_prob = final_model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= selected_threshold).astype(int)
cm = confusion_matrix(y_test, test_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
metrics = {
    "Model": best_name,
    "Threshold": selected_threshold,
    "Accuracy": accuracy_score(y_test, test_pred),
    "Precision": precision_score(y_test, test_pred, zero_division=0),
    "Recall": recall_score(y_test, test_pred, zero_division=0),
    "F1": f1_score(y_test, test_pred, zero_division=0),
    "ROC_AUC": roc_auc_score(y_test, test_prob),
    "PR_AUC": average_precision_score(y_test, test_prob),
}
print("\nFINAL HOLDOUT TEST METRICS")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}" if isinstance(v, (float, np.floating)) else f"{k}: {v}")
print("\nConfusion matrix [[TN, FP], [FN, TP]]:")
print(cm)
print("\nClassification report:")
print(classification_report(y_test, test_pred, target_names=["Not readmitted <30d", "Readmitted <30d"], zero_division=0))

# 10. Explainability: impurity-based tree importance, descriptive not causal.
classifier = final_model.named_steps["classifier"]
if hasattr(classifier, "feature_importances_"):
    names = final_model.named_steps["preprocessor"].get_feature_names_out()
    importance = pd.DataFrame({"Feature": names, "Importance": classifier.feature_importances_})
    importance = importance.sort_values("Importance", ascending=False)
else:
    importance = pd.DataFrame(columns=["Feature", "Importance"])
importance.to_csv(REPORT_DIR / "patient_level_feature_importance.csv", index=False)

# 11. Persist a single consistent artifact consumed by the dashboard.
artifact = {
    "model": final_model,
    "threshold": selected_threshold,
    "model_name": best_name,
    "feature_columns": feature_columns,
    "numeric_features": numeric_cols,
    "categorical_features": categorical_cols,
    "target_definition": "readmitted == '<30'",
    "threshold_rule": threshold_rule,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "sklearn_version": sklearn.__version__,
    "dataset": "UCI Diabetes 130-US Hospitals (1999-2008)",
    "evaluation": metrics,
}
joblib.dump(artifact, MODEL_DIR / "healthcare_readmission_model.joblib", compress=3)
pd.DataFrame([metrics]).to_csv(REPORT_DIR / "patient_level_final_metrics.csv", index=False)
pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]).to_csv(REPORT_DIR / "patient_level_confusion_matrix.csv")
metadata = {
    "model_name": best_name,
    "threshold": selected_threshold,
    "threshold_rule": threshold_rule,
    "train_records": int(len(X_train)),
    "validation_records": int(len(X_val)),
    "test_records": int(len(X_test)),
    "train_patients": int(g_train.nunique()),
    "validation_patients": int(g_val.nunique()),
    "test_patients": int(g_test.nunique()),
    "patient_overlap_train_test": int(len(set(g_train).intersection(set(g_test)))),
    "features_used": int(X.shape[1]),
    "positive_rate_train": float(y_train.mean()),
    "positive_rate_validation": float(y_val.mean()),
    "positive_rate_test": float(y_test.mean()),
    "sklearn_version": sklearn.__version__,
    "metrics": {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in metrics.items()},
    "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
}
with open(REPORT_DIR / "training_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\nTOP 20 FEATURE IMPORTANCES")
print(importance.head(20).to_string(index=False) if not importance.empty else "Selected model does not expose tree feature_importances_.")
print("\nSaved model:", MODEL_DIR / "healthcare_readmission_model.joblib")
print("Saved reports:", REPORT_DIR)
print("Patient-level separation check: PASSED (zero shared patients).")
print("NOTE: This is a retrospective academic prototype, not a clinically validated tool.")
