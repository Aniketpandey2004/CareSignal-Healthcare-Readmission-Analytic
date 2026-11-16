
"""
CareSignal - Experiment 5
SHAP Explainability Analysis

Uses the existing saved Random Forest pipeline.
Does NOT retrain or modify the saved model.

Run from project root:
python experiments/shap_explainability_analysis.py

Analysis:
1. Reproduce original patient-level test partition.
2. Transform test data using saved preprocessing pipeline.
3. Calculate SHAP values for a reproducible test sample.
4. Generate global feature importance.
5. Generate SHAP bar and beeswarm plots.
6. Generate one patient-level waterfall explanation.
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "raw" / "diabetic_data_original.csv"

MODEL_PATH = ROOT / "models" / "healthcare_readmission_model.joblib"

OUT_DIR = ROOT / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 500
RANDOM_STATE = 42

print("\n" + "=" * 65)
print("CARESIGNAL - EXPERIMENT 5")
print("SHAP EXPLAINABILITY ANALYSIS")
print("=" * 65)


# ============================================================
# 2. IMPORT SHAP
# ============================================================

try:
    import shap
except ImportError:
    raise ImportError(
        "SHAP is not installed. Run:\n"
        "pip install shap"
    )

print("\nSHAP version:", shap.__version__)


# ============================================================
# 3. LOAD DATA AND SAVED MODEL
# ============================================================

print("\nLoading dataset and saved model...")

df = pd.read_csv(DATA_PATH, low_memory=False)

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]

print("Dataset:", df.shape)
print("Saved model loaded.")


# ============================================================
# 4. REPRODUCE ORIGINAL PREPROCESSING
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

print("Preprocessing verified.")
print("Features:", len(features))
print("Patients:", groups.nunique())


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
        "Patient overlap detected in test split."
    )

print("Test encounters:", len(X_test))
print("Test patients:", len(test_patients))
print("Patient overlap: 0")


# ============================================================
# 6. IDENTIFY PIPELINE COMPONENTS
# ============================================================

print("\nIdentifying model pipeline components...")

if not hasattr(model, "named_steps"):

    raise TypeError(
        "Saved model is not a scikit-learn Pipeline."
    )

steps = model.named_steps

if "preprocessor" not in steps:

    raise KeyError(
        "Pipeline does not contain 'preprocessor'."
    )

if "classifier" not in steps:

    raise KeyError(
        "Pipeline does not contain 'classifier'."
    )

preprocessor = steps["preprocessor"]
classifier = steps["classifier"]

print("Preprocessor:", type(preprocessor).__name__)
print("Classifier:", type(classifier).__name__)


# ============================================================
# 7. SELECT REPRODUCIBLE TEST SAMPLE
# ============================================================

print("\nSelecting test sample...")

sample_size = min(
    SAMPLE_SIZE,
    len(X_test)
)

rng = np.random.default_rng(RANDOM_STATE)

sample_positions = np.sort(
    rng.choice(
        len(X_test),
        size=sample_size,
        replace=False
    )
)

X_sample = X_test.iloc[sample_positions].copy()

y_sample = y_test.iloc[sample_positions].copy()

print("SHAP sample size:", len(X_sample))


# ============================================================
# 8. TRANSFORM TEST SAMPLE
# ============================================================

print("\nTransforming test sample...")

X_transformed = preprocessor.transform(X_sample)

if hasattr(X_transformed, "toarray"):
    X_transformed = X_transformed.toarray()

X_transformed = np.asarray(
    X_transformed,
    dtype=np.float64
)

print("Transformed shape:", X_transformed.shape)


# ============================================================
# 9. GET TRANSFORMED FEATURE NAMES
# ============================================================

print("\nExtracting transformed feature names...")

try:

    feature_names = list(
        preprocessor.get_feature_names_out()
    )

except Exception as error:

    raise RuntimeError(
        "Could not obtain transformed feature names. "
        "Check the fitted preprocessing pipeline."
    ) from error


if len(feature_names) != X_transformed.shape[1]:

    raise ValueError(
        "Transformed feature-name count does not match "
        "transformed data columns."
    )


print("Transformed features:", len(feature_names))


# ============================================================
# 10. CALCULATE SHAP VALUES
# ============================================================

print("\nCreating SHAP TreeExplainer...")

explainer = shap.TreeExplainer(classifier)

print("Calculating SHAP values...")

try:

    explanation = explainer(
        X_transformed,
        check_additivity=False
    )

    shap_values = explanation.values
    base_values = explanation.base_values

except Exception:

    # Compatibility fallback for older SHAP versions.
    old_values = explainer.shap_values(
        X_transformed,
        check_additivity=False
    )

    if isinstance(old_values, list):

        shap_values = old_values[1]

    else:

        shap_values = np.asarray(old_values)

    base_values = explainer.expected_value

shap_values = np.asarray(shap_values)


# Handle binary classifier output dimensions.
if shap_values.ndim == 3:

    if shap_values.shape[-1] == 2:

        shap_values = shap_values[:, :, 1]

    elif shap_values.shape[0] == 2:

        shap_values = shap_values[1]

    else:

        raise ValueError(
            f"Unexpected SHAP shape: {shap_values.shape}"
        )


if shap_values.ndim != 2:

    raise ValueError(
        f"Expected 2D SHAP values, got {shap_values.shape}"
    )


if shap_values.shape != X_transformed.shape:

    raise ValueError(
        "SHAP values shape does not match transformed data."
    )


print("SHAP calculation completed.")
print("SHAP shape:", shap_values.shape)


# ============================================================
# 11. GLOBAL FEATURE IMPORTANCE
# ============================================================

print("\nCalculating global feature importance...")

mean_abs_shap = np.mean(
    np.abs(shap_values),
    axis=0
)

importance_df = pd.DataFrame({

    "feature": feature_names,

    "mean_absolute_shap": mean_abs_shap

})

importance_df = importance_df.sort_values(
    by="mean_absolute_shap",
    ascending=False
).reset_index(drop=True)

importance_df.insert(
    0,
    "rank",
    np.arange(1, len(importance_df) + 1)
)

importance_path = (
    OUT_DIR / "shap_global_feature_importance.csv"
)

importance_df.to_csv(
    importance_path,
    index=False
)


# ============================================================
# 12. SHAP BAR PLOT
# ============================================================

print("\nGenerating SHAP bar plot...")

bar_explanation = shap.Explanation(

    values=shap_values,

    data=X_transformed,

    feature_names=feature_names

)

plt.figure(figsize=(10, 8))

shap.plots.bar(
    bar_explanation,
    max_display=20,
    show=False
)

plt.title(
    "Global SHAP Feature Importance - CareSignal"
)

plt.tight_layout()

bar_path = (
    OUT_DIR / "shap_summary_bar.png"
)

plt.savefig(
    bar_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 13. SHAP BEESWARM PLOT
# ============================================================

print("\nGenerating SHAP beeswarm plot...")

plt.figure(figsize=(11, 9))

shap.plots.beeswarm(
    bar_explanation,
    max_display=20,
    show=False
)

plt.title(
    "SHAP Summary Plot - CareSignal"
)

plt.tight_layout()

beeswarm_path = (
    OUT_DIR / "shap_beeswarm.png"
)

plt.savefig(
    beeswarm_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 14. PATIENT-LEVEL WATERFALL EXPLANATION
# ============================================================

print("\nGenerating patient-level explanation...")

# Select the highest-risk encounter in the sample.
sample_probabilities = model.predict_proba(
    X_sample
)[:, 1]

local_position = int(
    np.argmax(sample_probabilities)
)

local_probability = float(
    sample_probabilities[local_position]
)

local_encounter_position = int(
    sample_positions[local_position]
)

local_base = np.asarray(base_values)

if local_base.ndim == 0:

    local_base_value = float(local_base)

elif local_base.ndim == 1:

    if len(local_base) == 2:
        local_base_value = float(local_base[1])
    else:
        local_base_value = float(local_base[0])

elif local_base.ndim == 2:

    if local_base.shape[1] == 2:
        local_base_value = float(
            local_base[local_position, 1]
        )
    else:
        local_base_value = float(
            local_base[local_position, 0]
        )

else:

    raise ValueError(
        "Unexpected SHAP base value dimensions."
    )


local_explanation = shap.Explanation(

    values=shap_values[local_position],

    base_values=local_base_value,

    data=X_transformed[local_position],

    feature_names=feature_names

)

plt.figure(figsize=(11, 8))

shap.plots.waterfall(
    local_explanation,
    max_display=15,
    show=False
)

plt.title(
    "Patient-Level SHAP Explanation - "
    f"Predicted Probability: {local_probability:.4f}"
)

plt.tight_layout()

waterfall_path = (
    OUT_DIR / "shap_patient_waterfall.png"
)

plt.savefig(
    waterfall_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 15. SAVE LOCAL EXPLANATION DATA
# ============================================================

local_df = pd.DataFrame({

    "feature": feature_names,

    "transformed_value": X_transformed[local_position],

    "shap_value": shap_values[local_position],

})

local_df["absolute_shap_value"] = (
    local_df["shap_value"].abs()
)

local_df = local_df.sort_values(
    by="absolute_shap_value",
    ascending=False
)

local_path = (
    OUT_DIR / "shap_patient_explanation.csv"
)

local_df.to_csv(
    local_path,
    index=False
)


# ============================================================
# 16. SAVE EXPERIMENT SUMMARY
# ============================================================

summary = {

    "experiment": "SHAP Explainability Analysis",

    "dataset_rows": int(len(df)),

    "original_feature_count": int(len(features)),

    "transformed_feature_count": int(len(feature_names)),

    "test_encounters": int(len(X_test)),

    "test_patients": int(len(test_patients)),

    "patient_overlap": 0,

    "shap_sample_size": int(sample_size),

    "explained_encounter_position_in_test": (
        local_encounter_position
    ),

    "explained_patient_nbr": int(
        groups.iloc[
            test_idx[local_encounter_position]
        ]
    ),

    "explained_readmission_target": int(
        y_test.iloc[local_encounter_position]
    ),

    "explained_predicted_probability": (
        local_probability
    ),

    "top_20_features": [

        {
            "feature": str(row["feature"]),

            "mean_absolute_shap": float(
                row["mean_absolute_shap"]
            )

        }

        for _, row in importance_df.head(20).iterrows()

    ],

    "methodology_note": (
        "SHAP TreeExplainer was applied to the classifier "
        "inside the existing saved Random Forest pipeline. "
        "The saved preprocessing pipeline transformed the "
        "patient-level test sample. No model retraining was "
        "performed. Global feature importance was calculated "
        "using mean absolute SHAP values. A high-risk sample "
        "encounter was selected for a local waterfall "
        "explanation. SHAP values describe model prediction "
        "contributions and do not establish causality or "
        "clinical relationships."
    ),

}

summary_path = (
    OUT_DIR / "shap_explainability_results.json"
)

summary_path.write_text(
    json.dumps(summary, indent=2)
)


# ============================================================
# 17. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)

print("TOP 20 GLOBAL SHAP FEATURES")

print("=" * 70)

print(
    importance_df.head(20).to_string(
        index=False
    )
)

print("\nPATIENT-LEVEL EXPLANATION")

print("Predicted probability:", round(local_probability, 6))

print(
    "Actual readmission target:",
    int(y_test.iloc[local_encounter_position])
)

print("\nOUTPUT FILES")

for path in [
    importance_path,
    bar_path,
    beeswarm_path,
    waterfall_path,
    local_path,
    summary_path,
]:

    print(path.relative_to(ROOT))


print("\nEXPERIMENT 5 COMPLETED SUCCESSFULLY.")

print(
    "\nNote: SHAP explains model behavior, "
    "not causation or clinical decision-making."
)

print(
    "Original saved model and dashboard remain unchanged."
)
