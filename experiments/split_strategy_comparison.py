"""Compare patient-level and random encounter-level evaluation.
Run from repo root: python experiments/split_strategy_comparison.py
Does not modify train_model.py, the saved model, or the dashboard.
"""
from __future__ import annotations
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "diabetic_data_original.csv"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df.columns = df.columns.str.strip()
    required = {"patient_nbr", "encounter_id", "readmitted"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Required columns missing: {sorted(missing)}")
    for canonical, duplicate in [
        ("patient_nbr", "patient_nbr.1"),
        ("encounter_id", "encounter_id.1"),
    ]:
        if duplicate in df.columns:
            if not df[canonical].astype(str).equals(df[duplicate].astype(str)):
                raise ValueError(f"{canonical} and {duplicate} do not match.")
            df = df.drop(columns=[duplicate])
    y = df["readmitted"].astype(str).str.strip().eq("<30").astype("int8")
    groups = df["patient_nbr"].copy()
    if groups.isna().any():
        raise ValueError("Missing patient_nbr values prevent overlap measurement.")
    excluded = {"patient_nbr", "encounter_id", "readmitted", "readmitted_30"}
    X = df[[c for c in df.columns if c not in excluded]].copy().replace("?", np.nan)
    for col in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
        if col in X.columns:
            X[col] = X[col].astype("string")
    empty = [c for c in X.columns if X[c].isna().all()]
    if empty:
        X = X.drop(columns=empty)
    return X, y, groups, empty


def make_pipeline(X):
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
    ])
    prep = ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ], remainder="drop")
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=20, min_samples_split=10,
        min_samples_leaf=5, class_weight="balanced_subsample",
        n_jobs=-1, random_state=42,
    )
    return Pipeline([("preprocessor", prep), ("classifier", rf)])


def select_threshold(y_val, probabilities):
    rows = []
    for threshold in np.unique(np.append(probabilities, 0.50)):
        pred = (probabilities >= threshold).astype(int)
        rows.append({
            "threshold": float(threshold),
            "precision": precision_score(y_val, pred, zero_division=0),
            "recall": recall_score(y_val, pred, zero_division=0),
            "f1": f1_score(y_val, pred, zero_division=0),
        })
    frame = pd.DataFrame(rows)
    eligible = frame[frame["recall"] >= 0.60]
    if eligible.empty:
        chosen = frame.sort_values(["f1", "threshold"], ascending=[False, False]).iloc[0]
        rule = "Maximum validation F1; no threshold met recall >= 0.60"
    else:
        chosen = eligible.sort_values(["f1", "threshold"], ascending=[False, False]).iloc[0]
        rule = "Maximum validation F1 among thresholds with recall >= 0.60"
    return float(chosen["threshold"]), rule


def evaluate(name, X, y, groups, train_idx, test_idx, grouped):
    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
    g_train, g_test = groups.iloc[train_idx], groups.iloc[test_idx]
    if grouped:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=21)
        sub_idx, val_idx = next(splitter.split(X_train, y_train, groups=g_train))
    else:
        sub_idx, val_idx = train_test_split(
            np.arange(len(X_train)), test_size=0.20,
            random_state=21, stratify=y_train,
        )

    validation_model = make_pipeline(X)
    validation_model.fit(X_train.iloc[sub_idx], y_train.iloc[sub_idx])
    val_prob = validation_model.predict_proba(X_train.iloc[val_idx])[:, 1]
    threshold, rule = select_threshold(y_train.iloc[val_idx], val_prob)

    final_model = make_pipeline(X)
    final_model.fit(X_train, y_train)
    test_prob = final_model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, test_pred, labels=[0, 1]).ravel()

    result = {
        "Split strategy": name,
        "Train encounters": len(train_idx),
        "Test encounters": len(test_idx),
        "Train patients": int(g_train.nunique()),
        "Test patients": int(g_test.nunique()),
        "Shared patients across train-test": int(len(set(g_train).intersection(set(g_test)))),
        "Train positive rate": float(y_train.mean()),
        "Test positive rate": float(y_test.mean()),
        "Validation-selected threshold": threshold,
        "Threshold rule": rule,
        "Accuracy": accuracy_score(y_test, test_pred),
        "Precision": precision_score(y_test, test_pred, zero_division=0),
        "Recall": recall_score(y_test, test_pred, zero_division=0),
        "F1": f1_score(y_test, test_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, test_prob),
        "PR-AUC": average_precision_score(y_test, test_prob),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }
    print(f"\n{name}\n" + "-" * 65)
    for key, value in result.items():
        print(f"{key}: {value:.4f}" if isinstance(value, (float, np.floating)) else f"{key}: {value}")
    return result


def main():
    print("=" * 72)
    print("CARESIGNAL | PATIENT-LEVEL vs RANDOM ENCOUNTER-LEVEL SPLIT")
    print("=" * 72)
    print("Project:", ROOT)
    print("scikit-learn:", sklearn.__version__)
    X, y, groups, empty = load_data()
    print(f"Rows: {len(X):,} | Patients: {groups.nunique():,} | Features: {X.shape[1]}")
    print(f"Overall positive rate: {y.mean():.2%}")
    print("Removed all-missing features:", empty)

    patient_split = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    p_train, p_test = next(patient_split.split(X, y, groups=groups))
    r_train, r_test = train_test_split(
        np.arange(len(X)), test_size=0.20, random_state=42, stratify=y
    )
    results = [
        evaluate("Patient-level split", X, y, groups, p_train, p_test, grouped=True),
        evaluate("Random encounter-level split", X, y, groups, r_train, r_test, grouped=False),
    ]
    comparison = pd.DataFrame(results)
    output = REPORT_DIR / "split_strategy_comparison.csv"
    comparison.to_csv(output, index=False)
    print("\nCOMPARISON")
    print(comparison[[
        "Split strategy", "Shared patients across train-test",
        "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC",
    ]].to_string(index=False))
    print(f"\nSaved results to: {output}")
    print("\nInterpretation: random encounter splitting may place encounters from the")
    print("same patient in train and test. Treat score differences as a leakage-")
    print("sensitivity diagnostic, not evidence of better clinical generalisation.")
    print("This is a retrospective academic experiment, not clinical validation.")


if __name__ == "__main__":
    main()
