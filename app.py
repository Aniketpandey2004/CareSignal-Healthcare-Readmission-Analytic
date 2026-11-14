from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

st.set_page_config(
    page_title="CareSignal | Readmission Intelligence",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown("""
<style>
.block-container {
    padding-top: 1.4rem;
    padding-bottom: 2.2rem;
    max-width: 1500px;
}

[data-testid="stMetric"] {
    background: rgba(128,128,128,.07);
    border: 1px solid rgba(128,128,128,.18);
    padding: 14px 16px;
    border-radius: 12px;
}

.hero {
    padding: 1.2rem 1.4rem;
    border-radius: 16px;
    background: linear-gradient(
        115deg,
        rgba(27,90,130,.18),
        rgba(24,150,130,.10)
    );
    border: 1px solid rgba(100,150,170,.25);
    margin-bottom: 1rem;
}

.small-note {
    font-size: .88rem;
    opacity: .78;
}

div[data-testid="stTabs"] button {
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA LOADING FUNCTIONS
# ============================================================

@st.cache_data(show_spinner="Loading encounter dataset...")
def load_clean_data():

    candidates = [
        DATA_DIR / "cleaned_healthcare_data.csv",
        DATA_DIR / "healthcare_cleaned.csv",
    ]

    path = next((p for p in candidates if p.exists()), None)

    if path is None:
        raise FileNotFoundError(
            "Could not find data/cleaned_healthcare_data.csv"
        )

    frame = pd.read_csv(path, low_memory=False)
    frame.columns = frame.columns.str.strip()

    if "target_30" not in frame.columns:

        if "readmitted_30" in frame.columns:
            frame["target_30"] = pd.to_numeric(
                frame["readmitted_30"],
                errors="coerce"
            )

        elif "readmitted" in frame.columns:
            frame["target_30"] = (
                frame["readmitted"]
                .astype(str)
                .str.strip()
                .eq("<30")
                .astype(int)
            )

    if "target_30" in frame.columns:
        frame["target_30"] = pd.to_numeric(
            frame["target_30"],
            errors="coerce"
        )

    return frame, path.name


@st.cache_data(show_spinner="Loading original dataset...")
def load_raw_data():

    path = RAW_DIR / "diabetic_data_original.csv"

    if not path.exists():
        return None

    frame = pd.read_csv(path, low_memory=False)
    frame.columns = frame.columns.str.strip()

    return frame


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner="Loading trained model...")
def load_model_artifact():

    candidates = [
        MODEL_DIR / "healthcare_readmission_model.joblib",
        MODEL_DIR / "readmission_model.pkl",
    ]

    path = next(
        (p for p in candidates if p.exists()),
        None
    )

    if path is None:
        return None, (
            "Model artifact not found. "
            "Run train_model.py first."
        )

    try:

        loaded = joblib.load(path)

        if isinstance(loaded, dict) and "model" in loaded:
            artifact = loaded

        else:
            artifact = {
                "model": loaded
            }

        return artifact, None

    except Exception as exc:

        return None, (
            f"Could not load model: "
            f"{type(exc).__name__}: {exc}"
        )


def get_model_schema(artifact, raw):

    model = artifact["model"]

    features = artifact.get("feature_columns")

    if features is None:
        features = getattr(
            model,
            "feature_names_in_",
            None
        )

    if features is None:
        st.error(
            "Model feature schema missing. "
            "Retrain the model and save feature_columns."
        )
        st.stop()

    features = list(features)

    missing = [
        col for col in features
        if col not in raw.columns
    ]

    if missing:
        st.error(
            "Original dataset is missing model features: "
            + ", ".join(missing[:15])
        )
        st.stop()

    return model, features


# ============================================================
# REPORT UTILITIES
# ============================================================

def load_report(name):

    path = REPORT_DIR / name

    if not path.exists():
        return None

    try:
        return pd.read_csv(path)

    except Exception:
        return None


def fmt_pct(value):

    if pd.notna(value):
        return f"{value * 100:.1f}%"

    return "—"


def safe_mean(frame, col):

    if col not in frame.columns:
        return float("nan")

    return pd.to_numeric(
        frame[col],
        errors="coerce"
    ).mean()


# ============================================================
# LOAD DATASET
# ============================================================

try:

    df, dataset_name = load_clean_data()

except Exception as exc:

    st.error(f"Dataset loading failed: {exc}")
    st.stop()


if "target_30" not in df.columns:

    st.error(
        "Target column missing. "
        "Expected readmitted_30 or readmitted."
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## 🏥 CareSignal")
st.sidebar.caption("Healthcare Readmission Intelligence")
st.sidebar.markdown("---")

st.sidebar.markdown("**Explore the historical cohort**")

filtered = df.copy()

for col in [
    c for c in ["gender", "age", "race"]
    if c in filtered.columns
]:

    options = sorted(
        filtered[col]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected = st.sidebar.multiselect(
        col.replace("_", " ").title(),
        options,
        default=options,
        key=f"filter_{col}"
    )

    if selected:
        filtered = filtered[
            filtered[col].astype(str).isin(selected)
        ]


st.sidebar.markdown("---")

st.sidebar.caption(
    "Dataset: UCI Diabetes 130-US Hospitals (1999–2008)"
)

st.sidebar.caption(
    "Unit of analysis: hospital encounter"
)

st.sidebar.caption(
    "Target: readmission within 30 days"
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="hero">',
    unsafe_allow_html=True
)

st.title("CareSignal | Readmission Intelligence")

st.markdown(
    "Explore historical readmission patterns, "
    "review model performance, and test an "
    "academic risk-scoring prototype."
)

st.markdown(
    f'<p class="small-note">Source: '
    f'<b>{dataset_name}</b> · '
    f'Filtered encounters: '
    f'<b>{len(filtered):,}</b></p>',
    unsafe_allow_html=True
)

st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# APPLICATION TABS
# ============================================================

(
    tab_overview,
    tab_population,
    tab_model,
    tab_predict,
    tab_method
) = st.tabs([
    "📊 Executive Overview",
    "🧭 Population Insights",
    "🧠 Model Performance",
    "🩺 Risk Scoring Demo",
    "📘 Methodology",
])


# ============================================================
# TAB 1: EXECUTIVE OVERVIEW
# ============================================================

with tab_overview:

    target = filtered["target_30"].dropna()

    total = len(filtered)

    readmit_n = (
        int(target.sum())
        if len(target)
        else 0
    )

    rate = (
        float(target.mean())
        if len(target)
        else float("nan")
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Hospital encounters",
        f"{total:,}"
    )

    c2.metric(
        "30-day readmissions",
        f"{readmit_n:,}"
    )

    c3.metric(
        "30-day readmission rate",
        fmt_pct(rate)
    )

    los = safe_mean(
        filtered,
        "time_in_hospital"
    )

    c4.metric(
        "Average length of stay",
        f"{los:.1f} days"
        if pd.notna(los)
        else "—"
    )

    left, right = st.columns(2)

    with left:

        counts = (
            target.value_counts()
            .reindex([0, 1], fill_value=0)
            .rename(
                index={
                    0: "No readmission <30d",
                    1: "Readmitted <30d"
                }
            )
            .rename_axis("Outcome")
            .reset_index(name="Encounters")
        )

        fig = px.pie(
            counts,
            names="Outcome",
            values="Encounters",
            hole=.58,
            title="30-day outcome distribution"
        )

        fig.update_traces(
            textposition="inside",
            textinfo="percent+label"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        if "age" in filtered.columns:

            age = (
                filtered.groupby(
                    "age",
                    dropna=False
                )["target_30"]
                .agg(
                    rate="mean",
                    encounters="size"
                )
                .reset_index()
            )

            age["rate_pct"] = age["rate"] * 100

            age["sort"] = pd.to_numeric(
                age["age"]
                .astype(str)
                .str.extract(r"(\d+)")[0],
                errors="coerce"
            )

            age = age.sort_values("sort")

            fig = px.bar(
                age,
                x="age",
                y="rate_pct",
                hover_data={"encounters": True},
                title="Observed readmission rate by age band",
                labels={
                    "age": "Age band",
                    "rate_pct": "Readmission rate (%)"
                }
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    st.info(
        "Retrospective encounter-level associations from "
        "historical US data. These are not causal effects "
        "or current hospital benchmarks."
    )


# ============================================================
# TAB 2: POPULATION INSIGHTS
# ============================================================

with tab_population:

    st.subheader("Patient and utilization patterns")

    a, b = st.columns(2)

    with a:

        if "gender" in filtered.columns:

            g = (
                filtered.groupby("gender")["target_30"]
                .agg(
                    rate="mean",
                    encounters="size"
                )
                .reset_index()
            )

            g["rate_pct"] = g["rate"] * 100

            fig = px.bar(
                g,
                x="gender",
                y="rate_pct",
                text="encounters",
                title="Observed rate by recorded gender",
                labels={
                    "rate_pct": "Readmission rate (%)"
                }
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    with b:

        if "race" in filtered.columns:

            r = (
                filtered.groupby("race")["target_30"]
                .agg(
                    rate="mean",
                    encounters="size"
                )
                .reset_index()
            )

            r["rate_pct"] = r["rate"] * 100

            r = (
                r.sort_values(
                    "rate_pct",
                    ascending=True
                )
                .tail(12)
            )

            fig = px.bar(
                r,
                x="rate_pct",
                y="race",
                orientation="h",
                title="Observed rate by recorded race",
                labels={
                    "rate_pct": "Readmission rate (%)"
                }
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    util_cols = [
        c for c in [
            "number_inpatient",
            "number_emergency",
            "number_outpatient"
        ]
        if c in filtered.columns
    ]

    if util_cols:

        util = (
            filtered.groupby("target_30")[util_cols]
            .mean()
            .T
            .reset_index()
            .rename(
                columns={
                    "index": "Prior utilization"
                }
            )
            .rename(
                columns={
                    0: "No readmission <30d",
                    1: "Readmitted <30d"
                }
            )
        )

        fig = go.Figure()

        for col in [
            "No readmission <30d",
            "Readmitted <30d"
        ]:

            if col in util.columns:

                fig.add_trace(
                    go.Bar(
                        name=col,
                        x=util["Prior utilization"],
                        y=util[col]
                    )
                )

        fig.update_layout(
            barmode="group",
            title="Mean prior utilization by outcome",
            yaxis_title="Mean count"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("#### Cohort data preview")

    st.dataframe(
        filtered.head(100),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download filtered cohort CSV",
        filtered.to_csv(index=False).encode("utf-8"),
        "filtered_readmission_cohort.csv",
        "text/csv"
    )


# ============================================================
# TAB 3: MODEL PERFORMANCE
# ============================================================

with tab_model:

    st.subheader(
        "Model evaluation — patient-level holdout"
    )

    metrics_df = load_report(
        "patient_level_final_metrics.csv"
    )

    if metrics_df is None or metrics_df.empty:

        metrics_df = load_report(
            "final_model_summary.csv"
        )

    if metrics_df is None or metrics_df.empty:

        st.warning(
            "No final metrics report found. "
            "Run train_model.py first."
        )

    else:

        row = metrics_df.iloc[0]

        metric_map = [
            ("Precision", ["Precision", "precision"]),
            ("Recall", ["Recall", "recall"]),
            ("F1", ["F1", "f1"]),
            ("ROC-AUC", ["ROC_AUC", "ROC-AUC", "roc_auc"]),
            ("PR-AUC", ["PR_AUC", "PR-AUC", "pr_auc"]),
        ]

        cols = st.columns(5)

        for col, (label, keys) in zip(
            cols,
            metric_map
        ):

            value = next(
                (
                    row.get(k)
                    for k in keys
                    if k in row.index
                ),
                np.nan
            )

            value = pd.to_numeric(
                value,
                errors="coerce"
            )

            col.metric(
                label,
                f"{value:.3f}"
                if pd.notna(value)
                else "—"
            )

        model_name = row.get(
            "Model",
            row.get("model", "—")
        )

        threshold = row.get(
            "Threshold",
            row.get("threshold", np.nan)
        )

        try:
            threshold_text = f"{float(threshold):.3f}"

        except (TypeError, ValueError):
            threshold_text = "—"

        st.caption(
            f"Selected model: {model_name} · "
            f"Decision threshold: {threshold_text}"
        )

    comparison = load_report(
        "patient_level_model_comparison.csv"
    )

    if comparison is None or comparison.empty:

        comparison = load_report(
            "improved_model_comparison.csv"
        )

    if comparison is not None and not comparison.empty:

        st.markdown("#### Model comparison")

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True
        )

        pr_col = next(
            (
                c for c in [
                    "PR_AUC",
                    "PR-AUC",
                    "pr_auc"
                ]
                if c in comparison.columns
            ),
            None
        )

        if pr_col:

            model_col = next(
                (
                    c for c in ["Model", "model"]
                    if c in comparison.columns
                ),
                comparison.columns[0]
            )

            fig = px.bar(
                comparison,
                x=model_col,
                y=pr_col,
                title="Average precision (PR-AUC) comparison",
                text_auto=".3f"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    cm_df = load_report(
        "patient_level_confusion_matrix.csv"
    )

    if cm_df is not None and not cm_df.empty:

        numeric = cm_df.select_dtypes(
            include=np.number
        )

        if numeric.shape[1] >= 2:

            values = numeric.iloc[:, -2:].to_numpy()

            if values.shape == (2, 2):

                fig = px.imshow(
                    values,
                    text_auto=True,
                    x=["Predicted 0", "Predicted 1"],
                    y=["Actual 0", "Actual 1"],
                    color_continuous_scale="Blues",
                    title="Holdout confusion matrix"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

    imp = load_report(
        "patient_level_feature_importance.csv"
    )

    if imp is None or imp.empty:

        imp = load_report(
            "final_feature_importance.csv"
        )

    if imp is not None and not imp.empty:

        feature_col = next(
            (
                c for c in ["Feature", "feature"]
                if c in imp.columns
            ),
            None
        )

        importance_col = next(
            (
                c for c in ["Importance", "importance"]
                if c in imp.columns
            ),
            None
        )

        if feature_col and importance_col:

            top = (
                imp.sort_values(
                    importance_col,
                    ascending=False
                )
                .head(15)
                .sort_values(importance_col)
            )

            fig = px.bar(
                top,
                x=importance_col,
                y=feature_col,
                orientation="h",
                title="Top model feature importances"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.caption(
                "Feature importance is descriptive "
                "and does not establish causality."
            )

    metadata_path = (
        REPORT_DIR / "training_metadata.json"
    )

    if metadata_path.exists():

        try:

            meta = json.loads(
                metadata_path.read_text()
            )

            st.markdown("#### Evaluation design")

            m1, m2, m3 = st.columns(3)

            m1.metric(
                "Training patients",
                f"{meta.get('train_patients', 0):,}"
            )

            m2.metric(
                "Test patients",
                f"{meta.get('test_patients', 0):,}"
            )

            overlap = meta.get(
                "patient_overlap_train_test",
                meta.get("patient_overlap", "—")
            )

            m3.metric(
                "Patient overlap",
                str(overlap)
            )

            st.caption(
                "Patient-grouped evaluation is used "
                "to reduce patient leakage."
            )

        except Exception:
            pass


# ============================================================
# TAB 4: LIVE RISK PREDICTION
# ============================================================

with tab_predict:

    st.subheader(
        "Individual encounter risk-scoring demonstration"
    )

    artifact, model_error = load_model_artifact()

    raw = load_raw_data()

    if artifact is None:

        st.warning(
            model_error or "Train a model first."
        )

    elif raw is None:

        st.warning(
            "Missing data/raw/diabetic_data_original.csv."
        )

    else:

        model, features = get_model_schema(
            artifact,
            raw
        )

        # ----------------------------------------------------
        # Create baseline feature values
        # ----------------------------------------------------

        defaults = {}

        for col in features:

            series = raw[col].replace(
                "?",
                np.nan
            )

            if pd.api.types.is_numeric_dtype(series):

                defaults[col] = pd.to_numeric(
                    series,
                    errors="coerce"
                ).median()

            else:

                modes = series.dropna().mode()

                defaults[col] = (
                    modes.iloc[0]
                    if not modes.empty
                    else np.nan
                )

        user_row = pd.DataFrame(
            [defaults],
            columns=features
        )

        # ----------------------------------------------------
        # User editable features
        # ----------------------------------------------------

        preferred = [
            "age",
            "gender",
            "race",
            "time_in_hospital",
            "num_medications",
            "number_inpatient",
            "number_emergency",
            "number_outpatient",
            "number_diagnoses",
            "num_lab_procedures",
            "num_procedures",
            "discharge_disposition_id",
            "admission_type_id",
            "admission_source_id",
            "insulin",
            "diabetesMed",
            "change",
            "max_glu_serum",
            "weight",
        ]

        editable = [
            c for c in preferred
            if c in features
        ]

        st.markdown(
            "Edit selected encounter details. "
            "Remaining inputs are held at typical "
            "values from the historical dataset."
        )

        # ----------------------------------------------------
        # Prediction form
        # ----------------------------------------------------

        with st.form("risk_demo_form"):

            cols = st.columns(3)

            edited = {}

            for i, col_name in enumerate(editable):

                with cols[i % 3]:

                    series = (
                        raw[col_name]
                        .replace("?", np.nan)
                        .dropna()
                    )

                    if pd.api.types.is_numeric_dtype(
                        raw[col_name]
                    ):

                        values = pd.to_numeric(
                            series,
                            errors="coerce"
                        ).dropna()

                        lo = (
                            float(values.quantile(.01))
                            if len(values)
                            else 0.0
                        )

                        hi = (
                            float(values.quantile(.99))
                            if len(values)
                            else 100.0
                        )

                        default = user_row.iloc[0][col_name]

                        default = (
                            float(default)
                            if pd.notna(default)
                            else lo
                        )

                        if hi <= lo:
                            hi = lo + 1.0

                        default = float(
                            np.clip(default, lo, hi)
                        )

                        edited[col_name] = st.number_input(
                            col_name.replace(
                                "_", " "
                            ).title(),

                            min_value=lo,
                            max_value=hi,
                            value=default,
                            step=1.0
                        )

                    else:

                        options = sorted(
                            series.astype(str)
                            .unique()
                            .tolist()
                        )[:100]

                        default = (
                            str(user_row.iloc[0][col_name])
                            if pd.notna(
                                user_row.iloc[0][col_name]
                            )
                            else "Unknown"
                        )

                        if default not in options:
                            options = [default] + options

                        if not options:
                            options = ["Unknown"]

                        edited[col_name] = st.selectbox(
                            col_name.replace(
                                "_", " "
                            ).title(),

                            options,

                            index=(
                                options.index(default)
                                if default in options
                                else 0
                            )
                        )

            submitted = st.form_submit_button(
                "Estimate readmission risk",
                type="primary",
                use_container_width=True
            )

        # ----------------------------------------------------
        # Calculate prediction
        # ----------------------------------------------------

        if submitted:

            for key, value in edited.items():

                user_row.loc[0, key] = value

            try:

                input_row = user_row.reindex(
                    columns=features
                )

                probability = float(
                    model.predict_proba(input_row)[0, 1]
                )

                threshold = float(
                    artifact.get("threshold", 0.5)
                )

                predicted = probability >= threshold

                st.markdown("---")

                r1, r2 = st.columns([1, 2])

                with r1:

                    st.metric(
                        "Estimated model score",
                        f"{probability:.1%}"
                    )

                    st.metric(
                        "Model decision threshold",
                        f"{threshold:.1%}"
                    )

                with r2:

                    if predicted:

                        st.warning(
                            "Above the selected model threshold — "
                            "flagged for review in this demonstration."
                        )

                    else:

                        st.success(
                            "Below the selected model threshold "
                            "in this demonstration."
                        )

                    st.caption(
                        "This score is not a diagnosis, an individual "
                        "clinical probability, or a treatment recommendation. "
                        "The model uses historical US data from 1999–2008 "
                        "and requires external validation."
                    )

                # ------------------------------------------------
                # Explainability: SHAP (local, model-specific)
                # ------------------------------------------------

                st.markdown("---")
                st.subheader("Why did the model produce this score?")

                st.caption(
                    "SHAP values describe how encoded input features move this model's "
                    "output relative to its background dataset. They are predictive "
                    "associations, not medical causes or treatment advice."
                )

                try:
                    import shap
                    from scipy import sparse

                    # The training artifact is expected to be a fitted sklearn Pipeline.
                    # Transform raw inputs with the pipeline's fitted preprocessing,
                    # then explain the final tree estimator in its encoded feature space.

                    if not hasattr(model, "named_steps"):
                        raise TypeError(
                            "SHAP currently requires the saved model to be a fitted "
                            "scikit-learn Pipeline with preprocessing and estimator steps."
                        )

                    steps = list(model.named_steps.values())

                    if len(steps) < 2:
                        raise TypeError(
                            "Pipeline must contain preprocessing and a final estimator."
                        )

                    preprocessor = steps[-2]
                    estimator = steps[-1]

                    if not hasattr(estimator, "feature_importances_"):
                        raise TypeError(
                            "The final estimator is not a supported tree model. "
                            "Use a tree-based estimator such as ExtraTrees or RandomForest."
                        )

                    # Use a reproducible sample of historical rows as background.

                    background_raw = raw.reindex(
                        columns=features
                    ).sample(
                        n=min(80, len(raw)),
                        random_state=42
                    )

                    background_encoded = model[:-1].transform(
                        background_raw
                    )

                    instance_encoded = model[:-1].transform(
                        input_row
                    )

                    if sparse.issparse(background_encoded):
                        background_encoded = background_encoded.toarray()

                    if sparse.issparse(instance_encoded):
                        instance_encoded = instance_encoded.toarray()

                    background_encoded = np.asarray(
                        background_encoded,
                        dtype=float
                    )

                    instance_encoded = np.asarray(
                        instance_encoded,
                        dtype=float
                    )

                    try:
                        encoded_names = list(
                            preprocessor.get_feature_names_out()
                        )

                    except Exception:
                        encoded_names = [
                            f"Feature {i + 1}"
                            for i in range(instance_encoded.shape[1])
                        ]

                    explainer = shap.TreeExplainer(
                        estimator,
                        data=background_encoded
                    )

                    shap_values = explainer.shap_values(
                        instance_encoded
                    )

                    # Handle different SHAP output formats.

                    if isinstance(shap_values, list):

                        contribution = np.asarray(
                            shap_values[-1]
                        ).reshape(-1)

                    else:

                        values_array = np.asarray(
                            shap_values
                        )

                        if values_array.ndim == 3:

                            if values_array.shape[0] == 1:

                                contribution = values_array[
                                    0, :, -1
                                ].reshape(-1)

                            elif values_array.shape[1] == 1:

                                contribution = values_array[
                                    -1, 0, :
                                ].reshape(-1)

                            else:

                                contribution = values_array[
                                    0, :, -1
                                ].reshape(-1)

                        else:

                            contribution = values_array.reshape(-1)

                    if len(contribution) != len(encoded_names):

                        raise ValueError(
                            "SHAP output does not match the transformed feature names."
                        )

                    shap_df = pd.DataFrame({
                        "Feature": encoded_names,
                        "SHAP value": contribution,
                        "Absolute impact": np.abs(contribution),
                    })

                    shap_df = (
                        shap_df
                        .sort_values(
                            "Absolute impact",
                            ascending=False
                        )
                        .head(15)
                    )

                    chart_df = shap_df.sort_values(
                        "SHAP value",
                        ascending=True
                    )

                    fig = px.bar(
                        chart_df,
                        x="SHAP value",
                        y="Feature",
                        orientation="h",
                        color="SHAP value",
                        color_continuous_scale="RdBu",
                        color_continuous_midpoint=0,
                        title="Top 15 local SHAP contributions",
                        labels={
                            "SHAP value": "Contribution to model output"
                        },
                    )

                    fig.update_layout(
                        coloraxis_showscale=False
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                    st.dataframe(
                        shap_df[
                            ["Feature", "SHAP value", "Absolute impact"]
                        ].sort_values(
                            "Absolute impact",
                            ascending=False
                        ),
                        use_container_width=True,
                        hide_index=True
                    )

                    st.caption(
                        "Positive SHAP values push the model output toward the "
                        "readmission class; negative values push it away. "
                        "Values are in the model output scale, not a causal effect "
                        "or guaranteed change in clinical risk."
                    )

                except ImportError:

                    st.warning(
                        "SHAP package is not installed. "
                        "Install it with: pip install shap"
                    )

                except Exception as shap_error:

                    st.warning(
                        f"SHAP explanation unavailable: "
                        f"{type(shap_error).__name__}: {shap_error}"
                    )

            except Exception as exc:

                st.error(
                    f"Prediction failed: "
                    f"{type(exc).__name__}: {exc}"
                )


# ============================================================
# TAB 5: METHODOLOGY
# ============================================================

with tab_method:

    st.subheader(
        "Study design and responsible use"
    )

    st.markdown("""
**Dataset.** UCI Diabetes 130-US Hospitals, covering hospital encounters from 1999–2008. Each row represents an encounter, not a unique person.

**Outcome.** `readmitted == '<30'` is coded as 1; `NO` and `>30` are coded as 0.

**Leakage control.** `patient_nbr` and `encounter_id` must be excluded from model inputs. Patient identifiers are used only to group encounters so a patient does not appear in both training and test partitions.

**Model development.** Preprocessing should be fitted inside a scikit-learn pipeline. Threshold selection must use validation data; the held-out test partition is reserved for final evaluation.

**Interpretation.** Precision, recall, F1, ROC-AUC and PR-AUC are reported. Accuracy alone is insufficient for an imbalanced outcome. Feature importance indicates predictive association, not cause.

**Limitations.** The dataset is old, US-specific and diabetes-focused. Results may not transfer to current hospitals, other populations or Indian healthcare settings. No prospective or external validation is included.

**Intended use.** Academic prototype only. Not for diagnosis, treatment decisions, triage or real clinical deployment without clinical governance, privacy review, prospective evaluation and external validation.
""")

    st.markdown("#### Project outputs")

    st.code(
        """models/healthcare_readmission_model.joblib
models/readmission_model.pkl
reports/patient_level_model_comparison.csv
reports/patient_level_final_metrics.csv
reports/patient_level_confusion_matrix.csv
reports/patient_level_feature_importance.csv
reports/training_metadata.json""",
        language="text"
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "CareSignal · Healthcare Readmission Analytics · "
    "Academic prototype · UCI historical dataset (1999–2008)"
)