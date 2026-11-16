# CareSignal — Healthcare Readmission Intelligence

An academic machine-learning and analytics prototype for retrospective analysis of 30-day hospital readmission patterns among diabetic patients.

> **Academic-use disclaimer:** CareSignal is an educational/research prototype. It is not a clinical diagnostic, prognostic, triage, discharge, treatment or care-allocation tool. Its outputs must not be used to make patient-care decisions. No external or prospective clinical validation has been performed.

## Project scope

- Encounter-level analytics dashboard built with Streamlit and Plotly.
- Patient-grouped model training pipeline (`train_model.py`).
- Historical cohort analysis using the UCI Diabetes 130-US Hospitals dataset.
- Model comparison, validation-only threshold selection and held-out patient-level evaluation.
- Separate research analyses covering split strategy, calibration, bootstrap uncertainty, SHAP explanations and descriptive subgroup performance.

The project is intended for **academic demonstration only**. The risk-scoring screen is a model-behaviour demonstration on a synthetic/hybrid input row: fields not explicitly edited are populated with typical values from the historical dataset. The displayed score is not a validated individual clinical probability.

## Dataset

The project uses the UCI **Diabetes 130-US Hospitals for Years 1999–2008** dataset. The source dataset contains 101,766 encounters and 71,518 unique patients. Each row represents a hospital encounter, not a unique patient.

Target definition:

- `readmitted == '<30'` → `readmitted_30 = 1`
- `readmitted == 'NO'` or `'>30'` → `readmitted_30 = 0`

The patient and encounter identifiers are excluded from predictive inputs. Patient identifiers are used only to group encounters during splitting.

## Primary model and evaluation

The primary saved artifact is `models/healthcare_readmission_model.joblib`. Its accompanying metadata is `reports/training_metadata.json`. Older model artifacts are retained for historical traceability; do not assume they are interchangeable with the primary artifact.

The documented primary patient-level holdout results are:

| Metric | Result |
|---|---:|
| Test encounters | 20,153 |
| Test patients | 14,304 |
| Train/test patient overlap | 0 |
| Features | 47 |
| Threshold | 0.481041 |
| Accuracy | 0.6220 |
| Precision | 0.1651 |
| Recall | 0.6267 |
| F1-score | 0.2614 |
| ROC-AUC | 0.6746 |
| PR-AUC (average precision) | 0.2085 |

Confusion matrix: TN=11,187; FP=6,815; FN=803; TP=1,348.

The positive class prevalence in the test partition is approximately 10.67%. Precision is therefore important to interpret alongside recall; the selected operating point produces a substantial number of false-positive flags. The 60% validation-recall floor used for threshold selection is an exploratory project rule, not a clinically established requirement.

### Evaluation design

- Outer patient-grouped split: 80% training / 20% held-out test, `random_state=42`.
- Inner patient-grouped validation split from the training partition, `random_state=21`.
- Preprocessing is fitted within the scikit-learn pipeline.
- Candidate models are compared on validation average precision (PR-AUC).
- Threshold selection uses validation data only.
- The held-out test partition is reserved for final evaluation.

See `reports/training_metadata.json` and `reports/patient_level_final_metrics.csv` for the primary run's recorded values.

## Research analyses

A six-study analysis set has been completed in the project work:

1. **Split strategy:** patient-grouped versus random encounter split. The random split had 7,001 patients shared across train and test, so it is not the preferred estimate of unseen-patient generalization.
2. **Calibration comparison:** uncalibrated, sigmoid and isotonic probability calibration. Calibration improved Brier score and log loss in the separate calibration experiment, but changes score scale and requires threshold re-selection.
3. **Patient-cluster bootstrap:** 1,000 valid patient-cluster bootstrap replicates for uncertainty intervals around ROC-AUC and PR-AUC.
4. **Calibrated threshold optimization:** calibration fitting and threshold selection were separated into disjoint validation patient groups; the final test partition remained untouched.
5. **SHAP explainability:** global and individual model explanations on a sample of the patient-level test data. SHAP values describe model associations, not causal effects.
6. **Subgroup performance:** descriptive performance by age, gender and recorded race on the original patient-level test partition. Small subgroup counts and positive counts require caution.

**Reproducibility note:** the current GitHub tree does not yet contain the six experiment source scripts and their full output bundle. The analyses and numerical summaries are recorded in the project handover context, but the experiment code and generated artefacts must be added and rerun from the project environment before claiming that this repository alone fully reproduces all six studies. See `reports/RESEARCH_AUDIT_SUMMARY.md` for the detailed summary and limitations.

## Setup (Mac / Windows)

Python 3.11 or 3.12 is recommended. From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train the primary model

```bash
python train_model.py
```

The training pipeline writes `models/healthcare_readmission_model.joblib` and primary patient-level reports in `reports/`. Training may take time because the dataset contains over 100,000 encounters.

## Run the dashboard

```bash
streamlit run app.py
```

The dashboard includes cohort overview, population insights, model performance, an academic risk-scoring demonstration and methodology/limitations. In the scoring demonstration, unedited fields are populated from historical dataset medians/modes; users must not interpret the resulting score as an actual patient's clinical risk.

## Model artifact compatibility

Scikit-learn does not guarantee that serialized models are portable across library versions. This repository pins `scikit-learn==1.5.1` to match the recorded model-generation environment. Install the pinned requirements before loading the supplied artifact. Re-training will regenerate the primary artifact and may change outputs if the data, code, dependency environment or randomization settings differ.

## Limitations and responsible use

- The data are historical (1999–2008), US-based and diabetes-focused.
- Transfer to current hospitals, other populations or Indian healthcare settings is unproven.
- The outcome is an encounter-level retrospective label, not a prospective clinical endpoint.
- There is no external, temporal or prospective validation.
- Calibration experiments are separate model-development analyses and do not replace the primary saved model's evaluation.
- Subgroup metrics are descriptive; they do not establish fairness, discrimination or statistical significance.
- SHAP and feature-importance values indicate model behaviour/association, not causation.
- False positives and false negatives are both substantial at the selected operating point.
- No clinical utility, decision-impact or deployment study has been performed.

Any future clinical research or deployment would require independent external validation, prospective evaluation, calibration assessment, adequate subgroup sample sizes, clinical governance, privacy/security review, monitoring and qualified human oversight.

## Citation

Dataset: Strack, B. et al. (2014). *Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records*. BioMed Research International. UCI Machine Learning Repository, Diabetes 130-US Hospitals for Years 1999–2008.
