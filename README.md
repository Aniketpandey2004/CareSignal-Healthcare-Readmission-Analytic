# CareSignal — Healthcare Readmission Intelligence

An academic machine-learning and analytics prototype for exploring historical 30-day hospital readmission patterns among diabetic patients.

> **Important:** This project is not a clinical diagnostic or treatment tool. It has not been prospectively or externally validated and must not be used to make patient-care decisions.

## What is included

- Encounter-level analytics dashboard built with Streamlit and Plotly.
- Patient-grouped model training pipeline (`train_model.py`).
- Notebook launcher (`notebooks/04_modeling.ipynb`).
- Validation-based model comparison and decision-threshold selection.
- Held-out patient-level evaluation reports.
- Optional individual encounter scoring demonstration after the model has been trained.

## Dataset

The project uses the UCI **Diabetes 130-US Hospitals for Years 1999–2008** dataset. It contains 101,766 hospital encounters and 71,518 unique patients in the source data. A row is an encounter, not a unique patient.

Target definition:

- `readmitted == '<30'` → `readmitted_30 = 1`
- `readmitted == 'NO'` or `'>30'` → `readmitted_30 = 0`

The patient and encounter identifiers are excluded from model inputs. Patient identifiers are used only to group encounters during splitting.

## Project structure

```text
healthcare-readmission-analytics/
├── app.py
├── train_model.py
├── requirements.txt
├── data/
│   ├── cleaned_healthcare_data.csv
│   └── raw/diabetic_data_original.csv
├── models/
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_model_evaluation.ipynb
└── reports/
```

## Setup (Mac / Windows)

Python 3.11 or 3.12 is recommended. From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train the model

Option A — terminal:

```bash
python train_model.py
```

Option B — VS Code:

1. Open `notebooks/04_modeling.ipynb`.
2. Select the Python environment where project requirements were installed.
3. Click **Run All**.

The training pipeline writes `models/healthcare_readmission_model.joblib` and the patient-level reports in `reports/`.

## Run the dashboard

```bash
streamlit run app.py
```

The dashboard has five sections:

1. Executive Overview — cohort KPIs and outcome distribution.
2. Population Insights — descriptive comparisons by recorded demographics and utilization.
3. Model Performance — validation comparison, holdout metrics, confusion matrix and feature importance.
4. Risk Scoring Demo — a form that uses the saved model artifact and full expected feature schema.
5. Methodology — target definition, split strategy, limitations and responsible-use notes.

## Evaluation approach

- A patient-grouped train/test split prevents encounters from the same patient crossing the train/test boundary.
- A second grouped split within training data creates a validation partition.
- Preprocessing is fitted within a scikit-learn pipeline to reduce preprocessing leakage.
- Candidate models are compared on validation average precision (PR-AUC).
- The operating threshold is selected using validation data only. A 60% recall floor is an exploratory project rule, not a clinically established requirement.
- The final test partition is evaluated once after model selection.

Reported metrics include precision, recall, F1, ROC-AUC and PR-AUC. Because the positive class is relatively uncommon, accuracy alone is not an adequate summary.

## Model artifact compatibility

Scikit-learn does not guarantee that serialized models are portable across library versions. This repository pins `scikit-learn==1.5.1` to match the supplied model-generation environment. If a model-load error occurs, install the requirements and rerun `train_model.py` to recreate the artifact in that environment.

## Limitations and ethics

- Historical data (1999–2008) may not represent modern clinical practice.
- The cohort is US-based and diabetes-focused; transfer to Indian hospitals or other populations is unproven.
- This is encounter-level prediction, not a longitudinal patient-level clinical assessment.
- No external validation, prospective validation, calibration study, fairness audit or clinical utility study is claimed.
- Model feature importance is predictive association, not causal explanation.
- False positives and false negatives both have operational consequences.
- Any real-world use would require clinical governance, privacy/security review, external validation, prospective evaluation, monitoring and human oversight.

## Suggested research direction

A defensible study can investigate how patient-grouped evaluation changes estimated performance compared with encounter-random evaluation, and how threshold selection changes precision/recall trade-offs. A literature review is needed before claiming novelty.

## Citation

Dataset: Strack, B. et al. (2014). *Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records*. BioMed Research International. UCI Machine Learning Repository, Diabetes 130-US Hospitals for Years 1999–2008.
