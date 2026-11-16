# Model Card — CareSignal 30-Day Readmission Prototype

## Model summary

A retrospective binary classifier estimating whether a hospital encounter in the UCI Diabetes 130-US Hospitals dataset has the recorded outcome `readmitted == '<30'`. The primary saved artifact is `models/healthcare_readmission_model.joblib`.

## Intended use

- Academic demonstration of an end-to-end tabular machine-learning workflow.
- Retrospective exploration of historical encounter-level readmission patterns.
- Demonstration of patient-grouped evaluation, threshold trade-offs and model explainability.

## Out-of-scope use

- Diagnosis, treatment, triage, discharge, care allocation or clinical decision-making.
- Use with live patient data or integration into a clinical workflow.
- Claims that the score is a validated individual clinical probability.
- Claims of clinical readiness, fairness certification or clinical utility.

## Data

- Source: UCI Diabetes 130-US Hospitals for Years 1999–2008.
- Source cohort: 101,766 encounters and 71,518 unique patients.
- Unit of analysis: hospital encounter.
- Positive outcome: `readmitted == '<30'`; `NO` and `>30` are coded as negative.
- Patient and encounter identifiers are excluded from model inputs; patient identifier is used for grouped splitting.

## Primary model evaluation

The primary run uses patient-grouped train/validation/test partitions. Candidate models are compared using validation average precision (PR-AUC). The threshold is selected on validation data only, using maximum F1 subject to a recall floor of 0.60. The held-out test partition is used for final reporting.

| Metric | Test result |
|---|---:|
| Encounters | 20,153 |
| Patients | 14,304 |
| Train/test patient overlap | 0 |
| Features | 47 |
| Threshold | 0.481041 |
| Accuracy | 0.6220 |
| Precision | 0.1651 |
| Recall | 0.6267 |
| F1-score | 0.2614 |
| ROC-AUC | 0.6746 |
| PR-AUC (average precision) | 0.2085 |

Confusion matrix: TN=11,187; FP=6,815; FN=803; TP=1,348. Test positive prevalence is approximately 10.67%.

The operating point has modest precision and a substantial false-positive count. The recall floor is an exploratory modelling choice, not a clinically validated requirement.

## Additional research analyses

Separate analyses have examined encounter-random versus patient-grouped splitting, probability calibration, patient-cluster bootstrap confidence intervals, calibrated threshold selection, SHAP explanations and descriptive subgroup performance. These analyses must not be interpreted as one single re-trained or calibrated production model. Calibration studies used a separate training/calibration design; the subgroup and SHAP analyses refer to the original patient-level test partition.

The six experiment source scripts and complete generated output bundle are not currently present in the GitHub tree. The detailed numerical summary is documented in `reports/RESEARCH_AUDIT_SUMMARY.md`; full repository-level reproducibility remains pending until scripts and artefacts are committed and rerun.

## Input and demonstration caveat

The Streamlit risk-scoring form allows selected fields to be edited while remaining fields are filled with typical values (medians/modes) from the historical dataset. This is a synthetic/hybrid academic demonstration row, not a complete observed patient record. Its output is a model score, not an actual patient's clinical probability.

## Known limitations and risks

- Historical, US-based, diabetes-focused data from 1999–2008 may not represent current practice.
- External, temporal and prospective validation have not been performed.
- No clinical utility or decision-impact study has been performed.
- Calibration results from separate experiments do not constitute calibration certification of the primary saved model.
- Subgroup metrics are descriptive, with small subgroup sample sizes and positive counts; they do not establish fairness, bias or discrimination.
- SHAP values describe contributions to model output and do not establish causal effects.
- False positives may prompt unnecessary attention; false negatives may miss encounters with the recorded outcome.
- Historical records may reflect healthcare access, coding, workflow and measurement patterns.

## Responsible-use requirements

Do not use this model for patient care. Any future clinical investigation would require data governance and privacy review, domain-expert review, external validation on an appropriate contemporary cohort, prospective evaluation, calibration and subgroup uncertainty analysis, clinical utility assessment, monitoring and qualified human oversight.

## Reproducibility

The recorded primary environment pins `scikit-learn==1.5.1`. See `requirements.txt`, `train_model.py` and `reports/training_metadata.json`. Serialized scikit-learn artifacts may not be portable across library versions. Preserve the supplied primary artifact and its metadata together.
