# Model Card — 30-Day Readmission Prototype

## Summary

A retrospective binary classifier estimating whether an encounter in the UCI Diabetes 130-US Hospitals dataset has the recorded outcome `readmitted == '<30'`.

## Intended use

- Academic demonstration of an end-to-end tabular ML workflow.
- Retrospective exploration of readmission-related patterns.
- Demonstration of patient-grouped evaluation and threshold trade-offs.

## Out-of-scope use

- Diagnosis, treatment, discharge, triage or care-allocation decisions.
- Use with live patient data or in a clinical workflow.
- Claims of validated individual clinical risk.

## Data

- Source: UCI Diabetes 130-US Hospitals for Years 1999–2008.
- Unit: hospital encounter.
- Outcome: `<30` versus `NO` / `>30`.
- Patient and encounter identifiers are excluded from predictive features; patient identifier is used only for grouped splitting.

## Evaluation

The pipeline uses a grouped train/validation/test split by patient. Candidate models are compared on validation PR-AUC. The decision threshold is selected on validation data. The holdout test partition is used for final reporting. Re-run `train_model.py` to regenerate the exact current metrics and split metadata.

## Known limitations

- Historical, US-based, diabetes-focused data.
- No external, prospective or temporal validation.
- No clinical utility or decision-impact study.
- No calibration or fairness certification is claimed.
- Feature importance is not causal explanation.
- A patient may have multiple encounters; this project predicts an encounter-level outcome.

## Risks

False positives can lead to unnecessary review; false negatives can miss encounters that later have a recorded readmission. Historical data may encode inequities or care-process artifacts. Any future research or deployment must include privacy controls, domain review, external validation, calibration, subgroup performance analysis, clinical governance and human oversight.
