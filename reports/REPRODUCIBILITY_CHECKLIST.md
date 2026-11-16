# Reproducibility and Release Checklist

## Data and environment
- [ ] Confirm the UCI source data provenance and exact downloaded file version.
- [ ] Record SHA-256 hashes for raw and cleaned data files.
- [ ] Record Python version and complete installed package versions (`python --version`, `pip freeze`).
- [ ] Install from `requirements.txt` in a fresh virtual environment.

## Primary model
- [ ] Confirm `models/healthcare_readmission_model.joblib` matches `reports/training_metadata.json`.
- [ ] Confirm the 47-feature schema and target definition.
- [ ] Confirm patient IDs do not cross train/validation/test partitions.
- [ ] Confirm threshold selection uses validation data only.
- [ ] Confirm test metrics match the committed metadata and CSV.
- [ ] Never overwrite the primary artifact while running separate experiments.

## Six research experiments
- [ ] Restore the six original experiment scripts from the working project directory.
- [ ] Add the exact generated CSV, JSON and plot outputs.
- [ ] Execute all scripts in a clean environment.
- [ ] Compare outputs to `reports/RESEARCH_AUDIT_SUMMARY.md`.
- [ ] Document any differences instead of silently replacing previously reported values.
- [ ] Confirm patient-cluster bootstrap resamples patients, not independent encounters.
- [ ] Confirm calibration fit, threshold selection and final test are separated by patient.
- [ ] Confirm SHAP sample, background data and output scale are documented.
- [ ] Report subgroup counts and uncertainty; do not claim fairness certification.

## Dashboard and responsible-use checks
- [ ] Keep the dashboard academic-only and visibly label the risk score as a model demonstration.
- [ ] Explain that unedited inputs are filled with historical medians/modes.
- [ ] Do not call the score an individual clinical probability.
- [ ] Preserve the original dashboard and primary model artifact.
- [ ] Verify dashboard loads the intended primary artifact.

## Release
- [ ] Inspect branch diff and ensure no secrets, personal data or unintended files are included.
- [ ] Confirm older model artifacts are retained and documented.
- [ ] Run training, experiment and dashboard smoke tests.
- [ ] Request faculty review before selecting a publication venue.
