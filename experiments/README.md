# Experiment Source Files — Reproducibility Status

The six analyses below were completed during project work and their numerical summaries are documented in [../reports/RESEARCH_AUDIT_SUMMARY.md](../reports/RESEARCH_AUDIT_SUMMARY.md).

**Important:** The source scripts and full output files are not yet committed to this repository. Do not treat this directory as a runnable experiment suite until the original scripts have been copied from the working project folder, reviewed and executed in a clean environment.

## Expected source scripts

| Study | Expected script |
|---|---|
| 1. Patient-grouped vs random encounter split | `split_strategy_comparison.py` |
| 2. Calibration comparison | `calibration_improvement.py` |
| 3. Patient-cluster bootstrap confidence intervals | `bootstrap_confidence_intervals.py` |
| 4. Calibrated threshold optimization | `calibrated_threshold_optimization.py` |
| 5. SHAP explainability | `shap_explainability_analysis.py` |
| 6. Descriptive subgroup analysis | `subgroup_fairness_analysis.py` |

## Expected output families

The original working run reported outputs under `reports/`, including:

- Split strategy comparison CSV.
- Calibration method comparison CSV, plots and JSON summary.
- Bootstrap confidence interval JSON/CSV summary.
- Calibrated threshold comparison CSV, plots and JSON summary.
- SHAP global importance, summary/bar/beeswarm/waterfall plots, patient explanation CSV and JSON summary.
- Subgroup metrics, overall metrics, metric gaps, JSON summary and age/gender/race plots.

Use the exact original script output names when restoring these files; do not create empty placeholder results or substitute hand-entered values for generated results.

## Safe execution protocol

1. Keep the primary model at `models/healthcare_readmission_model.joblib` unchanged.
2. Run each experiment from a clean checkout of this branch using its documented dependencies.
3. Confirm every experiment reads the intended raw data and reproduces the stated patient split.
4. Save experiment-specific models, if any, under a separate temporary or experiment-specific path—not over the primary model artifact.
5. Compare generated CSV/JSON values with the research summary before updating the manuscript or README.
6. Record Python version, package versions, random seeds, input data hash and command used.
7. Do not describe the subgroup analysis as fairness certification or the SHAP analysis as causal explanation.
