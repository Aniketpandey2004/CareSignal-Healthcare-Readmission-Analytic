# Research Audit Summary — CareSignal

**Status:** Numerical summary of six analyses completed during project work. This file is a reporting aid, not a substitute for experiment source code or raw outputs. The six experiment scripts and complete output bundle are not yet present in this GitHub branch. Re-run and reconcile them from the project environment before treating this repository as independently reproducible.

## Study dataset and primary model

- Dataset: UCI Diabetes 130-US Hospitals for Years 1999–2008.
- Source cohort: 101,766 encounters, 71,518 unique patients, 130 hospitals.
- Target: `readmitted == '<30'` coded as 1; `NO` and `>30` coded as 0.
- Primary model: Random Forest pipeline, 47 predictive features.
- Patient/encounter identifiers are excluded from predictors.
- Primary split: patient-grouped outer split (`test_size=0.20`, `random_state=42`); grouped validation split within training (`test_size=0.20`, `random_state=21`).
- Primary threshold: 0.48104127223505294, selected on validation data by maximum F1 subject to recall >= 0.60.
- Primary test: 20,153 encounters / 14,304 patients; train/test patient overlap = 0.

### Primary held-out test results

| Metric | Result |
|---|---:|
| Accuracy | 0.6219918 |
| Precision | 0.1651354 |
| Recall | 0.6266853 |
| F1 | 0.2613923 |
| ROC-AUC | 0.6745589 |
| PR-AUC / average precision | 0.2085380 |

Confusion matrix: TN=11,187; FP=6,815; FN=803; TP=1,348. Test prevalence = 10.673%.

Validation model comparison at threshold 0.50:

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.184168 | 0.525346 | 0.272727 | 0.672936 | 0.205303 |
| Logistic Regression | 0.161732 | 0.538018 | 0.248702 | 0.655025 | 0.198430 |
| Extra Trees | 0.161290 | 0.550115 | 0.249445 | 0.648642 | 0.192945 |

## Experiment 1 — Patient-grouped versus random encounter split

| Measure | Patient-grouped | Random encounter |
|---|---:|---:|
| Train encounters | 81,613 | 81,412 |
| Test encounters | 20,153 | 20,354 |
| Train patients | 57,214 | 60,132 |
| Test patients | 14,304 | 18,387 |
| Shared patients | 0 | 7,001 |
| Test positive rate | 0.1067 | 0.1116 |
| Accuracy | 0.6220 | 0.6299 |
| Precision | 0.1651 | 0.1800 |
| Recall | 0.6267 | 0.6517 |
| F1 | 0.2614 | 0.2821 |
| ROC-AUC | 0.6746 | 0.6889 |
| PR-AUC | 0.2085 | 0.2293 |

Interpretation: the random encounter split has patient overlap (7,001 people represented on both sides). Its higher metrics should not be used as the primary estimate of performance on unseen patients. The comparison demonstrates sensitivity to split design; it does not prove that overlap alone caused the entire metric difference.

## Experiment 2 — Probability calibration comparison

A fresh model was fit on the subtraining partition; calibration methods were fit using the validation partition; test patients were held out. This is a separate calibration experiment, not the primary artifact trained on the full outer training set.

Partition sizes: subtrain 65,358 encounters / 45,771 patients; validation 16,255 / 11,443; test 20,153 / 14,304.

| Method | Brier | Log loss | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|
| Uncalibrated | 0.214050 | 0.619339 | 0.673825 | 0.207299 |
| Sigmoid | 0.091360 | 0.320651 | 0.673825 | 0.207299 |
| Isotonic | 0.091428 | 0.321946 | 0.672961 | 0.195711 |

At the original uncalibrated threshold (0.481041), the calibrated scores did not yield a comparable operating point: sigmoid produced no positive predictions at that cutoff, while isotonic had precision 0.6667 and recall 0.00093. This illustrates that calibration changes score scale and the threshold must be selected again on validation data.

## Experiment 3 — Patient-cluster bootstrap confidence intervals

1,000 valid patient-cluster bootstrap replicates were used to account for repeated encounters within patients.

| Metric | Estimate | 95% interval |
|---|---:|---:|
| ROC-AUC | 0.6746 | 0.6603–0.6864 |
| PR-AUC | 0.2085 | 0.1902–0.2271 |

These intervals quantify resampling uncertainty under this test sample and bootstrap procedure; they do not represent external validation uncertainty.

## Experiment 4 — Calibrated threshold optimization

The validation partition was split into disjoint calibration-fit and threshold-validation patient groups. Calibration fitting and threshold choice therefore did not use the final test partition.

Partitions: subtrain 65,358 encounters / 45,771 patients; calibration-fit 8,198 / 5,721; threshold-validation 8,057 / 5,722; test 20,153 / 14,304.

Validation-selected thresholds:

| Method | Threshold | Validation precision | Validation recall | Validation F1 |
|---|---:|---:|---:|---:|
| Uncalibrated | 0.481044 | 0.1652 | 0.6083 | 0.2599 |
| Sigmoid | 0.112959 | 0.1652 | 0.6083 | 0.2599 |
| Isotonic | 0.108714 | 0.1487 | 0.6798 | 0.2441 |

Untouched test results:

| Method | Brier | Log loss | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Uncalibrated | 0.214050 | 0.619339 | 0.164630 | 0.607624 | 0.259068 | 0.673825 | 0.207299 |
| Sigmoid | 0.091356 | 0.320656 | 0.164630 | 0.607624 | 0.259068 | 0.673825 | 0.207299 |
| Isotonic | 0.091367 | 0.321834 | 0.156027 | 0.701069 | 0.255247 | 0.672686 | 0.194469 |

Sigmoid reduced Brier score by approximately 57.3% and log loss by approximately 48.2% relative to the uncalibrated scores in this experiment; after validation threshold re-selection, its classification metrics were unchanged. Isotonic increased test recall to about 70.1%, with lower precision and PR-AUC. These are experiment-specific findings and do not establish that the primary saved model is calibrated.

## Experiment 5 — SHAP explainability

SHAP analysis used the original saved pipeline and patient-level test partition, with a reproducible sample of 500. Top features by mean absolute SHAP value:

| Rank | Encoded feature | Mean absolute SHAP |
|---:|---|---:|
| 1 | numeric__number_inpatient | 0.025066 |
| 2 | categorical__discharge_disposition_id_1 | 0.017200 |
| 3 | numeric__number_diagnoses | 0.007662 |
| 4 | numeric__number_emergency | 0.007060 |
| 5 | numeric__num_medications | 0.006446 |
| 6 | categorical__discharge_disposition_id_3 | 0.006042 |
| 7 | numeric__time_in_hospital | 0.006039 |
| 8 | categorical__insulin_No | 0.005609 |
| 9 | categorical__diabetesMed_No | 0.004267 |
| 10 | categorical__discharge_disposition_id_22 | 0.004087 |

Mean absolute SHAP describes magnitude of contribution, not direction or causality. Discharge disposition and length-of-stay-related features are only available at/near discharge, so the model must not be represented as an admission-time prediction system. The declared project purpose is academic demonstration.

## Experiment 6 — Descriptive subgroup performance

Original patient-level test partition; minimum subgroup size 100 encounters. Metrics below are descriptive and are not a formal fairness certification or statistical test.

### Age

| Age group | N | Positives | Recall | Precision | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| [80-90) | 3,314 | 403 | 0.6973 | 0.1536 | 0.2518 | 0.6345 | 0.2041 |
| [70-80) | 5,125 | 560 | 0.6875 | 0.1614 | 0.2615 | 0.6734 | 0.2136 |
| [60-70) | 4,511 | 465 | 0.6387 | 0.1700 | 0.2685 | 0.6791 | 0.2164 |
| [50-60) | 3,518 | 356 | 0.5028 | 0.1946 | 0.2806 | 0.6990 | 0.2265 |
| [40-50) | 1,893 | 193 | 0.5855 | 0.1883 | 0.2850 | 0.7022 | 0.2474 |
| [30-40) | 804 | 99 | 0.5354 | 0.1893 | 0.2797 | 0.6823 | 0.2639 |
| [90-100) | 528 | 45 | 0.6000 | 0.0931 | 0.1612 | 0.5545 | 0.1087 |
| [20-30) | 285 | 23 | 0.5217 | 0.1333 | 0.2124 | 0.6371 | 0.1250 |
| [10-20) | 137 | 5 | 0.2000 | 0.0455 | 0.0741 | 0.7621 | 0.2707 |

[0-10) was excluded because N=38. The [10-20) group has only five positive cases; its metrics are highly unstable.

### Gender

| Group | N | Positives | Recall | Precision | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Male | 9,280 | 994 | 0.6006 | 0.1697 | 0.2646 | 0.6797 |
| Female | 10,871 | 1,157 | 0.6491 | 0.1617 | 0.2590 | 0.6705 |

Unknown/invalid gender had N=2 and was excluded.

### Recorded race

| Group | N | Positives | Recall | Precision | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Caucasian | 15,089 | 1,626 | 0.6248 | 0.1629 | 0.2584 | 0.6722 |
| AfricanAmerican | 3,847 | 425 | 0.6518 | 0.1778 | 0.2794 | 0.6705 |
| Other | 289 | 29 | 0.4483 | 0.1548 | 0.2301 | 0.6973 |
| Unknown/Not recorded | 410 | 21 | 0.5238 | 0.0853 | 0.1467 | 0.6752 |
| Hispanic | 393 | 31 | 0.7419 | 0.1870 | 0.2987 | 0.7750 |
| Asian | 125 | 19 | 0.4211 | 0.2500 | 0.3137 | 0.6852 |

The recorded race subgroup counts and positive counts vary substantially. Hispanic (31 positives) and Asian (19 positives) estimates are especially uncertain. Observed recall ranges across groups are descriptive gaps, not proof of discrimination or fairness.

## Cross-study interpretation and cautions

1. The patient-grouped split is the preferred primary estimate for generalization to patients not represented in training.
2. Precision is low at the selected operating point; the model should not be described as a high-confidence clinical screening system.
3. Calibration experiments use a separate fitting/validation design and should be reported separately from the primary model.
4. SHAP results are model-specific associations and do not establish clinical causation.
5. Subgroup metrics require uncertainty intervals, sufficient positive cases and domain review before stronger claims.
6. The cohort is historical and US-specific. No external, temporal or prospective validation has been performed.
7. All results are for academic analysis only; no patient-care use is supported.

## Repository reproducibility status

The current repository has the primary training code, saved artifact, metadata and older reports. The six experiment scripts and their complete result files have not yet been added to this repository. Before publication or independent reproduction:

- Add the exact experiment source scripts from the working project directory.
- Add the raw result CSV/JSON files and plots generated by those scripts.
- Record Python and package versions, data file hashes, script commands and random seeds.
- Re-run each experiment in a clean environment.
- Verify every reported number against the generated outputs.
- Preserve the primary model artifact and do not overwrite it during research experiments.
