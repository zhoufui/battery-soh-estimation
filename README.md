# Interpretable Offline Battery SOH Benchmark

This repository contains code, processed features, and results for the offline battery state-of-health (SOH) benchmark audited on 2026-10-14.

Associated study: **Interpretable Machine Learning for Offline Lithium-Ion Battery State-of-Health Estimation Using Public Aging Data**.

## Scope

Four NASA cells (B0005, B0006, B0007, B0018) and two CALCE cells (CS2_35, CS2_36) support cross-battery comparisons of five fixed regressors. NASA uses 40 indicators; CALCE and cross-dataset comparisons use 26 shared indicators. Complete-discharge inputs address offline estimation. They do not establish prospective online prediction.

The CALCE parser now identifies phases from current rather than positive cumulative capacity counters. The package also contains direct-integration diagnostics, duration/integral ablation, preceding-charge-only controls, and matched source-only, target-only, and pooled target-sample comparisons.

## Reproduce

The audited environment uses Python 3.13.5 on Windows. Versions and feature order are recorded in `results/tables/experiment_manifest_2026-10-14.json`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-audited.txt
.\run_audited_pipeline.ps1
```

The default entry point uses included processed features. To rebuild CALCE features, first obtain the original CS2_35 and CS2_36 Excel archives from CALCE under `data/raw/CALCE/`, then run `./run_audited_pipeline.ps1 -RebuildCalceFeatures`. Raw files are not redistributed. The individual Python scripts also run from this repository root on other operating systems; the locked environment itself was checked on Windows.

The entry point runs regression tests, model evaluations, matched controls, feature audits, figure generation, and output validation. `run_reproduce_paper.ps1` is a wrapper for the same entry point. To verify the included result tables without training models, run `python src/validate_audited_outputs.py` from the repository root.

## NASA leave one battery out

| Model | MAE | RMSE | MAPE (%) | R2 |
|---|---:|---:|---:|---:|
| Random Forest | 0.0315 | 0.0356 | 4.09 | 0.8592 |
| LightGBM | 0.0314 | 0.0357 | 4.07 | 0.8548 |
| XGBoost | 0.0323 | 0.0365 | 4.16 | 0.8493 |
| Gradient Boosting | 0.0323 | 0.0366 | 4.15 | 0.8478 |
| SVR-RBF | 0.0644 | 0.0745 | 8.26 | 0.3357 |

## Corrected CALCE leave one battery out

| Model | MAE | RMSE | MAPE (%) | R2 |
|---|---:|---:|---:|---:|
| Gradient Boosting | 0.0048 | 0.0051 | 0.62 | 0.9986 |
| Random Forest | 0.0049 | 0.0054 | 0.66 | 0.9984 |
| LightGBM | 0.0050 | 0.0057 | 0.68 | 0.9982 |
| XGBoost | 0.0062 | 0.0066 | 0.81 | 0.9976 |
| SVR-RBF | 0.0891 | 0.1118 | 12.31 | 0.3242 |

## Matched target sample controls

For the same Gradient Boosting model and identical test rows within each fraction:

| Labeled target fraction | Source only RMSE | Target only RMSE | Source plus target RMSE |
|---:|---:|---:|---:|
| 2% | 0.0957 | 0.0201 | 0.0193 |
| 5% | 0.0958 | 0.0105 | 0.0095 |
| 10% | 0.0955 | 0.0042 | 0.0065 |
| 20% | 0.0960 | 0.0025 | 0.0047 |

Target-only RMSE is lower at 10% and 20%. All five-model results, split roles, and 90225 prediction rows are retained. Selection covers the whole trajectory, including its endpoint, and tests interpolation within the same target cells. Different fractions have different test sets. The reported minima are descriptive comparisons, with no nested model-selection set.

## Capacity proxy and limitations

Full-discharge current integrals correlate above 0.9997 with capacity in every cell. A direct integration diagnostic already gives low error using the same offline capacity denominator. CALCE labels are normalized by maximum retained capacity over the complete trajectory, while NASA uses first recorded capacity. CALCE capacity screening also uses the complete trajectory. Neither convention should be interpreted as a validated prospective protocol.

The benchmark has six independent cells, not thousands of independent batteries. SHAP explains model associations, not degradation mechanisms. The early linear calibration and older lightweight ablation are auxiliary outputs and should not be confused with the matched controls.

## Data and citation

Processed features are included under `data/processed/`. See [data sources](DATA_SOURCES.md) for the original datasets, rebuilding instructions, and CALCE contributor references. Repository citation metadata are in `CITATION.cff`.

The analysis code is licensed under the MIT License. Data remain subject to their original providers' terms.
