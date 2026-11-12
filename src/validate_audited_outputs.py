"""Check phase sanity, split integrity, and model output metrics."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from train_baseline import metrics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/tables"


def main():
    calce = pd.read_csv(ROOT / "data/processed/calce_soh_features.csv")
    key = ["battery_id", "source_file", "source_cycle_index"]
    assert not calce.duplicated(key).any()
    assert len(calce) == 1657
    assert (calce.current_mean < -.01).all()
    assert (calce.charge_current_mean > .01).all()
    assert not (calce.duration_s == calce.charge_duration_s).any()

    metric_cols = ["MAE", "RMSE", "MAPE_pct", "R2"]
    audit = pd.read_csv(OUT / "feature_proxy_ablation_summary.csv")
    for dataset, filename in [("NASA", "standard_cross_battery_summary.csv"), ("CALCE", "calce_internal_summary.csv")]:
        full = audit[(audit.dataset == dataset) & (audit.feature_group == "full_features")].set_index("model").sort_index()
        base = pd.read_csv(OUT / filename).set_index("model").sort_index()
        np.testing.assert_allclose(full[metric_cols], base[metric_cols], rtol=1e-10, atol=1e-12)

    split = pd.read_csv(OUT / "adaptation_split_manifest.csv")
    for fraction, group in split.groupby("target_fraction"):
        assert len(group) == len(calce)
        assert not group.duplicated(key).any()
        assert set(group.role) == {"calibration", "test"}
        assert set(map(tuple, group[key].to_numpy())) == set(map(tuple, calce[key].to_numpy()))
    predictions = pd.read_csv(OUT / "adaptation_controls_predictions.csv")
    controls = pd.read_csv(OUT / "adaptation_controls_metrics.csv")
    group_keys = ["target_fraction", "training_mode", "model"]
    prediction_groups = predictions.groupby(group_keys)
    assert len(controls) == len(prediction_groups) == 60
    assert not controls.duplicated(group_keys).any()
    assert set(prediction_groups.groups) == set(map(tuple, controls[group_keys].to_numpy()))
    for (fraction, mode, model), pred in prediction_groups:
        test = split[(split.target_fraction == fraction) & (split.role == "test")]
        assert not pred.duplicated(key).any()
        assert set(map(tuple, pred[key].to_numpy())) == set(map(tuple, test[key].to_numpy()))
        label_check = pred.merge(calce[key + ["soh"]], on=key, validate="one_to_one", suffixes=("_pred", "_data"))
        np.testing.assert_allclose(label_check.soh_pred, label_check.soh_data, atol=1e-12)
        row = controls[(controls.target_fraction == fraction) & (controls.training_mode == mode) & (controls.model == model)].iloc[0]
        recalculated = metrics(pred.soh.to_numpy(), pred.y_pred.to_numpy())
        assert len(pred) == row.test_samples
        np.testing.assert_allclose([recalculated[c] for c in metric_cols], row[metric_cols].to_numpy(dtype=float), rtol=1e-10, atol=1e-12)
    pooled = pd.read_csv(OUT / "representative_domain_adaptation_metrics.csv").set_index(["target_fraction", "model"]).sort_index()
    controls_pooled = controls[controls.training_mode == "source_plus_target"].set_index(["target_fraction", "model"]).sort_index()
    np.testing.assert_allclose(pooled[metric_cols], controls_pooled[metric_cols], atol=1e-12)

    result = {"calce_cycles": len(calce), "control_metric_rows": len(controls),
              "prediction_rows": len(predictions), "independently_recomputed_metric_groups": len(prediction_groups),
              "checks": "phase signs; unique cycle keys; split coverage and separation; shared test rows; prediction labels; recomputed metrics; full-feature benchmark agreement"}
    (OUT / "audit_validation_2026-10-14.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
