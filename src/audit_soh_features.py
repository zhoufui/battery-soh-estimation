"""Audit full-cycle capacity proxies and remove duration/integral features.

This is an offline sensitivity analysis. Removing these columns does not make
the remaining full-curve summaries available before a discharge finishes.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from representative_domain_adaptation import get_models, numeric_cols
from train_baseline import feature_columns, metrics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "tables"
if (ROOT / ".python-packages").exists():
    sys.path.insert(0, str(ROOT / ".python-packages"))


def main():
    frames = {
        name: pd.read_csv(ROOT / f"data/processed/{name.lower()}_soh_features.csv")
        .replace([np.inf, -np.inf], np.nan).dropna(subset=["soh"])
        for name in ["NASA", "CALCE"]
    }
    common = sorted(numeric_cols(frames["NASA"]) & numeric_cols(frames["CALCE"]))
    proxy_rows, ablation_rows, cell_rows, contracts = [], [], [], {}
    for dataset, frame in frames.items():
        # Preserve the original NASA column order: tied tree splits and random
        # feature sampling can depend on order even with the same random seed.
        cols = feature_columns(frame) if dataset == "NASA" else common
        groups = {
            "full_features": cols,
            "without_duration_or_integrals": [c for c in cols if "duration" not in c and not c.endswith("_time_area")],
            "preceding_charge_only": [c for c in cols if c.startswith("charge_")],
        }
        contracts[dataset] = groups
        for battery, group in frame.groupby("battery_id"):
            reference = float((group.capacity_Ah / group.soh).median())
            # Reference is the same known, offline label denominator as in the
            # manuscript. No model or per-cycle fitted correction is used.
            soh_from_integral = group.current_time_area.abs().to_numpy() / 3600. / reference
            proxy_rows.append({"dataset": dataset, "battery_id": battery, "n_samples": len(group),
                               "reference_capacity_Ah": reference,
                               "capacity_integral_correlation": group.capacity_Ah.corr(group.current_time_area.abs()),
                               **metrics(group.soh.to_numpy(), soh_from_integral)})
            cell_rows.append({"dataset": dataset, "battery_id": battery, "n_samples": len(group),
                              "reference_capacity_Ah": reference,
                              "first_valid_capacity_Ah": float(group.capacity_Ah.iloc[0]),
                              "maximum_capacity_Ah": float(group.capacity_Ah.max()),
                              "max_capacity_position": int(np.argmax(group.capacity_Ah.to_numpy())) + 1,
                              "identical_charge_discharge_duration": int((group.duration_s == group.charge_duration_s).sum()),
                              "positive_discharge_current_mean": int((group.current_mean >= 0).sum()),
                              "negative_charge_current_mean": int((group.charge_current_mean <= 0).sum())})
        for held_out in sorted(frame.battery_id.unique()):
            train = frame[frame.battery_id != held_out]
            test = frame[frame.battery_id == held_out]
            for group_name, selected in groups.items():
                for model_name, model in get_models().items():
                    print(f"{dataset} {held_out} {group_name} {model_name}", flush=True)
                    model.fit(train[selected], train.soh)
                    pred = model.predict(test[selected])
                    ablation_rows.append({"dataset": dataset, "held_out": held_out,
                                          "feature_group": group_name, "n_features": len(selected),
                                          "model": model_name, "n_train": len(train), "n_test": len(test),
                                          **metrics(test.soh.to_numpy(), pred)})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(proxy_rows).to_csv(OUT / "capacity_proxy_metrics.csv", index=False)
    pd.DataFrame(cell_rows).to_csv(OUT / "preprocessing_audit.csv", index=False)
    ablation = pd.DataFrame(ablation_rows)
    ablation.to_csv(OUT / "feature_proxy_ablation_metrics.csv", index=False)
    summary = ablation.groupby(["dataset", "feature_group", "n_features", "model"], as_index=False)[
        ["MAE", "RMSE", "MAPE_pct", "R2"]].mean()
    summary.to_csv(OUT / "feature_proxy_ablation_summary.csv", index=False)
    versions = {}
    for package in ["numpy", "pandas", "scipy", "scikit-learn", "xgboost", "lightgbm", "openpyxl", "shap", "Pillow"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    paths = [ROOT / f"data/processed/{n.lower()}_soh_features.csv" for n in frames]
    paths += [ROOT / "src" / name for name in ["prepare_calce_soh.py", "train_standard_models.py",
              "calce_validation.py", "representative_domain_adaptation.py", "audit_soh_features.py"]]
    manifest = {"python": sys.version, "platform": platform.platform(), "packages": versions,
                "seed": 42, "common_features": common, "feature_groups": contracts,
                "sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                "scope": "Offline full-cycle analysis; fixed hyperparameters; unweighted cell means; no causal claims"}
    (OUT / "experiment_manifest_2026-10-14.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.sort_values(["dataset", "feature_group", "RMSE"]).to_string(index=False))


if __name__ == "__main__":
    main()
