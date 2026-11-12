from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone

from train_baseline import TARGET, metrics
from train_standard_models import build_models


EXCLUDE = {
    "battery_id",
    "source_file",
    "source_date",
    "source_cycle_index",
    "cycle_index",
    "raw_cycle_index",
    "discharge_cycle_index",
    "capacity_Ah",
    "capacity_retention_pct",
    TARGET,
}


def numeric_cols(df: pd.DataFrame) -> set[str]:
    return {c for c in df.columns if c not in EXCLUDE and pd.api.types.is_numeric_dtype(df[c])}


def get_models() -> dict[str, Any]:
    return {name: model for name, model in build_models().items() if name != "MLP"}


def representative_split(calce: pd.DataFrame, fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < fraction < 1:
        raise ValueError("target fraction must be between zero and one")
    cal_parts = []
    test_parts = []
    for _, group in calce.groupby("battery_id"):
        group = group.sort_values("cycle_index").reset_index(drop=False)
        n = len(group)
        if n < 8:
            raise ValueError("each target cell needs at least 8 samples")
        cal_n = max(3, int(round(n * fraction)))
        cal_n = min(cal_n, n - 5)
        # Evenly spaced target samples cover the degradation range.
        positions = np.linspace(0, n - 1, cal_n).round().astype(int)
        positions = sorted(set(int(x) for x in positions))
        mask = np.zeros(n, dtype=bool)
        mask[positions] = True
        cal_parts.append(group.loc[mask])
        test_parts.append(group.loc[~mask])
    cal = pd.concat(cal_parts, ignore_index=True).drop(columns=["index"], errors="ignore")
    test = pd.concat(test_parts, ignore_index=True).drop(columns=["index"], errors="ignore")
    return cal, test


def save_adaptation_svg(summary: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 960, 560
    pad_l, pad_r, pad_t, pad_b = 78, 35, 58, 70
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    x_values = sorted(summary["target_fraction"].unique())
    y_max = float(summary["RMSE"].max()) * 1.1
    y_min = max(0.0, float(summary["RMSE"].min()) * 0.8)
    colors = {
        "SVR_RBF": "#7c3aed",
        "RandomForest": "#b91c1c",
        "GradientBoosting": "#a16207",
        "XGBoost": "#2563eb",
        "LightGBM": "#0f766e",
        "Source only": "#6b7280",
        "Target only": "#2563eb",
        "Source + target": "#b91c1c",
    }

    def sx(x: float) -> float:
        return pad_l + (x - min(x_values)) / max(max(x_values) - min(x_values), 1e-12) * plot_w

    def sy(y: float) -> float:
        return pad_t + (y_max - y) / max(y_max - y_min, 1e-12) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">Representative Target-Domain Adaptation</text>',
        f'<line x1="{pad_l}" y1="{height-pad_b}" x2="{width-pad_r}" y2="{height-pad_b}" stroke="#111827" stroke-width="1.2"/>',
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{height-pad_b}" stroke="#111827" stroke-width="1.2"/>',
    ]
    for i in range(6):
        y = y_min + (y_max - y_min) * i / 5
        yy = sy(y)
        parts.append(f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{width-pad_r}" y2="{yy:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{pad_l-10}" y="{yy+4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{y:.3f}</text>')
    for x in x_values:
        xx = sx(float(x))
        parts.append(f'<line x1="{xx:.1f}" y1="{height-pad_b}" x2="{xx:.1f}" y2="{height-pad_b+6}" stroke="#111827"/>')
        parts.append(f'<text x="{xx:.1f}" y="{height-pad_b+24}" text-anchor="middle" font-family="Arial" font-size="12">{int(round(x*100))}%</text>')

    legend_y = pad_t + 10
    for idx, (model, group) in enumerate(summary.groupby("model")):
        group = group.sort_values("target_fraction")
        color = colors.get(model, "#334155")
        points = " ".join(
            f"{sx(float(row['target_fraction'])):.1f},{sy(float(row['RMSE'])):.1f}"
            for _, row in group.iterrows()
        )
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.3"/>')
        for _, row in group.iterrows():
            parts.append(
                f'<circle cx="{sx(float(row["target_fraction"])):.1f}" cy="{sy(float(row["RMSE"])):.1f}" r="3.5" fill="{color}"/>'
            )
        parts.append(f'<rect x="{pad_l+15+idx*165}" y="{legend_y}" width="13" height="13" fill="{color}"/>')
        parts.append(f'<text x="{pad_l+34+idx*165}" y="{legend_y+12}" font-family="Arial" font-size="12">{model}</text>')

    parts.append(f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-family="Arial" font-size="14">Representative target samples used for adaptation</text>')
    parts.append(f'<text x="24" y="{height/2}" transform="rotate(-90 24,{height/2})" text-anchor="middle" font-family="Arial" font-size="14">RMSE</text>')
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    nasa_path = Path("data/processed/nasa_soh_features.csv")
    calce_path = Path("data/processed/calce_soh_features.csv")
    if not nasa_path.exists() or not calce_path.exists():
        print("Missing NASA or CALCE feature table.")
        return 1

    nasa = pd.read_csv(nasa_path).replace([np.inf, -np.inf], np.nan).dropna(subset=[TARGET])
    calce = pd.read_csv(calce_path).replace([np.inf, -np.inf], np.nan).dropna(subset=[TARGET])
    common = sorted(numeric_cols(nasa) & numeric_cols(calce))
    print(f"Common features ({len(common)}): {common}")

    fractions = [0.02, 0.05, 0.1, 0.2]
    rows = []
    control_rows = []
    prediction_frames = []
    split_frames = []
    templates = get_models()
    source_models = {}
    for model_name, template in templates.items():
        source_models[model_name] = clone(template).fit(nasa[common], nasa[TARGET])
    for frac in fractions:
        cal_target, cal_test = representative_split(calce, frac)
        keys = ["battery_id", "source_file", "source_cycle_index"]
        assert not set(map(tuple, cal_target[keys].to_numpy())) & set(map(tuple, cal_test[keys].to_numpy()))
        for role, part in [("calibration", cal_target), ("test", cal_test)]:
            split_frames.append(part[keys + ["cycle_index"]].assign(target_fraction=frac, role=role))
        train = pd.concat([nasa, cal_target], ignore_index=True)
        for model_name, template in templates.items():
            for mode, train_frame in [("source_only", None), ("target_only", cal_target), ("source_plus_target", train)]:
                print(f"fraction={frac:.2f}: {mode} {model_name}", flush=True)
                if train_frame is None:
                    model = source_models[model_name]
                else:
                    model = clone(template).fit(train_frame[common], train_frame[TARGET])
                pred = model.predict(cal_test[common])
                row = {
                    "model": model_name,
                    "target_fraction": frac,
                    "target_samples": len(cal_target),
                    "test_samples": len(cal_test),
                    **metrics(cal_test[TARGET].to_numpy(dtype=float), pred),
                }
                control_rows.append({**row, "training_mode": mode})
                if mode == "source_plus_target":
                    rows.append(row)
                prediction_frames.append(cal_test[keys + ["cycle_index", TARGET]].assign(
                    target_fraction=frac, training_mode=mode, model=model_name, y_pred=pred
                ))

    result = pd.DataFrame(rows).sort_values(["target_fraction", "RMSE"])
    out_dir = Path("results/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_dir / "representative_domain_adaptation_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(control_rows).to_csv(out_dir / "adaptation_controls_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(prediction_frames, ignore_index=True).to_csv(out_dir / "adaptation_controls_predictions.csv", index=False, encoding="utf-8-sig")
    pd.concat(split_frames, ignore_index=True).to_csv(out_dir / "adaptation_split_manifest.csv", index=False, encoding="utf-8-sig")
    controls = pd.DataFrame(control_rows)
    plot = controls[controls["model"] == "GradientBoosting"].copy()
    plot["model"] = plot["training_mode"].map({"source_only": "Source only", "target_only": "Target only", "source_plus_target": "Source + target"})
    save_adaptation_svg(plot, Path("results/figures/representative_domain_adaptation_rmse.svg"))
    print("Saved: results/tables/representative_domain_adaptation_metrics.csv")
    print("Saved: results/figures/representative_domain_adaptation_rmse.svg")
    print(result.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

