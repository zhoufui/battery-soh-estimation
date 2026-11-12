from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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


def fit_linear_calibrator(pred: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    pred = np.asarray(pred, dtype=float)
    truth = np.asarray(truth, dtype=float)
    mask = np.isfinite(pred) & np.isfinite(truth)
    pred = pred[mask]
    truth = truth[mask]
    if pred.size < 2 or float(np.std(pred)) < 1e-12:
        return 1.0, 0.0
    design = np.column_stack([pred, np.ones_like(pred)])
    coef, *_ = np.linalg.lstsq(design, truth, rcond=None)
    return float(coef[0]), float(coef[1])


def save_calibration_svg(summary: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 960, 560
    pad_l, pad_r, pad_t, pad_b = 78, 35, 58, 70
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    x_values = sorted(summary["calibration_fraction"].unique())
    y_max = float(summary["RMSE"].max()) * 1.1
    y_min = max(0.0, float(summary["RMSE"].min()) * 0.8)
    colors = {
        "SVR_RBF": "#7c3aed",
        "RandomForest": "#b91c1c",
        "GradientBoosting": "#a16207",
        "XGBoost": "#2563eb",
        "LightGBM": "#0f766e",
    }

    def sx(x: float) -> float:
        return pad_l + (x - min(x_values)) / max(max(x_values) - min(x_values), 1e-12) * plot_w

    def sy(y: float) -> float:
        return pad_t + (y_max - y) / max(y_max - y_min, 1e-12) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">NASA to CALCE Target-Domain Calibration</text>',
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
        group = group.sort_values("calibration_fraction")
        color = colors.get(model, "#334155")
        points = " ".join(
            f"{sx(float(row['calibration_fraction'])):.1f},{sy(float(row['RMSE'])):.1f}"
            for _, row in group.iterrows()
        )
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.3"/>')
        for _, row in group.iterrows():
            parts.append(
                f'<circle cx="{sx(float(row["calibration_fraction"])):.1f}" cy="{sy(float(row["RMSE"])):.1f}" r="3.5" fill="{color}"/>'
            )
        parts.append(f'<rect x="{pad_l+15+idx*165}" y="{legend_y}" width="13" height="13" fill="{color}"/>')
        parts.append(f'<text x="{pad_l+34+idx*165}" y="{legend_y+12}" font-family="Arial" font-size="12">{model}</text>')

    parts.append(f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-family="Arial" font-size="14">Calibration fraction of target CALCE cycles</text>')
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

    x_nasa = nasa[common]
    y_nasa = nasa[TARGET].to_numpy(dtype=float)
    fractions = [0.0, 0.05, 0.1, 0.2, 0.3]
    rows = []
    prediction_rows = []

    for model_name, model in get_models().items():
        print(f"Training source model: {model_name}")
        model.fit(x_nasa, y_nasa)
        calce_pred = model.predict(calce[common])
        working = calce[["battery_id", "cycle_index", TARGET]].copy()
        working["raw_pred"] = calce_pred

        for frac in fractions:
            cal_parts = []
            test_parts = []
            for _, group in working.groupby("battery_id"):
                group = group.sort_values("cycle_index").reset_index(drop=True)
                if frac <= 0:
                    cal_n = 0
                else:
                    cal_n = max(5, int(round(len(group) * frac)))
                    cal_n = min(cal_n, len(group) - 5)
                if cal_n > 0:
                    cal_parts.append(group.iloc[:cal_n])
                    test_parts.append(group.iloc[cal_n:])
                else:
                    test_parts.append(group)
            test_df = pd.concat(test_parts, ignore_index=True)
            if cal_parts:
                cal_df = pd.concat(cal_parts, ignore_index=True)
                a, b = fit_linear_calibrator(cal_df["raw_pred"].to_numpy(), cal_df[TARGET].to_numpy())
            else:
                a, b = 1.0, 0.0
            adjusted = a * test_df["raw_pred"].to_numpy(dtype=float) + b
            row = {
                "model": model_name,
                "calibration_fraction": frac,
                "calibration_samples": int(sum(len(x) for x in cal_parts)) if cal_parts else 0,
                "test_samples": len(test_df),
                "slope": a,
                "intercept": b,
                **metrics(test_df[TARGET].to_numpy(dtype=float), adjusted),
            }
            rows.append(row)
            for idx, pred_value in enumerate(adjusted):
                prediction_rows.append(
                    {
                        "model": model_name,
                        "calibration_fraction": frac,
                        "battery_id": test_df.iloc[idx]["battery_id"],
                        "cycle_index": test_df.iloc[idx]["cycle_index"],
                        "y_true": test_df.iloc[idx][TARGET],
                        "y_pred": pred_value,
                    }
                )

    result = pd.DataFrame(rows).sort_values(["calibration_fraction", "RMSE"])
    predictions = pd.DataFrame(prediction_rows)
    out_dir = Path("results/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_dir / "domain_calibration_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(out_dir / "domain_calibration_predictions.csv", index=False, encoding="utf-8-sig")
    save_calibration_svg(result, Path("results/figures/domain_calibration_rmse.svg"))
    print("Saved: results/tables/domain_calibration_metrics.csv")
    print("Saved: results/tables/domain_calibration_predictions.csv")
    print("Saved: results/figures/domain_calibration_rmse.svg")
    print(result.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

