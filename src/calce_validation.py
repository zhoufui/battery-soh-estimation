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
    return {k: v for k, v in build_models().items() if k != "MLP"}


def evaluate_models(train_df: pd.DataFrame, test_df: pd.DataFrame, cols: list[str], label: str) -> pd.DataFrame:
    rows = []
    x_train = train_df[cols]
    y_train = train_df[TARGET].to_numpy(dtype=float)
    x_test = test_df[cols]
    y_test = test_df[TARGET].to_numpy(dtype=float)
    for model_name, model in get_models().items():
        print(f"{label}: training {model_name}")
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        rows.append({"experiment": label, "model": model_name, **metrics(y_test, pred)})
    return pd.DataFrame(rows)


def save_experiment_svg(summary: pd.DataFrame, out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = summary.sort_values("RMSE")
    width, height = 900, 500
    pad_l, pad_r, pad_t, pad_b = 95, 35, 55, 80
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    max_v = float(data["RMSE"].max()) * 1.15
    bar_w = plot_w / max(len(data), 1) * 0.58
    colors = ["#0f766e", "#2563eb", "#b91c1c", "#a16207", "#7c3aed"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">{title}</text>',
        f'<line x1="{pad_l}" y1="{height-pad_b}" x2="{width-pad_r}" y2="{height-pad_b}" stroke="#111827" stroke-width="1.2"/>',
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{height-pad_b}" stroke="#111827" stroke-width="1.2"/>',
    ]
    for i in range(6):
        value = max_v * i / 5
        y = height - pad_b - value / max(max_v, 1e-12) * plot_h
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width-pad_r}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{pad_l-10}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{value:.3f}</text>')
    for idx, (_, row) in enumerate(data.iterrows()):
        x_center = pad_l + (idx + 0.5) / len(data) * plot_w
        value = float(row["RMSE"])
        h = value / max(max_v, 1e-12) * plot_h
        x = x_center - bar_w / 2
        y = height - pad_b - h
        color = colors[idx % len(colors)]
        model = str(row["model"])
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}"/>')
        parts.append(f'<text x="{x_center:.1f}" y="{y-8:.1f}" text-anchor="middle" font-family="Arial" font-size="12">{value:.4f}</text>')
        parts.append(f'<text x="{x_center:.1f}" y="{height-pad_b+22}" text-anchor="middle" font-family="Arial" font-size="12">{model}</text>')
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
    if not common:
        print("No common numeric features found.")
        return 1
    print(f"Common features ({len(common)}): {common}")

    rows = []
    for test_battery in sorted(calce["battery_id"].unique()):
        train_df = calce[calce["battery_id"] != test_battery].copy()
        test_df = calce[calce["battery_id"] == test_battery].copy()
        if train_df.empty or test_df.empty:
            continue
        rows.append(evaluate_models(train_df, test_df, common, f"CALCE_leave_{test_battery}"))
    calce_internal = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    cross_dataset = evaluate_models(nasa, calce, common, "NASA_train_CALCE_test")

    out_dir = Path("results/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    calce_internal.to_csv(out_dir / "calce_internal_metrics.csv", index=False, encoding="utf-8-sig")
    cross_dataset.to_csv(out_dir / "nasa_to_calce_metrics.csv", index=False, encoding="utf-8-sig")
    if not calce_internal.empty:
        internal_summary = (
            calce_internal.groupby("model", as_index=False)[["MAE", "RMSE", "MAPE_pct", "R2"]]
            .mean()
            .sort_values("RMSE")
        )
        internal_summary.to_csv(out_dir / "calce_internal_summary.csv", index=False, encoding="utf-8-sig")
        save_experiment_svg(internal_summary, Path("results/figures/calce_internal_rmse.svg"), "CALCE Internal Validation")
        print("CALCE internal summary")
        print(internal_summary.to_string(index=False))
    save_experiment_svg(cross_dataset, Path("results/figures/nasa_to_calce_rmse.svg"), "NASA to CALCE Validation")
    print("NASA to CALCE")
    print(cross_dataset.sort_values("RMSE").to_string(index=False))
    print("Saved CALCE validation tables and figures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

