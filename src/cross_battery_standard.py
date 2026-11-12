from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from train_baseline import TARGET, feature_columns, metrics, split_by_battery
from train_standard_models import build_models


def save_standard_summary_svg(summary: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = summary.sort_values("RMSE")
    width, height = 900, 500
    pad_l, pad_r, pad_t, pad_b = 95, 35, 55, 80
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    max_v = float(data["RMSE"].max()) * 1.15
    bar_w = plot_w / max(len(data), 1) * 0.58
    colors = ["#0f766e", "#2563eb", "#b91c1c", "#a16207", "#7c3aed", "#0891b2"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">Standard Models: Cross-Battery RMSE</text>',
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
    feature_path = Path("data/processed/nasa_soh_features.csv")
    if not feature_path.exists():
        print("Feature table not found. Run: python src/prepare_nasa_soh.py")
        return 1

    df = pd.read_csv(feature_path).replace([np.inf, -np.inf], np.nan).dropna(subset=[TARGET])
    cols = feature_columns(df)
    batteries = sorted(df["battery_id"].dropna().unique())
    rows = []

    for test_battery in batteries:
        train_df, test_df, _ = split_by_battery(df, test_battery)
        x_train = train_df[cols]
        y_train = train_df[TARGET].to_numpy(dtype=float)
        x_test = test_df[cols]
        y_test = test_df[TARGET].to_numpy(dtype=float)

        for model_name, model in build_models().items():
            # MLP was unstable in the first run; keep it in single-test script but skip it here.
            if model_name == "MLP":
                continue
            print(f"Training {model_name}, test={test_battery}")
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            rows.append(
                {
                    "model": model_name,
                    "test_battery": test_battery,
                    "train_samples": len(train_df),
                    "test_samples": len(test_df),
                    **metrics(y_test, y_pred),
                }
            )

    out_dir = Path("results/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows).sort_values(["test_battery", "RMSE"])
    result.to_csv(out_dir / "standard_cross_battery_metrics.csv", index=False, encoding="utf-8-sig")
    summary = (
        result.groupby("model", as_index=False)[["MAE", "RMSE", "MAPE_pct", "R2"]]
        .mean()
        .sort_values("RMSE")
    )
    summary.to_csv(out_dir / "standard_cross_battery_summary.csv", index=False, encoding="utf-8-sig")
    save_standard_summary_svg(summary, Path("results/figures/standard_cross_battery_rmse.svg"))
    print("Saved: results/tables/standard_cross_battery_metrics.csv")
    print("Saved: results/tables/standard_cross_battery_summary.csv")
    print("Saved: results/figures/standard_cross_battery_rmse.svg")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
