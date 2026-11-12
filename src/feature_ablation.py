from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from train_baseline import (
    TARGET,
    GradientBoostingRegressorLite,
    KNNRegressorLite,
    RandomForestRegressorLite,
    RidgeRegressorLite,
    StandardScalerLite,
    metrics,
    split_by_battery,
)


BASE_NON_FEATURES = {
    "battery_id",
    "raw_cycle_index",
    "discharge_cycle_index",
    "capacity_Ah",
    "capacity_retention_pct",
    TARGET,
}


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in df.columns
        if col not in BASE_NON_FEATURES and pd.api.types.is_numeric_dtype(df[col])
    ]


def feature_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    all_cols = numeric_feature_columns(df)
    discharge = [
        col
        for col in all_cols
        if not col.startswith("charge_") and not col.startswith("impedance_")
    ]
    charge = [col for col in all_cols if col.startswith("charge_")]
    impedance = [col for col in all_cols if col.startswith("impedance_")]
    return {
        "discharge_only": discharge,
        "discharge_charge": discharge + charge,
        "discharge_charge_impedance": discharge + charge + impedance,
    }


def models() -> dict[str, object]:
    return {
        "Ridge": RidgeRegressorLite(alpha=1.0),
        "KNN": KNNRegressorLite(k=7),
        "RandomForestLite": RandomForestRegressorLite(n_estimators=80, max_depth=6, min_samples_leaf=5),
        "GradientBoostingLite": GradientBoostingRegressorLite(
            n_estimators=120,
            learning_rate=0.04,
            max_depth=3,
            min_samples_leaf=5,
        ),
    }


def save_ablation_svg(summary: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 980, 560
    pad_l, pad_r, pad_t, pad_b = 95, 35, 60, 90
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    groups = list(summary["feature_group"].drop_duplicates())
    model_names = list(summary["model"].drop_duplicates())
    max_v = float(summary["RMSE"].max()) * 1.15
    colors = {
        "Ridge": "#2563eb",
        "KNN": "#a16207",
        "RandomForestLite": "#b91c1c",
        "GradientBoostingLite": "#0f766e",
    }

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="32" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">Feature Group Ablation: Mean RMSE</text>',
        f'<line x1="{pad_l}" y1="{height-pad_b}" x2="{width-pad_r}" y2="{height-pad_b}" stroke="#111827" stroke-width="1.2"/>',
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{height-pad_b}" stroke="#111827" stroke-width="1.2"/>',
    ]
    for i in range(6):
        value = max_v * i / 5
        y = height - pad_b - value / max_v * plot_h
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width-pad_r}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{pad_l-10}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{value:.3f}</text>')

    group_band = plot_w / max(len(groups), 1)
    bar_w = group_band / (len(model_names) + 1) * 0.72
    for gi, group in enumerate(groups):
        group_x = pad_l + gi * group_band
        for mi, model in enumerate(model_names):
            row = summary[(summary["feature_group"] == group) & (summary["model"] == model)]
            if row.empty:
                continue
            value = float(row.iloc[0]["RMSE"])
            h = value / max_v * plot_h
            x = group_x + (mi + 0.65) * group_band / (len(model_names) + 1)
            y = height - pad_b - h
            color = colors.get(model, "#334155")
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}"/>')
        label = group.replace("_", " ")
        cx = group_x + group_band / 2
        parts.append(f'<text x="{cx:.1f}" y="{height-pad_b+24}" text-anchor="middle" font-family="Arial" font-size="12">{label}</text>')

    legend_x = pad_l
    legend_y = height - 38
    for idx, model in enumerate(model_names):
        x = legend_x + idx * 205
        color = colors.get(model, "#334155")
        parts.append(f'<rect x="{x}" y="{legend_y}" width="14" height="14" fill="{color}"/>')
        parts.append(f'<text x="{x+22}" y="{legend_y+12}" font-family="Arial" font-size="13">{model}</text>')

    parts.append(f'<text x="24" y="{height/2}" transform="rotate(-90 24,{height/2})" text-anchor="middle" font-family="Arial" font-size="14">Mean RMSE</text>')
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    feature_path = Path("data/processed/nasa_soh_features.csv")
    if not feature_path.exists():
        print("Feature table not found. Run: python src/prepare_nasa_soh.py")
        return 1

    df = pd.read_csv(feature_path).replace([np.inf, -np.inf], np.nan).dropna(subset=[TARGET])
    batteries = sorted(df["battery_id"].dropna().unique())
    groups = feature_groups(df)
    rows = []

    for group_name, cols in groups.items():
        if not cols:
            continue
        for test_battery in batteries:
            train_df, test_df, _ = split_by_battery(df, test_battery)
            scaler = StandardScalerLite().fit(train_df[cols].to_numpy(dtype=float))
            x_train = scaler.transform(train_df[cols].to_numpy(dtype=float))
            y_train = train_df[TARGET].to_numpy(dtype=float)
            x_test = scaler.transform(test_df[cols].to_numpy(dtype=float))
            y_test = test_df[TARGET].to_numpy(dtype=float)

            for model_name, model in models().items():
                print(f"group={group_name}, model={model_name}, test={test_battery}")
                model.fit(x_train, y_train)  # type: ignore[attr-defined]
                y_pred = model.predict(x_test)  # type: ignore[attr-defined]
                rows.append(
                    {
                        "feature_group": group_name,
                        "n_features": len(cols),
                        "model": model_name,
                        "test_battery": test_battery,
                        **metrics(y_test, y_pred),
                    }
                )

    out_dir = Path("results/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "feature_ablation_metrics.csv", index=False, encoding="utf-8-sig")
    summary = (
        result.groupby(["feature_group", "n_features", "model"], as_index=False)[
            ["MAE", "RMSE", "MAPE_pct", "R2"]
        ]
        .mean()
        .sort_values(["feature_group", "RMSE"])
    )
    summary.to_csv(out_dir / "feature_ablation_summary.csv", index=False, encoding="utf-8-sig")
    save_ablation_svg(summary, Path("results/figures/feature_ablation_rmse.svg"))
    print("Saved: results/tables/feature_ablation_metrics.csv")
    print("Saved: results/tables/feature_ablation_summary.csv")
    print("Saved: results/figures/feature_ablation_rmse.svg")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

