from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_baseline import TARGET, feature_columns, metrics, split_by_battery


def import_sklearn():
    try:
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVR
    except Exception as exc:
        raise RuntimeError(
            "scikit-learn is not installed. Install with: python -m pip install scikit-learn"
        ) from exc
    return {
        "GradientBoostingRegressor": GradientBoostingRegressor,
        "RandomForestRegressor": RandomForestRegressor,
        "SimpleImputer": SimpleImputer,
        "MLPRegressor": MLPRegressor,
        "Pipeline": Pipeline,
        "StandardScaler": StandardScaler,
        "SVR": SVR,
    }


def optional_xgboost() -> Any | None:
    try:
        from xgboost import XGBRegressor
    except Exception:
        return None
    return XGBRegressor


def optional_lightgbm() -> Any | None:
    try:
        from lightgbm import LGBMRegressor
    except Exception:
        return None
    return LGBMRegressor


def build_models(seed: int = 42) -> dict[str, Any]:
    sk = import_sklearn()
    Pipeline = sk["Pipeline"]
    SimpleImputer = sk["SimpleImputer"]
    StandardScaler = sk["StandardScaler"]
    RandomForestRegressor = sk["RandomForestRegressor"]
    GradientBoostingRegressor = sk["GradientBoostingRegressor"]
    SVR = sk["SVR"]
    MLPRegressor = sk["MLPRegressor"]

    models: dict[str, Any] = {
        "SVR_RBF": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", SVR(kernel="rbf", C=20.0, epsilon=0.01, gamma="scale")),
            ]
        ),
        "RandomForest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        max_depth=None,
                        min_samples_leaf=4,
                        random_state=seed,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "GradientBoosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=300,
                        learning_rate=0.04,
                        max_depth=3,
                        min_samples_leaf=4,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "MLP": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        alpha=1e-3,
                        learning_rate_init=1e-3,
                        max_iter=2000,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }

    XGBRegressor = optional_xgboost()
    if XGBRegressor is not None:
        models["XGBoost"] = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=400,
                        max_depth=3,
                        learning_rate=0.03,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="reg:squarederror",
                        random_state=seed,
                    ),
                ),
            ]
        )

    LGBMRegressor = optional_lightgbm()
    if LGBMRegressor is not None:
        models["LightGBM"] = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    LGBMRegressor(
                        n_estimators=400,
                        max_depth=4,
                        learning_rate=0.03,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        verbose=-1,
                        random_state=seed,
                    ),
                ),
            ]
        )
    return models


def save_predictions_svg(pred_df: pd.DataFrame, out_path: Path, test_battery: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 960, 560
    pad_l, pad_r, pad_t, pad_b = 75, 35, 55, 70
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    x_min = int(pred_df["discharge_cycle_index"].min())
    x_max = int(pred_df["discharge_cycle_index"].max())
    model_cols = [c for c in pred_df.columns if c not in {"battery_id", "discharge_cycle_index", "y_true"}]
    y_min = min([float(pred_df["y_true"].min())] + [float(pred_df[c].min()) for c in model_cols] + [0.65])
    y_max = max([float(pred_df["y_true"].max())] + [float(pred_df[c].max()) for c in model_cols] + [1.02])

    def sx(x: float) -> float:
        return pad_l + (x - x_min) / max(x_max - x_min, 1) * plot_w

    def sy(y: float) -> float:
        return pad_t + (y_max - y) / max(y_max - y_min, 1e-9) * plot_h

    colors = ["#111827", "#2563eb", "#b91c1c", "#0f766e", "#a16207", "#7c3aed", "#0891b2"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">Standard Models on {test_battery}</text>',
        f'<line x1="{pad_l}" y1="{height-pad_b}" x2="{width-pad_r}" y2="{height-pad_b}" stroke="#111827" stroke-width="1.5"/>',
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{height-pad_b}" stroke="#111827" stroke-width="1.5"/>',
    ]
    series = [("Actual", "y_true")] + [(col, col) for col in model_cols[:6]]
    for idx, (label, col) in enumerate(series):
        points = []
        for _, row in pred_df.sort_values("discharge_cycle_index").iterrows():
            points.append(f"{sx(float(row['discharge_cycle_index'])):.1f},{sy(float(row[col])):.1f}")
        color = colors[idx % len(colors)]
        width_line = 2.6 if col == "y_true" else 1.9
        parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="{width_line}"/>')
        parts.append(f'<rect x="{pad_l+15}" y="{pad_t+12+idx*20}" width="13" height="13" fill="{color}"/>')
        parts.append(f'<text x="{pad_l+35}" y="{pad_t+24+idx*20}" font-family="Arial" font-size="12">{label}</text>')
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def maybe_shap(best_pipeline: Any, x_train: pd.DataFrame, x_test: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame | None:
    try:
        import shap
    except Exception:
        return None

    try:
        model = best_pipeline.named_steps["model"]
        imputer = best_pipeline.named_steps.get("imputer")
        x_train_imp = imputer.transform(x_train) if imputer is not None else x_train.to_numpy()
        x_test_imp = imputer.transform(x_test) if imputer is not None else x_test.to_numpy()
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(x_test_imp)
        mean_abs = np.mean(np.abs(values), axis=0)
        return pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs}).sort_values(
            "mean_abs_shap", ascending=False
        )
    except Exception as exc:
        print(f"SHAP skipped: {exc}")
        return None


def save_shap_svg(shap_df: pd.DataFrame, out_path: Path, top_n: int = 12) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = shap_df.head(top_n).iloc[::-1]
    width, height = 980, 560
    pad_l, pad_r, pad_t, pad_b = 250, 40, 55, 50
    plot_w = width - pad_l - pad_r
    row_h = (height - pad_t - pad_b) / max(len(data), 1)
    max_v = float(data["mean_abs_shap"].max())
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">SHAP Feature Importance</text>',
        f'<line x1="{pad_l}" y1="{height-pad_b}" x2="{width-pad_r}" y2="{height-pad_b}" stroke="#111827" stroke-width="1.2"/>',
    ]
    for i, (_, row) in enumerate(data.iterrows()):
        y = pad_t + i * row_h + row_h * 0.18
        bar_h = row_h * 0.64
        value = float(row["mean_abs_shap"])
        bar_w = value / max(max_v, 1e-12) * plot_w
        label = str(row["feature"])
        parts.append(f'<text x="{pad_l-12}" y="{y+bar_h*0.68:.1f}" text-anchor="end" font-family="Arial" font-size="13">{label}</text>')
        parts.append(f'<rect x="{pad_l}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="#2563eb"/>')
        parts.append(f'<text x="{pad_l+bar_w+7:.1f}" y="{y+bar_h*0.68:.1f}" font-family="Arial" font-size="12">{value:.4f}</text>')
    parts.append(f'<text x="{pad_l + plot_w/2}" y="{height-16}" text-anchor="middle" font-family="Arial" font-size="14">Mean absolute SHAP value</text>')
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train standard sklearn/XGBoost/LightGBM SOH models.")
    parser.add_argument("--features", default="data/processed/nasa_soh_features.csv")
    parser.add_argument("--test-battery", default="B0018")
    args = parser.parse_args()

    feature_path = Path(args.features)
    if not feature_path.exists():
        print(f"Feature table not found: {feature_path}")
        return 1

    try:
        model_specs = build_models()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    df = pd.read_csv(feature_path).replace([np.inf, -np.inf], np.nan).dropna(subset=[TARGET])
    cols = feature_columns(df)
    train_df, test_df, test_battery = split_by_battery(df, args.test_battery)
    x_train = train_df[cols]
    y_train = train_df[TARGET].to_numpy(dtype=float)
    x_test = test_df[cols]
    y_test = test_df[TARGET].to_numpy(dtype=float)

    rows = []
    pred_df = test_df[["battery_id", "discharge_cycle_index"]].copy()
    pred_df["y_true"] = y_test
    fitted: dict[str, Any] = {}

    for name, model in model_specs.items():
        print(f"Training {name}")
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        pred_df[name] = y_pred
        rows.append({"model": name, "test_battery": test_battery, **metrics(y_test, y_pred)})
        fitted[name] = model

    out_dir = Path("results/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(rows).sort_values("RMSE")
    metrics_df.to_csv(out_dir / "standard_model_metrics.csv", index=False, encoding="utf-8-sig")
    pred_df.to_csv(out_dir / "standard_model_predictions.csv", index=False, encoding="utf-8-sig")
    save_predictions_svg(pred_df, Path("results/figures/standard_model_predictions.svg"), test_battery)

    best_name = str(metrics_df.iloc[0]["model"])
    shap_df = maybe_shap(fitted[best_name], x_train, x_test, cols)
    if shap_df is not None:
        shap_df.to_csv(out_dir / "shap_feature_importance.csv", index=False, encoding="utf-8-sig")
        save_shap_svg(shap_df, Path("results/figures/shap_feature_importance.svg"))
        print("Saved: results/tables/shap_feature_importance.csv")
        print("Saved: results/figures/shap_feature_importance.svg")

    print("Saved: results/tables/standard_model_metrics.csv")
    print("Saved: results/tables/standard_model_predictions.csv")
    print("Saved: results/figures/standard_model_predictions.svg")
    print(metrics_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
