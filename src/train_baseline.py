from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TARGET = "soh"
NON_FEATURES = {
    "battery_id",
    "raw_cycle_index",
    "discharge_cycle_index",
    "capacity_Ah",
    "capacity_retention_pct",
    TARGET,
}


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    mape = float(np.mean(np.abs(err / np.maximum(np.abs(y_true), 1e-9))) * 100.0)
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - float(np.sum(err**2)) / denom if denom > 0 else math.nan
    return {"MAE": mae, "RMSE": rmse, "MAPE_pct": mape, "R2": r2}


@dataclass
class StandardScalerLite:
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "StandardScalerLite":
        self.mean_ = np.nanmean(x, axis=0)
        self.std_ = np.nanstd(x, axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler is not fitted.")
        out = (x - self.mean_) / self.std_
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


class RidgeRegressorLite:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.coef_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgeRegressorLite":
        xb = np.column_stack([np.ones(x.shape[0]), x])
        eye = np.eye(xb.shape[1])
        eye[0, 0] = 0.0
        self.coef_ = np.linalg.pinv(xb.T @ xb + self.alpha * eye) @ xb.T @ y
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted.")
        xb = np.column_stack([np.ones(x.shape[0]), x])
        return xb @ self.coef_


class KNNRegressorLite:
    def __init__(self, k: int = 7):
        self.k = k
        self.x_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "KNNRegressorLite":
        self.x_train = x
        self.y_train = y
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.x_train is None or self.y_train is None:
            raise RuntimeError("Model is not fitted.")
        preds = []
        for row in x:
            distances = np.sqrt(np.sum((self.x_train - row) ** 2, axis=1))
            idx = np.argsort(distances)[: self.k]
            weights = 1.0 / np.maximum(distances[idx], 1e-9)
            preds.append(float(np.sum(weights * self.y_train[idx]) / np.sum(weights)))
        return np.array(preds)


class TreeNode:
    def __init__(
        self,
        value: float,
        feature: int | None = None,
        threshold: float | None = None,
        left: "TreeNode | None" = None,
        right: "TreeNode | None" = None,
    ):
        self.value = value
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right


class DecisionTreeRegressorLite:
    def __init__(self, max_depth: int = 5, min_samples_leaf: int = 5, max_features: int | None = None, rng: np.random.Generator | None = None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.rng = rng or np.random.default_rng(0)
        self.root: TreeNode | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "DecisionTreeRegressorLite":
        self.root = self._build(x, y, depth=0)
        return self

    def _build(self, x: np.ndarray, y: np.ndarray, depth: int) -> TreeNode:
        value = float(np.mean(y))
        if depth >= self.max_depth or len(y) < self.min_samples_leaf * 2 or float(np.var(y)) < 1e-12:
            return TreeNode(value=value)

        n_features = x.shape[1]
        max_features = self.max_features or max(1, int(math.sqrt(n_features)))
        features = self.rng.choice(n_features, size=min(max_features, n_features), replace=False)
        best: tuple[float, int, float] | None = None

        for feature in features:
            values = x[:, feature]
            candidates = np.unique(np.quantile(values, np.linspace(0.1, 0.9, 9)))
            for threshold in candidates:
                left_mask = values <= threshold
                right_mask = ~left_mask
                if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
                    continue
                loss = float(np.var(y[left_mask]) * left_mask.sum() + np.var(y[right_mask]) * right_mask.sum())
                if best is None or loss < best[0]:
                    best = (loss, int(feature), float(threshold))

        if best is None:
            return TreeNode(value=value)

        _, feature, threshold = best
        mask = x[:, feature] <= threshold
        return TreeNode(
            value=value,
            feature=feature,
            threshold=threshold,
            left=self._build(x[mask], y[mask], depth + 1),
            right=self._build(x[~mask], y[~mask], depth + 1),
        )

    def _predict_one(self, row: np.ndarray, node: TreeNode) -> float:
        while node.feature is not None and node.threshold is not None and node.left is not None and node.right is not None:
            node = node.left if row[node.feature] <= node.threshold else node.right
        return node.value

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise RuntimeError("Model is not fitted.")
        return np.array([self._predict_one(row, self.root) for row in x])


class RandomForestRegressorLite:
    def __init__(self, n_estimators: int = 80, max_depth: int = 6, min_samples_leaf: int = 5, seed: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.seed = seed
        self.trees: list[DecisionTreeRegressorLite] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RandomForestRegressorLite":
        rng = np.random.default_rng(self.seed)
        self.trees = []
        n = x.shape[0]
        for i in range(self.n_estimators):
            idx = rng.integers(0, n, size=n)
            tree = DecisionTreeRegressorLite(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                rng=np.random.default_rng(self.seed + i + 1),
            )
            tree.fit(x[idx], y[idx])
            self.trees.append(tree)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self.trees:
            raise RuntimeError("Model is not fitted.")
        return np.mean([tree.predict(x) for tree in self.trees], axis=0)


class GradientBoostingRegressorLite:
    def __init__(
        self,
        n_estimators: int = 80,
        learning_rate: float = 0.05,
        max_depth: int = 3,
        min_samples_leaf: int = 5,
        seed: int = 42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.seed = seed
        self.init_: float = 0.0
        self.trees: list[DecisionTreeRegressorLite] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> "GradientBoostingRegressorLite":
        self.init_ = float(np.mean(y))
        pred = np.full_like(y, self.init_, dtype=float)
        self.trees = []
        for i in range(self.n_estimators):
            residual = y - pred
            tree = DecisionTreeRegressorLite(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=x.shape[1],
                rng=np.random.default_rng(self.seed + i + 1),
            )
            tree.fit(x, residual)
            update = tree.predict(x)
            pred += self.learning_rate * update
            self.trees.append(tree)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        pred = np.full(x.shape[0], self.init_, dtype=float)
        for tree in self.trees:
            pred += self.learning_rate * tree.predict(x)
        return pred


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        if col in NON_FEATURES:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def split_by_battery(df: pd.DataFrame, test_battery: str | None) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    batteries = sorted(df["battery_id"].dropna().unique())
    if len(batteries) < 2:
        raise ValueError("Need at least two batteries for cross-battery validation.")
    chosen = test_battery or batteries[-1]
    if chosen not in batteries:
        raise ValueError(f"Test battery {chosen} not found. Available: {batteries}")
    train = df[df["battery_id"] != chosen].copy()
    test = df[df["battery_id"] == chosen].copy()
    return train, test, chosen


def fit_predict_models(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> dict[str, np.ndarray]:
    models: dict[str, Any] = {
        "Ridge": RidgeRegressorLite(alpha=1.0),
        "KNN": KNNRegressorLite(k=7),
        "RandomForestLite": RandomForestRegressorLite(n_estimators=80, max_depth=6, min_samples_leaf=5),
        "GradientBoostingLite": GradientBoostingRegressorLite(n_estimators=120, learning_rate=0.04, max_depth=3, min_samples_leaf=5),
    }
    preds = {}
    for name, model in models.items():
        print(f"Training {name}")
        model.fit(x_train, y_train)
        preds[name] = model.predict(x_test)
    return preds


def permutation_importance(model: Any, x: np.ndarray, y: np.ndarray, columns: list[str], seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    baseline = metrics(y, model.predict(x))["RMSE"]
    rows = []
    for j, col in enumerate(columns):
        x_perm = x.copy()
        x_perm[:, j] = rng.permutation(x_perm[:, j])
        score = metrics(y, model.predict(x_perm))["RMSE"]
        rows.append({"feature": col, "rmse_increase": score - baseline})
    return pd.DataFrame(rows).sort_values("rmse_increase", ascending=False)


def save_prediction_svg(pred_df: pd.DataFrame, out_path: Path, test_battery: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 920, 560
    pad_l, pad_r, pad_t, pad_b = 75, 35, 55, 70
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    x_min = int(pred_df["discharge_cycle_index"].min())
    x_max = int(pred_df["discharge_cycle_index"].max())
    y_cols = [c for c in pred_df.columns if c not in {"battery_id", "discharge_cycle_index", "y_true"}]
    y_min = min(float(pred_df["y_true"].min()), *(float(pred_df[c].min()) for c in y_cols), 0.65)
    y_max = max(float(pred_df["y_true"].max()), *(float(pred_df[c].max()) for c in y_cols), 1.02)

    def sx(x: float) -> float:
        return pad_l + (x - x_min) / max(x_max - x_min, 1) * plot_w

    def sy(y: float) -> float:
        return pad_t + (y_max - y) / max(y_max - y_min, 1e-9) * plot_h

    series = [
        ("Actual", "y_true", "#111827"),
        ("Ridge", "Ridge", "#2563eb"),
        ("KNN", "KNN", "#a16207"),
        ("RandomForestLite", "RandomForestLite", "#b91c1c"),
        ("GradientBoostingLite", "GradientBoostingLite", "#0f766e"),
    ]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">SOH Prediction on {test_battery}</text>',
        f'<line x1="{pad_l}" y1="{height-pad_b}" x2="{width-pad_r}" y2="{height-pad_b}" stroke="#111827" stroke-width="1.5"/>',
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{height-pad_b}" stroke="#111827" stroke-width="1.5"/>',
    ]
    for i in range(6):
        y = y_min + (y_max - y_min) * i / 5
        yy = sy(y)
        parts.append(f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{width-pad_r}" y2="{yy:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{pad_l-10}" y="{yy+4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{y:.2f}</text>')
    for idx, (label, col, color) in enumerate(series):
        if col not in pred_df.columns:
            continue
        points = []
        for _, row in pred_df.sort_values("discharge_cycle_index").iterrows():
            points.append(f"{sx(float(row['discharge_cycle_index'])):.1f},{sy(float(row[col])):.1f}")
        parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2.2"/>')
        parts.append(f'<rect x="{pad_l+15}" y="{pad_t+12+idx*22}" width="14" height="14" fill="{color}"/>')
        parts.append(f'<text x="{pad_l+37}" y="{pad_t+24+idx*22}" font-family="Arial" font-size="13">{label}</text>')
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def save_importance_svg(importance: pd.DataFrame, out_path: Path, top_n: int = 12) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = importance.head(top_n).copy()
    data = data[data["rmse_increase"] > 0]
    if data.empty:
        return
    data = data.iloc[::-1]
    width, height = 980, 560
    pad_l, pad_r, pad_t, pad_b = 260, 40, 55, 50
    plot_w = width - pad_l - pad_r
    row_h = (height - pad_t - pad_b) / max(len(data), 1)
    max_v = float(data["rmse_increase"].max())
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">Permutation Feature Importance</text>',
        f'<line x1="{pad_l}" y1="{height-pad_b}" x2="{width-pad_r}" y2="{height-pad_b}" stroke="#111827" stroke-width="1.2"/>',
    ]
    for i, (_, row) in enumerate(data.iterrows()):
        y = pad_t + i * row_h + row_h * 0.18
        bar_h = row_h * 0.64
        v = float(row["rmse_increase"])
        bar_w = v / max_v * plot_w
        label = str(row["feature"])
        parts.append(f'<text x="{pad_l-12}" y="{y+bar_h*0.68:.1f}" text-anchor="end" font-family="Arial" font-size="13">{label}</text>')
        parts.append(f'<rect x="{pad_l}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="#0f766e"/>')
        parts.append(f'<text x="{pad_l+bar_w+7:.1f}" y="{y+bar_h*0.68:.1f}" font-family="Arial" font-size="12">{v:.4f}</text>')
    parts.append(f'<text x="{pad_l + plot_w/2}" y="{height-16}" text-anchor="middle" font-family="Arial" font-size="14">RMSE increase after permutation</text>')
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train baseline SOH models on NASA features.")
    parser.add_argument("--features", default="data/processed/nasa_soh_features.csv")
    parser.add_argument("--test-battery", default=None)
    args = parser.parse_args()

    feature_path = Path(args.features)
    if not feature_path.exists():
        print(f"Feature table not found: {feature_path}")
        print("Run: python src/prepare_nasa_soh.py")
        return 1

    df = pd.read_csv(feature_path)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[TARGET])
    train_df, test_df, test_battery = split_by_battery(df, args.test_battery)
    cols = feature_columns(df)

    x_train_raw = train_df[cols].to_numpy(dtype=float)
    y_train = train_df[TARGET].to_numpy(dtype=float)
    x_test_raw = test_df[cols].to_numpy(dtype=float)
    y_test = test_df[TARGET].to_numpy(dtype=float)

    scaler = StandardScalerLite().fit(x_train_raw)
    x_train = scaler.transform(x_train_raw)
    x_test = scaler.transform(x_test_raw)

    preds = fit_predict_models(x_train, y_train, x_test)
    metric_rows = []
    pred_df = test_df[["battery_id", "discharge_cycle_index"]].copy()
    pred_df["y_true"] = y_test
    for name, y_pred in preds.items():
        metric_rows.append({"model": name, "test_battery": test_battery, **metrics(y_test, y_pred)})
        pred_df[name] = y_pred

    out_tables = Path("results/tables")
    out_tables.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(metric_rows).sort_values("RMSE")
    metrics_df.to_csv(out_tables / "baseline_metrics.csv", index=False, encoding="utf-8-sig")
    pred_df.to_csv(out_tables / "baseline_predictions.csv", index=False, encoding="utf-8-sig")

    # Interpretability proxy: permutation importance for the best baseline model.
    best_name = str(metrics_df.iloc[0]["model"])
    best_model: Any
    if best_name == "Ridge":
        best_model = RidgeRegressorLite(alpha=1.0).fit(x_train, y_train)
    elif best_name == "KNN":
        best_model = KNNRegressorLite(k=7).fit(x_train, y_train)
    elif best_name == "GradientBoostingLite":
        best_model = GradientBoostingRegressorLite(n_estimators=120, learning_rate=0.04, max_depth=3, min_samples_leaf=5).fit(x_train, y_train)
    else:
        best_model = RandomForestRegressorLite(n_estimators=80, max_depth=6, min_samples_leaf=5).fit(x_train, y_train)
    importance = permutation_importance(best_model, x_test, y_test, cols)
    importance.to_csv(out_tables / "permutation_importance.csv", index=False, encoding="utf-8-sig")

    save_prediction_svg(pred_df, Path("results/figures/baseline_soh_prediction.svg"), test_battery)
    save_importance_svg(importance, Path("results/figures/permutation_importance.svg"))
    print("Saved metrics: results/tables/baseline_metrics.csv")
    print("Saved predictions: results/tables/baseline_predictions.csv")
    print("Saved importance: results/tables/permutation_importance.csv")
    print("Saved figure: results/figures/baseline_soh_prediction.svg")
    print("Saved figure: results/figures/permutation_importance.svg")
    print(metrics_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
