from pathlib import Path
import math

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures_png"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#17becf",
    "#8c564b",
]


def font(size, bold=False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def nice_bounds(values, pad=0.05):
    vals = [float(v) for v in values if pd.notna(v)]
    lo, hi = min(vals), max(vals)
    if math.isclose(lo, hi):
        lo -= 1
        hi += 1
    span = hi - lo
    return lo - span * pad, hi + span * pad


def draw_axes(draw, box, x_label, y_label, title):
    left, top, right, bottom = box
    draw.rectangle([left, top, right, bottom], outline="#333333", width=2)
    draw.text((70, 28), title, fill="#111111", font=font(30, True))
    draw.text(((left + right) // 2 - 80, bottom + 55), x_label, fill="#333333", font=font(20))
    draw.text((20, (top + bottom) // 2 - 20), y_label, fill="#333333", font=font(20))


def map_xy(x, y, x_min, x_max, y_min, y_max, box):
    left, top, right, bottom = box
    px = left + (float(x) - x_min) / (x_max - x_min) * (right - left)
    py = bottom - (float(y) - y_min) / (y_max - y_min) * (bottom - top)
    return px, py


def line_chart(groups, out_path, title, x_label="Cycle", y_label="SOH", named_colors=None):
    img = Image.new("RGB", (1400, 900), "white")
    draw = ImageDraw.Draw(img)
    box = (120, 90, 1280, 650)
    all_x = [x for points in groups.values() for x, _ in points]
    all_y = [y for points in groups.values() for _, y in points]
    x_min, x_max = nice_bounds(all_x, 0.02)
    y_min, y_max = nice_bounds(all_y, 0.08)
    if y_label == "RMSE":
        y_min, y_max = 0.0, max(all_y) * 1.1
    draw_axes(draw, box, x_label, y_label, title)

    for i in range(6):
        y = box[3] - i / 5 * (box[3] - box[1])
        val = y_min + i / 5 * (y_max - y_min)
        draw.line([(box[0], y), (box[2], y)], fill="#e6e6e6", width=1)
        draw.text((50, y - 12), f"{val:.2f}", fill="#555555", font=font(16))

    ticks = sorted(set(all_x)) if len(set(all_x)) <= 8 else [x_min + i / 5 * (x_max - x_min) for i in range(6)]
    for val in ticks:
        x = box[0] + (val - x_min) / (x_max - x_min) * (box[2] - box[0])
        draw.text((x - 16, box[3] + 16), f"{val:.0f}", fill="#555555", font=font(16))

    for idx, (name, points) in enumerate(groups.items()):
        color = (named_colors or {}).get(name, COLORS[idx % len(COLORS)])
        coords = [map_xy(x, y, x_min, x_max, y_min, y_max, box) for x, y in points]
        if len(coords) > 1:
            draw.line(coords, fill=color, width=4)
        for px, py in coords[:: max(1, len(coords) // 16)]:
            draw.ellipse([px - 3, py - 3, px + 3, py + 3], fill=color)
        lx = 130 + (idx % 4) * 285
        ly = 735 + (idx // 4) * 42
        draw.line([(lx, ly + 10), (lx + 34, ly + 10)], fill=color, width=5)
        draw.text((lx + 42, ly), str(name), fill="#222222", font=font(18))

    img.save(out_path)


def bar_chart(labels, values, out_path, title, y_label="RMSE"):
    img = Image.new("RGB", (1400, 900), "white")
    draw = ImageDraw.Draw(img)
    box = (130, 90, 1280, 710)
    y_min = 0
    y_max = max(values) * 1.18
    draw_axes(draw, box, "", y_label, title)
    for i in range(6):
        y = box[3] - i / 5 * (box[3] - box[1])
        val = y_min + i / 5 * (y_max - y_min)
        draw.line([(box[0], y), (box[2], y)], fill="#e6e6e6", width=1)
        draw.text((55, y - 12), f"{val:.3f}", fill="#555555", font=font(15))

    gap = 20
    bar_w = (box[2] - box[0] - gap * (len(labels) + 1)) / len(labels)
    for idx, (label, value) in enumerate(zip(labels, values)):
        x0 = box[0] + gap + idx * (bar_w + gap)
        x1 = x0 + bar_w
        y0 = box[3] - (value - y_min) / (y_max - y_min) * (box[3] - box[1])
        color = COLORS[idx % len(COLORS)]
        draw.rectangle([x0, y0, x1, box[3]], fill=color)
        draw.text((x0 + 6, y0 - 28), f"{value:.4f}", fill="#222222", font=font(16))
        draw.text((x0, box[3] + 18), label.replace("_", "\n"), fill="#222222", font=font(15))

    img.save(out_path)


def hbar_chart(labels, values, out_path, title, x_label="Mean |SHAP|"):
    img = Image.new("RGB", (1400, 900), "white")
    draw = ImageDraw.Draw(img)
    draw.text((70, 28), title, fill="#111111", font=font(30, True))
    left, top, right, bottom = 390, 95, 1280, 775
    max_v = max(values) * 1.12
    row_h = (bottom - top) / len(labels)
    for idx, (label, value) in enumerate(zip(labels, values)):
        y = top + idx * row_h + 8
        w = value / max_v * (right - left)
        draw.text((70, y + 8), label, fill="#222222", font=font(18))
        draw.rectangle([left, y, left + w, y + row_h * 0.6], fill=COLORS[idx % len(COLORS)])
        draw.text((left + w + 10, y + 8), f"{value:.4f}", fill="#333333", font=font(16))
    draw.line([(left, bottom + 10), (right, bottom + 10)], fill="#333333", width=2)
    draw.text(((left + right) // 2 - 60, bottom + 40), x_label, fill="#333333", font=font(18))
    img.save(out_path)


def main():
    nasa = pd.read_csv(ROOT / "data" / "processed" / "nasa_soh_features.csv")
    groups = {}
    for battery, part in nasa.groupby("battery_id"):
        groups[battery] = list(zip(part["discharge_cycle_index"], part["capacity_Ah"]))
    line_chart(groups, OUT / "nasa_capacity_fade.png", "NASA Capacity Degradation", y_label="Capacity (Ah)")

    groups = {}
    for battery, part in nasa.groupby("battery_id"):
        groups[battery] = list(zip(part["discharge_cycle_index"], part["soh"]))
    line_chart(groups, OUT / "nasa_soh_fade.png", "NASA SOH Trajectories", y_label="SOH")

    pred = pd.read_csv(ROOT / "results" / "tables" / "standard_model_predictions.csv")
    groups = {
        "True": list(zip(pred["discharge_cycle_index"], pred["y_true"])),
        "RandomForest": list(zip(pred["discharge_cycle_index"], pred["RandomForest"])),
        "XGBoost": list(zip(pred["discharge_cycle_index"], pred["XGBoost"])),
        "LightGBM": list(zip(pred["discharge_cycle_index"], pred["LightGBM"])),
        "SVR_RBF": list(zip(pred["discharge_cycle_index"], pred["SVR_RBF"])),
    }
    line_chart(groups, OUT / "standard_model_predictions.png", "B0018 SOH Prediction", y_label="SOH")

    cross = pd.read_csv(ROOT / "results" / "tables" / "standard_cross_battery_summary.csv")
    bar_chart(cross["model"].tolist(), cross["RMSE"].tolist(), OUT / "standard_cross_battery_rmse.png", "NASA Cross-Battery RMSE")

    adapt = pd.read_csv(ROOT / "results" / "tables" / "adaptation_controls_metrics.csv")
    adapt = adapt[adapt["model"] == "GradientBoosting"]
    names = {"source_only": "Source only", "target_only": "Target only", "source_plus_target": "Source + target"}
    groups = {names[mode]: list(zip(group["target_fraction"] * 100, group["RMSE"]))
              for mode, group in adapt.groupby("training_mode")}
    line_chart(groups, OUT / "representative_domain_adaptation_rmse.png",
               "Target Sample Controls with Gradient Boosting", x_label="Target samples (%)", y_label="RMSE",
               named_colors={"Source only": "#6b7280", "Target only": "#2563eb", "Source + target": "#b91c1c"})

    audit = pd.read_csv(ROOT / "results" / "tables" / "feature_proxy_ablation_summary.csv")
    best_audit = audit.loc[audit.groupby(["dataset", "feature_group"])["RMSE"].idxmin()]
    labels, values = [], []
    for dataset in ["NASA", "CALCE"]:
        for group_name, short_name in [("full_features", "Full"), ("without_duration_or_integrals", "No integrals"), ("preceding_charge_only", "Charge")]:
            row = best_audit[(best_audit["dataset"] == dataset) & (best_audit["feature_group"] == group_name)].iloc[0]
            labels.append(f"{dataset} {short_name}")
            values.append(float(row["RMSE"]))
    bar_chart(labels, values, OUT / "feature_proxy_ablation_rmse.png", "Sensitivity to Feature Availability")

    abl = pd.read_csv(ROOT / "results" / "tables" / "feature_ablation_summary.csv")
    best_abl = abl.sort_values("RMSE").groupby("feature_group", as_index=False).first()
    labels = ["Discharge", "Discharge+Charge", "All"]
    order = ["discharge_only", "discharge_charge", "discharge_charge_impedance"]
    rmse_map = dict(zip(best_abl["feature_group"], best_abl["RMSE"]))
    bar_chart(labels, [rmse_map[k] for k in order], OUT / "feature_ablation_rmse.png", "Feature-Group Ablation")

    shap = pd.read_csv(ROOT / "results" / "tables" / "shap_feature_importance.csv").head(10)
    hbar_chart(shap["feature"].tolist(), shap["mean_abs_shap"].tolist(), OUT / "shap_feature_importance.png", "SHAP Feature Importance")


if __name__ == "__main__":
    main()
