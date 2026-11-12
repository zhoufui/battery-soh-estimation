from __future__ import annotations

import argparse
import math
import zipfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


def require_scipy():
    try:
        from scipy.io import loadmat  # type: ignore
    except Exception as exc:
        try:
            from mat5lite import loadmat_lite
        except Exception:
            raise RuntimeError(
                "This script needs scipy or the local mat5lite reader to read NASA .mat files. "
                "Install scipy with: python -m pip install scipy"
            ) from exc
        print("scipy is not installed; using local MATLAB 5 reader.")
        def loadmat(path: str | Path, **_: Any) -> dict[str, Any]:
            return loadmat_lite(path)
        return loadmat
    return loadmat


def maybe_extract_zip(raw_dir: Path) -> None:
    zip_paths = list(raw_dir.glob("*.zip"))
    if not zip_paths:
        return
    if list(raw_dir.rglob("*.mat")):
        return
    for zip_path in zip_paths:
        extract_dir = raw_dir / zip_path.stem
        extract_dir.mkdir(parents=True, exist_ok=True)
        print(f"Extracting {zip_path.name} -> {extract_dir}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)


def as_list(value: Any) -> list[Any]:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return [value.item()]
        return [x for x in value.ravel()]
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def get_field(obj: Any, name: str, default: Any = np.nan) -> Any:
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, np.ndarray) and obj.dtype.names and name in obj.dtype.names:
        return obj[name]
    if isinstance(obj, dict):
        return obj.get(name, default)
    return default


def scalar(value: Any, default: float = math.nan) -> float:
    try:
        arr = np.asarray(value).astype(float).ravel()
        if arr.size == 0:
            return default
        return float(arr[0])
    except Exception:
        return default


def series(value: Any) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        return arr
    except Exception:
        return np.array([], dtype=float)


def stat_features(prefix: str, arr: np.ndarray) -> dict[str, float]:
    if arr.size == 0:
        return {
            f"{prefix}_mean": math.nan,
            f"{prefix}_min": math.nan,
            f"{prefix}_max": math.nan,
            f"{prefix}_std": math.nan,
            f"{prefix}_range": math.nan,
        }
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_max": float(np.max(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_range": float(np.max(arr) - np.min(arr)),
    }


def curve_area(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return math.nan
    n = min(x.size, y.size)
    try:
        integrator = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        return float(integrator(y[:n], x[:n]))
    except Exception:
        return math.nan


def time_series_features(data: Any, prefix: str = "") -> dict[str, float]:
    time_s = series(get_field(data, "Time", []))
    voltage = series(get_field(data, "Voltage_measured", []))
    current = series(get_field(data, "Current_measured", []))
    temp = series(get_field(data, "Temperature_measured", []))
    p = f"{prefix}_" if prefix else ""
    features: dict[str, float] = {
        f"{p}duration_s": float(time_s[-1] - time_s[0]) if time_s.size >= 2 else math.nan,
        f"{p}voltage_time_area": curve_area(time_s, voltage),
        f"{p}current_time_area": curve_area(time_s, current),
        f"{p}temperature_time_area": curve_area(time_s, temp),
    }
    features.update(stat_features(f"{p}voltage", voltage))
    features.update(stat_features(f"{p}current", current))
    features.update(stat_features(f"{p}temperature", temp))
    return features


def parse_battery_mat(path: Path, loadmat) -> pd.DataFrame:
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    keys = [k for k in mat if not k.startswith("__")]
    if not keys:
        raise ValueError(f"No MATLAB variables found in {path}")

    # NASA files usually store one variable named like B0005.
    variable_name = path.stem if path.stem in mat else keys[0]
    battery_obj = mat[variable_name]
    cycles = as_list(get_field(battery_obj, "cycle", []))
    rows: list[dict[str, Any]] = []
    discharge_index = 0
    last_charge_features: dict[str, float] = {}
    last_impedance_features: dict[str, float] = {}

    for raw_index, cycle in enumerate(cycles, start=1):
        cycle_type = str(get_field(cycle, "type", "")).strip().lower()
        data = get_field(cycle, "data", None)
        if data is None:
            continue

        if cycle_type == "charge":
            last_charge_features = time_series_features(data, prefix="charge")
            continue

        if cycle_type == "impedance":
            last_impedance_features = {
                "impedance_re_ohm": scalar(get_field(data, "Re", math.nan)),
                "impedance_rct_ohm": scalar(get_field(data, "Rct", math.nan)),
            }
            continue

        if cycle_type != "discharge":
            continue

        discharge_index += 1
        capacity = scalar(get_field(data, "Capacity", math.nan))

        row: dict[str, Any] = {
            "battery_id": variable_name,
            "raw_cycle_index": raw_index,
            "discharge_cycle_index": discharge_index,
            "capacity_Ah": capacity,
        }
        row.update(time_series_features(data))
        row.update(last_charge_features)
        row.update(last_impedance_features)
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    first_capacity = float(df["capacity_Ah"].dropna().iloc[0])
    df["soh"] = df["capacity_Ah"] / first_capacity
    df["capacity_retention_pct"] = df["soh"] * 100.0
    return df


def find_mat_files(raw_dir: Path, batteries: Iterable[str] | None = None) -> list[Path]:
    files = sorted(raw_dir.rglob("*.mat"))
    selected = {b.strip().upper() for b in batteries or [] if b.strip()}
    found: dict[str, Path] = {}
    for path in files:
        stem = path.stem.upper()
        if not stem.startswith("B"):
            continue
        if selected and stem not in selected:
            continue
        found.setdefault(stem, path)
    return [found[k] for k in sorted(found)]


def save_svg_capacity(df: pd.DataFrame, out_path: Path, y_col: str, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 920, 560
    pad_l, pad_r, pad_t, pad_b = 75, 35, 55, 70
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    x_min = 1
    x_max = max(2, int(df["discharge_cycle_index"].max()))
    y_min = float(df[y_col].min())
    y_max = float(df[y_col].max())
    if y_col == "soh":
        y_min = min(y_min, 0.65)
        y_max = max(y_max, 1.02)
    if y_max <= y_min:
        y_max = y_min + 1.0

    def sx(x: float) -> float:
        return pad_l + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y: float) -> float:
        return pad_t + (y_max - y) / (y_max - y_min) * plot_h

    colors = ["#0f766e", "#b91c1c", "#2563eb", "#a16207", "#7c3aed", "#334155"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">{title}</text>',
        f'<line x1="{pad_l}" y1="{height-pad_b}" x2="{width-pad_r}" y2="{height-pad_b}" stroke="#111827" stroke-width="1.5"/>',
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{height-pad_b}" stroke="#111827" stroke-width="1.5"/>',
    ]
    for i in range(6):
        y = y_min + (y_max - y_min) * i / 5
        yy = sy(y)
        parts.append(f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{width-pad_r}" y2="{yy:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{pad_l-10}" y="{yy+4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{y:.2f}</text>')
    for i in range(6):
        x = x_min + (x_max - x_min) * i / 5
        xx = sx(x)
        parts.append(f'<line x1="{xx:.1f}" y1="{height-pad_b}" x2="{xx:.1f}" y2="{height-pad_b+6}" stroke="#111827"/>')
        parts.append(f'<text x="{xx:.1f}" y="{height-pad_b+24}" text-anchor="middle" font-family="Arial" font-size="12">{int(round(x))}</text>')

    legend_x = pad_l + 15
    legend_y = pad_t + 12
    for idx, (battery, group) in enumerate(df.groupby("battery_id")):
        group = group.sort_values("discharge_cycle_index")
        points = []
        for _, row in group.iterrows():
            x = float(row["discharge_cycle_index"])
            y = float(row[y_col])
            if math.isfinite(x) and math.isfinite(y):
                points.append(f"{sx(x):.1f},{sy(y):.1f}")
        if not points:
            continue
        color = colors[idx % len(colors)]
        parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2.2"/>')
        parts.append(f'<rect x="{legend_x}" y="{legend_y + idx*22}" width="14" height="14" fill="{color}"/>')
        parts.append(f'<text x="{legend_x+22}" y="{legend_y + idx*22 + 12}" font-family="Arial" font-size="13">{battery}</text>')

    parts.append(f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-family="Arial" font-size="14">Discharge cycle index</text>')
    ylabel = "SOH" if y_col == "soh" else "Capacity (Ah)"
    parts.append(f'<text x="20" y="{height/2}" transform="rotate(-90 20,{height/2})" text-anchor="middle" font-family="Arial" font-size="14">{ylabel}</text>')
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare NASA Battery SOH feature table.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out", default="data/processed/nasa_soh_features.csv")
    parser.add_argument("--batteries", default="B0005,B0006,B0007,B0018")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_path = Path(args.out)
    maybe_extract_zip(raw_dir)

    batteries = [item.strip() for item in args.batteries.split(",") if item.strip()]
    mat_files = find_mat_files(raw_dir, batteries)
    if not mat_files:
        print("No NASA .mat files found.")
        print("Put NASA_Battery_Data_Set.zip in data/raw, or extract B0005.mat/B0006.mat/etc under data/raw.")
        return 1

    loadmat = require_scipy()
    frames = []
    for path in mat_files:
        print(f"Parsing {path}")
        frame = parse_battery_mat(path, loadmat)
        if not frame.empty:
            frames.append(frame)

    if not frames:
        print("No discharge cycles were parsed.")
        return 1

    df = pd.concat(frames, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    save_svg_capacity(df, Path("results/figures/nasa_capacity_fade.svg"), "capacity_Ah", "NASA Battery Capacity Fade")
    save_svg_capacity(df, Path("results/figures/nasa_soh_fade.svg"), "soh", "NASA Battery SOH Fade")
    print(f"Saved feature table: {out_path}")
    print("Saved figures: results/figures/nasa_capacity_fade.svg, results/figures/nasa_soh_fade.svg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
