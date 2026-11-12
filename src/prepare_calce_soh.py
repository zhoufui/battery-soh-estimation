from __future__ import annotations

import argparse
import math
import re
import sys
import zipfile
from datetime import date
from pathlib import Path
from typing import Any

PROJECT_PACKAGES = Path(__file__).resolve().parents[1] / ".python-packages"
if PROJECT_PACKAGES.exists():
    sys.path.insert(0, str(PROJECT_PACKAGES))

import numpy as np
import pandas as pd


def maybe_extract_zip(raw_dir: Path) -> None:
    for zip_path in sorted(raw_dir.glob("*.zip")):
        extract_dir = raw_dir / zip_path.stem
        if extract_dir.exists() and list(extract_dir.rglob("*.xlsx")):
            continue
        print(f"Extracting {zip_path.name} -> {extract_dir}")
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)


def norm_col(col: Any) -> str:
    text = str(col).strip().lower()
    text = re.sub(r"[\s_\-]+", "", text)
    text = text.replace("(", "").replace(")", "").replace("[", "").replace("]", "")
    return text


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lookup = {norm_col(c): c for c in df.columns}
    for cand in candidates:
        key = norm_col(cand)
        if key in lookup:
            return lookup[key]
    for col in df.columns:
        key = norm_col(col)
        if any(norm_col(cand) in key for cand in candidates):
            return col
    return None


def series(df: pd.DataFrame, col: str | None) -> np.ndarray:
    if col is None or col not in df.columns:
        return np.array([], dtype=float)
    arr = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def scalar_max(df: pd.DataFrame, col: str | None) -> float:
    arr = series(df, col)
    if arr.size == 0:
        return math.nan
    return float(np.nanmax(arr))


def cycle_capacity(df: pd.DataFrame, col: str | None) -> float:
    arr = series(df, col)
    if arr.size == 0:
        return math.nan
    cap_max = float(np.nanmax(arr))
    cap_min = float(np.nanmin(arr))
    # Arbin exports may store cumulative discharge capacity across many cycles.
    # The per-cycle capacity is therefore the within-cycle increment.
    increment = cap_max - cap_min
    if increment > 0:
        return increment
    return cap_max


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


def area(time_s: np.ndarray, values: np.ndarray) -> float:
    n = min(time_s.size, values.size)
    if n < 2:
        return math.nan
    integrator = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    try:
        return float(integrator(values[:n], time_s[:n]))
    except Exception:
        return math.nan


def battery_id_from_path(path: Path) -> str:
    for part in [path.stem, *[p.name for p in path.parents]]:
        m = re.search(r"(CS2[_\-]?\d+)", part, flags=re.I)
        if m:
            return m.group(1).replace("-", "_").upper()
    return path.stem.split("_")[0].upper()


def date_key_from_file(name: str) -> str:
    m = re.search(r"_(\d{1,2})_(\d{1,2})_(\d{2})", name)
    if not m:
        return "9999-12-31"
    month, day, year = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    year += 2000 if year < 70 else 1900
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return "9999-12-31"


def read_excel(path: Path) -> pd.DataFrame:
    try:
        sheets = pd.read_excel(path, engine="openpyxl", sheet_name=None)
    except Exception as exc:
        raise RuntimeError(f"Failed reading {path}: {exc}") from exc
    for _, df in sheets.items():
        keys = {norm_col(c) for c in df.columns}
        if {"cycleindex", "voltagev", "currenta"}.issubset(keys):
            return df
    # Fall back to the first non-empty sheet for better debug output.
    for _, df in sheets.items():
        if not df.empty:
            return df
    return pd.DataFrame()


def process_excel(path: Path) -> pd.DataFrame:
    df = read_excel(path)
    return process_frame(df, path)


def process_frame(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Extract aligned phases using current, not positive cumulative counters."""
    if df.empty:
        return pd.DataFrame()

    cycle_col = find_column(df, ["Cycle_Index", "CycleIndex", "Cycle"])
    voltage_col = find_column(df, ["Voltage(V)", "Voltage"])
    current_col = find_column(df, ["Current(A)", "Current"])
    test_time_col = find_column(df, ["Test_Time(s)", "TestTime"])
    discharge_cap_col = find_column(df, ["Discharge_Capacity(Ah)", "DischargeCapacity"])
    resistance_col = find_column(df, ["Internal_Resistance(Ohm)", "InternalResistance", "Resistance"])

    if cycle_col is None or discharge_cap_col is None or voltage_col is None or current_col is None:
        print(f"Skipping {path.name}: required columns not found")
        print("Columns:", list(df.columns)[:20])
        return pd.DataFrame()

    if test_time_col is None:
        raise ValueError(f"{path}: cumulative Test_Time(s) is required; Step_Time can reset between steps")

    rows: list[dict[str, Any]] = []
    battery_id = battery_id_from_path(path)
    for cycle_value, group in df.groupby(cycle_col):
        # Include the pre-discharge counter baseline. The first negative-current
        # sample already includes charge passed during one logging interval.
        capacity = cycle_capacity(group, discharge_cap_col)
        group = group.copy()
        aligned_cols = [test_time_col, voltage_col, current_col]
        group[aligned_cols] = group[aligned_cols].apply(pd.to_numeric, errors="coerce")
        group = group.replace([np.inf, -np.inf], np.nan).dropna(subset=aligned_cols)
        group = group.sort_values(test_time_col, kind="stable").drop_duplicates(test_time_col)
        discharge_group = group[group[current_col] < -0.01].copy()
        if discharge_group.empty:
            continue

        if not math.isfinite(capacity) or capacity <= 0:
            continue

        time_col = test_time_col
        time_s = series(discharge_group, time_col)
        voltage = series(discharge_group, voltage_col)
        current = series(discharge_group, current_col)
        resistance = series(group, resistance_col)

        charge_group = group[
            (group[current_col] > 0.01)
            & (group[time_col] < discharge_group[time_col].iloc[0])
        ].copy()
        charge_time = series(charge_group, time_col)
        charge_voltage = series(charge_group, voltage_col)
        charge_current = series(charge_group, current_col)

        row: dict[str, Any] = {
            "battery_id": battery_id,
            "source_file": path.name,
            "source_date": date_key_from_file(path.name),
            "source_cycle_index": cycle_value,
            "capacity_Ah": capacity,
            "duration_s": float(time_s[-1] - time_s[0]) if time_s.size >= 2 else math.nan,
            "voltage_time_area": area(time_s, voltage),
            "current_time_area": area(time_s, current),
            "charge_duration_s": float(charge_time[-1] - charge_time[0]) if charge_time.size >= 2 else math.nan,
            "charge_voltage_time_area": area(charge_time, charge_voltage),
            "charge_current_time_area": area(charge_time, charge_current),
        }
        row.update(stat_features("voltage", voltage))
        row.update(stat_features("current", current))
        row.update(stat_features("internal_resistance", resistance))
        row.update(stat_features("charge_voltage", charge_voltage))
        row.update(stat_features("charge_current", charge_current))
        rows.append(row)
    return pd.DataFrame(rows)


def find_excel_files(raw_dir: Path, batteries: set[str]) -> list[Path]:
    files = sorted(list(raw_dir.rglob("*.xlsx")) + list(raw_dir.rglob("*.xls")))
    if not batteries:
        return files
    selected = []
    for path in files:
        bid = battery_id_from_path(path)
        if bid in batteries:
            selected.append(path)
    return selected


def save_svg(df: pd.DataFrame, out_path: Path, y_col: str, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 920, 560
    pad_l, pad_r, pad_t, pad_b = 75, 35, 55, 70
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    x_min = 1
    x_max = max(2, int(df["cycle_index"].max()))
    y_min = float(df[y_col].min())
    y_max = float(df[y_col].max())
    if y_col == "soh":
        y_min = min(y_min, 0.65)
        y_max = max(y_max, 1.02)

    def sx(x: float) -> float:
        return pad_l + (x - x_min) / max(x_max - x_min, 1) * plot_w

    def sy(y: float) -> float:
        return pad_t + (y_max - y) / max(y_max - y_min, 1e-9) * plot_h

    colors = ["#0f766e", "#b91c1c", "#2563eb", "#a16207", "#7c3aed", "#334155"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">{title}</text>',
        f'<line x1="{pad_l}" y1="{height-pad_b}" x2="{width-pad_r}" y2="{height-pad_b}" stroke="#111827" stroke-width="1.5"/>',
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{height-pad_b}" stroke="#111827" stroke-width="1.5"/>',
    ]
    for idx, (battery, group) in enumerate(df.groupby("battery_id")):
        group = group.sort_values("cycle_index")
        points = []
        for _, row in group.iterrows():
            points.append(f"{sx(float(row['cycle_index'])):.1f},{sy(float(row[y_col])):.1f}")
        color = colors[idx % len(colors)]
        parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2.2"/>')
        parts.append(f'<rect x="{pad_l+15}" y="{pad_t+12+idx*22}" width="14" height="14" fill="{color}"/>')
        parts.append(f'<text x="{pad_l+37}" y="{pad_t+24+idx*22}" font-family="Arial" font-size="13">{battery}</text>')
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare CALCE SOH feature table.")
    parser.add_argument("--raw-dir", default="data/raw/CALCE")
    parser.add_argument("--out", default="data/processed/calce_soh_features.csv")
    parser.add_argument("--batteries", default="CS2_35,CS2_36,CS2_37,CS2_38")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    maybe_extract_zip(raw_dir)
    batteries = {item.strip().upper().replace("-", "_") for item in args.batteries.split(",") if item.strip()}
    files = find_excel_files(raw_dir, batteries)
    if not files:
        print("No CALCE Excel files found.")
        print("Download official CALCE files such as CS2_35.zip into data/raw/CALCE.")
        print("Official page: https://calce.umd.edu/data")
        return 1

    frames = []
    for path in files:
        print(f"Parsing {path}")
        frame = process_excel(path)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        print("No CALCE cycles parsed.")
        return 1

    df = pd.concat(frames, ignore_index=True)
    # Remove duplicate source cycles if the archive contains repeated files.
    df = df.sort_values(["battery_id", "source_date", "source_file", "source_cycle_index"]).reset_index(drop=True)
    output_frames = []
    for battery, group in df.groupby("battery_id"):
        group = group.copy().reset_index(drop=True)
        median_capacity = float(group["capacity_Ah"].median())
        lower = max(0.2, 0.5 * median_capacity)
        upper = 1.5 * median_capacity
        before = len(group)
        group = group[(group["capacity_Ah"] >= lower) & (group["capacity_Ah"] <= upper)].copy()
        removed = before - len(group)
        if removed:
            print(f"{battery}: removed {removed} incomplete/anomalous cycles outside [{lower:.3f}, {upper:.3f}] Ah")
        group = group.reset_index(drop=True)
        group["cycle_index"] = np.arange(1, len(group) + 1)
        reference_capacity = float(group["capacity_Ah"].max())
        group["soh"] = group["capacity_Ah"] / reference_capacity
        group["capacity_retention_pct"] = group["soh"] * 100.0
        output_frames.append(group)
    out = pd.concat(output_frames, ignore_index=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    save_svg(out, Path("results/figures/calce_capacity_fade.svg"), "capacity_Ah", "CALCE Battery Capacity Fade")
    save_svg(out, Path("results/figures/calce_soh_fade.svg"), "soh", "CALCE Battery SOH Fade")
    print(f"Saved feature table: {out_path}")
    print("Saved figures: results/figures/calce_capacity_fade.svg, results/figures/calce_soh_fade.svg")
    print(out.groupby("battery_id").size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
