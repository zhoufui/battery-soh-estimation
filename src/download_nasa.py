from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path


NASA_BATTERY_URL = "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip"


def download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading: {url}")
    print(f"Output: {out_path}")
    with urllib.request.urlopen(url, timeout=120) as response:
        total = response.headers.get("Content-Length")
        total_int = int(total) if total and total.isdigit() else None
        done = 0
        with out_path.open("wb") as fh:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                if total_int:
                    pct = done / total_int * 100
                    print(f"\r{done / 1024 / 1024:.1f} MB / {total_int / 1024 / 1024:.1f} MB ({pct:.1f}%)", end="")
                else:
                    print(f"\r{done / 1024 / 1024:.1f} MB", end="")
    print("\nDone.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download NASA Battery Data Set.")
    parser.add_argument("--url", default=NASA_BATTERY_URL)
    parser.add_argument(
        "--out",
        default="data/raw/NASA_Battery_Data_Set.zip",
        help="Output zip path.",
    )
    args = parser.parse_args()
    try:
        download(args.url, Path(args.out))
    except Exception as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        print("Manual official URL:", args.url, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

