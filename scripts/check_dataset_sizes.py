"""Check the download size of MedMNIST / MedMNIST+ files before a run.

Kaggle only gives 20 GB of auto-saved disk (/kaggle/working), so check before
requesting size=128 or size=224.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ALL_2D_DATASETS  # noqa: E402


def size_gb(url: str | None) -> float | None:
    if not url:
        return None
    try:
        response = requests.head(url, allow_redirects=True, timeout=30)
        length = int(response.headers.get("content-length", 0))
        return length / 1e9 if length else None
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(ALL_2D_DATASETS))
    args = parser.parse_args()

    from medmnist import INFO

    flags = [f.strip() for f in args.datasets.split(",") if f.strip()]
    print(f"{'dataset':16s} {'28x28':>9s} {'64':>9s} {'128':>9s} {'224':>9s}")
    totals = {}
    for flag in flags:
        sizes = {key: size_gb(INFO[flag].get(f"url{key}")) for key in ("", "_64", "_128", "_224")}
        cells = " ".join(f"{v:8.2f}G" if v else "     n/a" for v in sizes.values())
        print(f"{flag:16s} {cells}")
        for key, value in sizes.items():
            if value:
                totals[key] = totals.get(key, 0.0) + value
    print("-" * 56)
    summary = " ".join(f"{key or '28'}={value:6.2f}G" for key, value in totals.items())
    print(f"total: {summary}")
    print("\nnote: Kaggle /kaggle/working is 20 GB; each size variant is downloaded separately.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
