"""Plot risk-coverage curves from the npz written by run_selective.py."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import FIG_DIR  # noqa: E402
from src.plotting import plot_risk_coverage  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", required=True, help="e.g. outputs/results/pathmnist_resnet18_28_risk_coverage.npz")
    parser.add_argument("--out", default=None, help="output image path (default: outputs/figures/<stem>_risk_coverage.png)")
    parser.add_argument("--title", default=None)
    parser.add_argument("--max-coverage", type=float, default=1.0)
    args = parser.parse_args()

    with np.load(args.npz, allow_pickle=False) as data:
        curves = {key: data[key] for key in data.files}
    out = Path(args.out) if args.out else FIG_DIR / (Path(args.npz).stem + ".png")
    plot_risk_coverage(curves, out, title=args.title, max_coverage=args.max_coverage)
    print(f"saved figure: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
