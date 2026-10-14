"""Per-dataset verdict table: winner, tie accounting, ensemble CI, deployable protocol.

    python scripts/report_verdicts.py --size 28
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RESULT_DIR  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", default="28")
    args = parser.parse_args()

    rows = []
    for path in sorted(glob.glob(str(RESULT_DIR / f"*_resnet18_{args.size}_summary.json"))):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        dataset = os.path.basename(path).split(f"_resnet18_{args.size}")[0]
        advantage = payload.get("ensemble_advantage") or {}
        rows.append({
            "dataset": dataset,
            "best_oracle": payload["best_method"],
            "cov_at_0.1": payload["best_coverage"],
            "verdict": payload["best_method_verdict"],
            "n_tied": len(payload["tied_methods"]),
            "ens_delta": advantage.get("delta"),
            "ens_ci": (f"[{advantage['ci_low']:+.3f},{advantage['ci_high']:+.3f}]"
                       if advantage else "n/a"),
            "test_risk_val_selected": payload.get("operating_point", {}).get("risk"),
        })

    header = (f"{'dataset':16s} {'best(oracle)':14s} {'cov@0.1':>8s} {'verdict':22s} "
              f"{'ties':>4s} {'ens-vs-best-single':>22s}")
    print(header)
    print("-" * len(header))
    for row in rows:
        delta = "n/a" if row["ens_delta"] is None else f"{row['ens_delta']:+.3f} {row['ens_ci']}"
        print(f"{row['dataset']:16s} {row['best_oracle']:14s} {row['cov_at_0.1']:8.3f} "
              f"{row['verdict']:22s} {row['n_tied']:4d} {delta:>22s}")

    strict = sum(1 for r in rows if r["verdict"] == "ensemble_strictly_best")
    tied = sum(1 for r in rows if r["verdict"] == "ensemble_tied_for_best")
    other = sum(1 for r in rows if r["verdict"] == "another_method_better")
    print(f"\nensemble strictly best: {strict}/{len(rows)} | tied for best: {tied} | "
          f"another method better: {other}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
