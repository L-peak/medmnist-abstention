"""Merge a Kaggle download folder into the local analysis layout.

Copies logit bundles into outputs/logits/ and per-run metrics into outputs/results/
so run_selective.py / summarize_results.py can pick them up.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import LOGIT_DIR, RESULT_DIR  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True,
                        help="download folder, e.g. outputs/kaggle_download_medmnist-abstention-128")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        raise SystemExit(f"source not found: {source}")
    roots = list(source.glob("*/outputs")) + list(source.glob("outputs"))
    if not roots:
        raise SystemExit(f"no */outputs found under {source}")
    root = roots[0]

    copied = {"logits": 0, "results": 0}
    for bundle in sorted((root / "logits").glob("*.npz")):
        target = LOGIT_DIR / bundle.name
        copied["logits"] += 1
        if not args.dry_run:
            shutil.copy2(bundle, target)
    for metrics in sorted((root / "results").glob("*_seed*_baseline.json")):
        target = RESULT_DIR / metrics.name
        copied["results"] += 1
        if not args.dry_run:
            shutil.copy2(metrics, target)

    verb = "would copy" if args.dry_run else "copied"
    print(f"{verb} {copied['logits']} bundle(s) -> {LOGIT_DIR}")
    print(f"{verb} {copied['results']} metrics file(s) -> {RESULT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
