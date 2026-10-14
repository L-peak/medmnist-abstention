"""Build the cross-dataset summary tables from the per-dataset analysis outputs.

Reads  outputs/results/<stem>_selective.csv      (written by run_selective.py)
       outputs/results/<stem>_seed<k>_baseline.json
Writes outputs/results/SUMMARY.md and SUMMARY.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RESULT_DIR  # noqa: E402

METHOD_ORDER = ["msp", "entropy", "temp_scaling", "margin", "mc_dropout",
                "ensemble", "conformal_lac", "conformal_aps"]

STEM_RE = re.compile(r"^(?P<dataset>.+?)_(?P<model>resnet18|smallcnn)_(?P<size>\d+)$")


def parse_stem(path: str):
    """`pathmnist_resnet18_28_selective.csv` -> ("pathmnist", "resnet18", 28)."""
    name = Path(path).name
    if not name.endswith("_selective.csv"):
        return None
    match = STEM_RE.match(name[: -len("_selective.csv")])
    return (match.group("dataset"), match.group("model"), int(match.group("size"))) if match else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", default="*_selective.csv")
    parser.add_argument("--model", default="resnet18", help="which backbone to summarise")
    parser.add_argument("--size", default="28",
                        help="input resolution to summarise (28, 128, ...) or 'all' to keep them separate")
    args = parser.parse_args()

    frames, skipped = [], []
    for path in sorted(glob.glob(str(RESULT_DIR / args.pattern))):
        parsed = parse_stem(path)
        if parsed is None or parsed[1] != args.model or str(parsed[2]) != str(args.size):
            skipped.append(Path(path).name)
            continue
        frame = pd.read_csv(path)
        frame["dataset"] = parsed[0] if args.size != "all" else f"{parsed[0]}_{parsed[2]}"
        frames.append(frame)
    if not frames:
        print("no *_selective.csv found under outputs/results - run run_selective.py first")
        return 1
    if skipped:
        print(f"ignored {len(skipped)} file(s) that are not {args.model} results: {', '.join(skipped)}")
    table = pd.concat(frames, ignore_index=True)

    accuracy = {}
    macro_aucs = {}
    for path in sorted(glob.glob(str(RESULT_DIR / "*_seed*_baseline.json"))):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("model") not in (None, args.model):
            continue
        if str(payload.get("size", 28)) != str(args.size) and args.size != "all":
            continue
        key = payload["dataset"] if args.size != "all" else f"{payload['dataset']}_{payload.get('size', 28)}"
        accuracy.setdefault(key, []).append(payload["test_acc"])
        if payload.get("macro_auc") is not None:
            macro_aucs.setdefault(key, []).append(payload["macro_auc"])

    coverage = (table.pivot_table(index="dataset", columns="method",
                                  values="coverage_at_risk", aggfunc="mean")
                .reindex(columns=[m for m in METHOD_ORDER if m in set(table["method"])]))
    best = (table.sort_values("aurc")
            .groupby("dataset", as_index=False)
            .first()[["dataset", "method", "aurc", "e_aurc", "coverage_at_risk"]]
            .rename(columns={"method": "best_method"}))

    lines = ["# MedMNIST abstention benchmark - summary", ""]
    lines.append("## Test accuracy (mean of seeds)")
    lines.append("")
    lines.append("| dataset | seeds | test acc (mean) |")
    lines.append("| --- | --- | --- |")
    for dataset, accs in sorted(accuracy.items()):
        auc = macro_aucs.get(dataset)
        cell = f"{sum(accs) / len(accs):.3f}"
        if auc:
            cell += f" (subset; macro-AUC {sum(auc) / len(auc):.3f})"
        lines.append(f"| {dataset} | {len(accs)} | {cell} |")
    lines.append("")
    lines.append("Multi-label subsets report subset (exact-match) accuracy plus macro-AUC,")
    lines.append("and their abstention risk is per-sample Hamming error.")
    lines.append("")

    lines.append("## Best method per dataset (lowest AURC)")
    lines.append("")
    lines.append(best.round(4).to_markdown(index=False))
    lines.append("")

    # AURC has known flaws (Traub et al., NeurIPS 2024); report whether switching to
    # AUGRC changes which method wins.
    if "augrc" in table.columns:
        agreement = []
        for dataset, group in table.groupby("dataset"):
            by_aurc = group.loc[group["aurc"].idxmin()]
            by_augrc = group.loc[group["augrc"].idxmin()]
            agreement.append({
                "dataset": dataset,
                "best_by_AURC": by_aurc["method"],
                "best_by_AUGRC": by_augrc["method"],
                "agree": bool(by_aurc["method"] == by_augrc["method"]),
                "augrc_AURC_best": round(float(by_aurc["augrc"]), 4),
                "augrc_AUGRC_best": round(float(by_augrc["augrc"]), 4),
            })
        report = pd.DataFrame(agreement)
        lines.append("## Does the winner change when using AUGRC instead of AURC?")
        lines.append("")
        lines.append(report.to_markdown(index=False))
        lines.append("")
        lines.append(f"Agreement: **{int(report['agree'].sum())}/{len(report)}** datasets "
                     "pick the same best method under both metrics.")
        lines.append("")

    lines.append("## coverage@risk<=0.1 by method (higher is better; 1.000 = safe at full coverage)")
    lines.append("")
    lines.append(coverage.round(3).to_markdown())
    lines.append("")

    summary_md = RESULT_DIR / "SUMMARY.md"
    summary_md.write_text("\n".join(lines), encoding="utf-8")
    table.to_csv(RESULT_DIR / "SUMMARY.csv", index=False)
    print("\n".join(lines[:24]))
    print(f"\nwritten: {summary_md}\nwritten: {RESULT_DIR / 'SUMMARY.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
