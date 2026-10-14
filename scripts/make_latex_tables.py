"""Generate the paper's tables from the result CSVs (no hand-typed numbers).

    python scripts/make_latex_tables.py

Writes LaTeX fragments under paper/tables/ that main.tex \\input{}s, so every number
in the manuscript comes straight from outputs/results/*.csv.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RESULT_DIR  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper" / "tables"

METHODS = ["msp", "entropy", "temp_scaling", "margin", "mc_dropout",
           "ensemble", "conformal_lac", "conformal_aps"]
HEADERS = {
    "msp": "MSP", "entropy": "Entr.", "temp_scaling": "Temp.", "margin": "Marg.",
    "mc_dropout": "MC drop.", "ensemble": "\\textbf{Ens.}", "conformal_lac": "LAC",
    "conformal_aps": "APS",
}


def load(size: int) -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(str(RESULT_DIR / f"*_resnet18_{size}_selective.csv"))):
        dataset = os.path.basename(path).split(f"_resnet18_{size}")[0]
        frame = pd.read_csv(path)
        frame["dataset"] = dataset
        frames.append(frame)
    if not frames:
        raise SystemExit(f"no *_resnet18_{size}_selective.csv found")
    return pd.concat(frames, ignore_index=True)


def coverage_table(size: int) -> str:
    table = load(size)
    pivot = (table.pivot_table(index="dataset", columns="method", values="coverage_at_risk")
             .reindex(columns=[m for m in METHODS if m in set(table["method"])]))
    # sort rows by the ensemble result, hardest last
    if "ensemble" in pivot.columns:
        pivot = pivot.sort_values("ensemble", ascending=False)

    columns = list(pivot.columns)
    lines = [
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{l" + "c" * len(columns) + "}",
        "\\toprule",
        "Subset & " + " & ".join(HEADERS[c] for c in columns) + " \\\\",
        "\\midrule",
    ]
    for dataset, row in pivot.iterrows():
        values = row.to_numpy(dtype=float)
        best = np.nanmax(values) if np.isfinite(values).any() else np.nan
        cells = []
        for value in values:
            if not np.isfinite(value):
                cells.append("n/a")
            elif np.isfinite(best) and abs(value - best) < 1e-9:
                cells.append(f"\\textbf{{{value:.3f}}}")
            else:
                cells.append(f"{value:.3f}")
        lines.append(f"\\dataset{{{dataset}}} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\normalsize", ""]
    return "\n".join(lines)


def resolution_table() -> str:
    accuracy = {}
    for size in (28, 128):
        for path in glob.glob(str(RESULT_DIR / f"*_resnet18_{size}_seed*_baseline.json")):
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            accuracy.setdefault((payload["dataset"], size), []).append(payload["test_acc"])
    table28, table128 = load(28), load(128)

    def best(size: int, dataset: str):
        frame = table128 if size == 128 else table28
        rows = frame[frame["dataset"] == dataset]
        if rows.empty:
            return "n/a", "n/a"
        row = rows.sort_values("aurc").iloc[0]
        return row["method"], f"{row['coverage_at_risk']:.3f}"

    datasets = sorted(set(table128["dataset"]))
    lines = [
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Subset & Acc $28^2$ & Acc $128^2$ & Best by AURC at $128^2$ & "
        "Cov@0.1 $28^2$ & Cov@0.1 $128^2$ \\\\",
        "\\midrule",
    ]
    for dataset in datasets:
        acc28 = np.mean(accuracy.get((dataset, 28), [np.nan]))
        acc128 = np.mean(accuracy.get((dataset, 128), [np.nan]))
        _, cov28 = best(28, dataset)
        method128, cov128 = best(128, dataset)
        method128 = method128.replace("conformal_", "conformal ").replace("_", " ")
        acc_cells = (f"{acc28:.3f}" if np.isfinite(acc28) else "n/a",
                     f"{acc128:.3f}" if np.isfinite(acc128) else "n/a")
        lines.append(f"\\dataset{{{dataset}}} & {acc_cells[0]} & {acc_cells[1]} & "
                     f"{method128} & {cov28} & \\textbf{{{cov128}}} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\normalsize", ""]
    return "\n".join(lines)


def sanity_table() -> str:
    """Backbone sanity check against the published MedMNIST v2 ResNet-18 (28) numbers.

    Published values live in paper/baselines_medmnist_v2.csv (transcribed from
    Table 3 of the MedMNIST v2 paper, arXiv:2110.14795v2); our side is recomputed
    from the baseline JSONs so the table can never drift from the runs.
    """
    ours: dict[str, dict] = {}
    for path in glob.glob(str(RESULT_DIR / "*_resnet18_28_seed*_baseline.json")):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        entry = ours.setdefault(payload["dataset"], {"acc": [], "macro_auc": []})
        entry["acc"].append(float(payload["test_acc"]))
        if payload.get("macro_auc") is not None:
            entry["macro_auc"].append(float(payload["macro_auc"]))

    published = pd.read_csv(ROOT / "paper" / "baselines_medmnist_v2.csv")
    lines = [
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Subset & Ours & Published & Metric & $\\Delta$ \\\\",
        "\\midrule",
    ]
    outside = []
    for _, row in published.iterrows():
        entry = ours.get(row["dataset"])
        if entry is None:
            continue
        if row["metric"] == "auc":
            value = float(np.mean(entry["macro_auc"]))
            metric = "macro-AUC"
        else:
            value = float(np.mean(entry["acc"]))
            metric = "accuracy"
        gap = value - float(row["published"])
        flag = "" if abs(gap) <= 0.03 else "\\textbf{"
        flag_end = "" if abs(gap) <= 0.03 else "}"
        lines.append(f"\\dataset{{{row['dataset']}}} & {flag}{value:.3f}{flag_end} & "
                     f"{row['published']:.3f} & {metric} & {flag}{gap:+.3f}{flag_end} \\\\")
        if abs(gap) > 0.03:
            outside.append(row["dataset"])
    lines += ["\\bottomrule", "\\end{tabular}", "",
              f"% subsets outside the 0.03 tolerance: {', '.join(outside)}", ""]
    return "\n".join(lines)


def deployable_table() -> str:
    """Validation-selected thresholds: does the risk budget transfer to the test split?"""
    rows = []
    for path in sorted(glob.glob(str(RESULT_DIR / "*_resnet18_28_selective.csv"))):
        dataset = os.path.basename(path).split("_resnet18_28")[0]
        frame = pd.read_csv(path)
        if "risk_val_selected" not in frame.columns:
            continue
        frame = frame.dropna(subset=["risk_val_selected"])
        if frame.empty:
            continue
        overshoot = int((frame["risk_val_selected"] > 0.1 + 1e-9).sum())
        median = float(frame["risk_val_selected"].median())
        rows.append((dataset, len(frame), median, overshoot))
    lines = [
        "\\small",
        "\\setlength{\\tabcolsep}{5pt}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Subset & Scores evaluated & Median test risk & Scores over budget \\\\",
        "\\midrule",
    ]
    for dataset, n, median, overshoot in rows:
        lines.append(f"\\dataset{{{dataset}}} & {n} & {median:.3f} & {overshoot}/{n} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\normalsize", ""]
    return "\n".join(lines)


def conformal_table() -> str:
    """Split-conformal diagnostics: empirical coverage against the nominal 1 - alpha."""
    table = load(28)
    frames = {}
    for method in ("conformal_lac", "conformal_aps"):
        rows = table[table["method"] == method].dropna(subset=["empirical_coverage"])
        frames[method] = rows.set_index("dataset")
    datasets = sorted(set(frames["conformal_lac"].index) | set(frames["conformal_aps"].index))
    lines = [
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        " & \\multicolumn{2}{c}{LAC} & \\multicolumn{2}{c}{APS} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
        "Subset & set size & coverage & set size & coverage \\\\",
        "\\midrule",
    ]
    for dataset in datasets:
        cells = []
        for method in ("conformal_lac", "conformal_aps"):
            frame = frames[method]
            if dataset in frame.index:
                size = float(frame.loc[dataset, "mean_set_size"])
                cov = float(frame.loc[dataset, "empirical_coverage"])
                flag = "" if cov >= 0.9 - 1e-9 else "\\textbf{"
                end = "" if cov >= 0.9 - 1e-9 else "}"
                cells += [f"{size:.2f}", f"{flag}{cov:.3f}{end}"]
            else:
                cells += ["n/a", "n/a"]
        lines.append(f"\\dataset{{{dataset}}} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "",
              "% test-split empirical coverage; the nominal target is 0.900", ""]
    return "\n".join(lines)
    rows = []
    for path in sorted(glob.glob(str(RESULT_DIR / "*_resnet18_28_selective.csv"))):
        dataset = os.path.basename(path).split("_resnet18_28")[0]
        frame = pd.read_csv(path)
        if "risk_val_selected" not in frame.columns:
            continue
        frame = frame.dropna(subset=["risk_val_selected"])
        if frame.empty:
            continue
        overshoot = int((frame["risk_val_selected"] > 0.1 + 1e-9).sum())
        worst = float(frame["risk_val_selected"].max())
        median = float(frame["risk_val_selected"].median())
        rows.append((dataset, len(frame), median, worst, overshoot))
    lines = [
        "\\setlength{\\tabcolsep}{5pt}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Subset & Scores evaluated & Median test risk & Scores over budget \\\\",
        "\\midrule",
    ]
    for dataset, n, median, worst, overshoot in rows:
        lines.append(f"\\dataset{{{dataset}}} & {n} & {median:.3f} & {overshoot}/{n} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=28)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "table_sanity.tex": sanity_table(),
        "table_main.tex": coverage_table(28),
        "table_resolution.tex": resolution_table(),
        "table_deployable.tex": deployable_table(),
        "table_conformal.tex": conformal_table(),
    }
    for name, body in outputs.items():
        (OUT_DIR / name).write_text(body, encoding="utf-8")
        print(f"wrote {OUT_DIR / name} ({len(body.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
