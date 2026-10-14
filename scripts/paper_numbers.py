"""Consolidate every number quoted in the manuscript into paper/NUMBERS.md.

    python scripts/paper_numbers.py

The file is a build artefact: the manuscript's tables \\input{} the generated LaTeX
fragments, and the prose numbers are copied from this dump, so the audit trail runs
outputs/results -> this dump -> manuscript.
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
PUBLISHED = ROOT / "paper" / "baselines_medmnist_v2.csv"

METHODS = ["msp", "entropy", "temp_scaling", "margin", "mc_dropout",
           "ensemble", "conformal_lac", "conformal_aps"]
BINARY = ["pneumoniamnist", "breastmnist"]


def load_size(size: int) -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(str(RESULT_DIR / f"*_resnet18_{size}_selective.csv"))):
        dataset = os.path.basename(path).split(f"_resnet18_{size}")[0]
        frame = pd.read_csv(path)
        frame["dataset"] = dataset
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_summaries(size: int) -> dict[str, dict]:
    out = {}
    for path in sorted(glob.glob(str(RESULT_DIR / f"*_resnet18_{size}_summary.json"))):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        dataset = os.path.basename(path).split(f"_resnet18_{size}")[0]
        out[dataset] = payload
    return out


def load_accuracy(size: int) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in sorted(glob.glob(str(RESULT_DIR / f"*_resnet18_{size}_seed*_baseline.json"))):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        entry = out.setdefault(payload["dataset"], {"acc": [], "macro_auc": [], "seeds": 0})
        entry["acc"].append(payload["test_acc"])
        entry["seeds"] += 1
        if payload.get("macro_auc") is not None:
            entry["macro_auc"].append(payload["macro_auc"])
    return out


def section(lines: list[str], title: str) -> None:
    lines.append(f"\n## {title}\n")


def sanity_table(lines: list[str]) -> None:
    ours = load_accuracy(28)
    published = pd.read_csv(PUBLISHED)
    section(lines, "Table 2 - backbone sanity check (28x28, mean of 3 seeds)")
    lines.append("| subset | ours | published ResNet-18 (28) | source metric | gap | within 0.03 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    gaps = []
    for _, row in published.iterrows():
        dataset = row["dataset"]
        entry = ours.get(dataset)
        if entry is None:
            continue
        if row["metric"] == "auc":
            value = float(np.mean(entry["macro_auc"])) if entry["macro_auc"] else float("nan")
            label = "macro-AUC"
        else:
            value = float(np.mean(entry["acc"]))
            label = "accuracy"
        gap = value - row["published"]
        gaps.append((dataset, label, value, row["published"], gap, abs(gap) <= 0.03))
        lines.append(f"| {dataset} | {value:.3f} | {row['published']:.3f} | {label} | "
                     f"{gap:+.3f} | {'yes' if abs(gap) <= 0.03 else 'NO'} |")
    inside = sum(1 for g in gaps if g[5])
    lines.append(f"\nwithin 0.03: {inside}/{len(gaps)}; "
                 f"outside: {', '.join(f'{g[0]} {g[4]:+.3f}' for g in gaps if not g[5])}")
    for dataset, label, value, pub, gap, ok in gaps:
        if not ok:
            lines.append(f"- {dataset}: ours {value:.3f} ({label}) vs published {pub:.3f}, gap {gap:+.3f}")
    lines.append("\nown test accuracy per subset (28x28, mean of seeds, from SUMMARY.md):")
    for dataset, entry in sorted(ours.items()):
        extra = (f", macro-AUC {float(np.mean(entry['macro_auc'])):.3f}"
                 if entry["macro_auc"] else "")
        lines.append(f"- {dataset}: {float(np.mean(entry['acc'])):.3f} "
                     f"({entry['seeds']} seeds{extra})")


def coverage_table(lines: list[str], size: int) -> None:
    table = load_size(size)
    pivot = table.pivot_table(index="dataset", columns="method", values="coverage_at_risk")
    section(lines, f"Coverage at risk<=0.1 ({size}x{size})")
    lines.append(pivot.reindex(columns=[m for m in METHODS if m in pivot.columns]).round(3)
                 .to_markdown())
    lines.append("\nper-subset oracle winner, ties, ensemble advantage:")
    summaries = load_summaries(size)
    for dataset, payload in sorted(summaries.items()):
        advantage = payload.get("ensemble_advantage") or {}
        ci = (f"{advantage['delta']:+.3f} [{advantage['ci_low']:+.3f}, {advantage['ci_high']:+.3f}]"
              if advantage else "n/a")
        lines.append(f"- {dataset}: best={payload['best_method']} "
                     f"({payload['best_coverage']:.3f}), verdict={payload['best_method_verdict']}, "
                     f"tied={len(payload['tied_methods'])}, ens-vs-best-single {ci}")
    verdicts = [p["best_method_verdict"] for p in summaries.values()]
    strict = verdicts.count("ensemble_strictly_best")
    tied = verdicts.count("ensemble_tied_for_best")
    lines.append(f"\nstrict {strict}/{len(verdicts)}, tied {tied}, "
                 f"lost {verdicts.count('another_method_better')}")
    significant = [d for d, p in summaries.items()
                   if (p.get("ensemble_advantage") or {}).get("ci_low", -1) > 0]
    lines.append(f"ensemble advantage CI excludes 0 on: {', '.join(sorted(significant))} "
                 f"({len(significant)}/{len(summaries)})")


def augrc_agreement(lines: list[str], size: int) -> None:
    table = load_size(size)
    section(lines, f"AURC vs AUGRC winner agreement ({size}x{size})")
    agree = 0
    for dataset, group in table.groupby("dataset"):
        by_aurc = group.loc[group["aurc"].idxmin()]
        by_augrc = group.loc[group["augrc"].idxmin()]
        same = by_aurc["method"] == by_augrc["method"]
        agree += int(same)
        lines.append(f"- {dataset}: AURC->{by_aurc['method']}, AUGRC->{by_augrc['method']}"
                     f"{'' if same else '  <-- differs'}")
    lines.append(f"\nagreement {agree}/{table['dataset'].nunique()}")


def deployable(lines: list[str], size: int = 28, max_risk: float = 0.1) -> None:
    table = load_size(size)
    section(lines, "Deployable protocol (threshold from validation, evaluated on test)")
    rows = []
    for dataset, group in table.groupby("dataset"):
        usable = group.dropna(subset=["risk_val_selected"])
        if usable.empty:
            continue
        over = usable["risk_val_selected"] > max_risk + 1e-9
        rows.append({
            "dataset": dataset,
            "scores": len(usable),
            "excluded": ", ".join(sorted(set(group["method"]) - set(usable["method"]))),
            "median_test_risk": float(usable["risk_val_selected"].median()),
            "min_test_risk": float(usable["risk_val_selected"].min()),
            "max_test_risk": float(usable["risk_val_selected"].max()),
            "over_budget": int(over.sum()),
            "median_test_coverage": float(usable["coverage_val_selected"].median()),
        })
    frame = pd.DataFrame(rows)
    lines.append(frame.round(3).to_markdown(index=False))
    total = int(frame["scores"].sum())
    over = int(frame["over_budget"].sum())
    lines.append(f"\nscores landing above the {max_risk:.0%} budget on test: {over}/{total} "
                 f"({over / total:.0%}); subsets where every score is over budget: "
                 f"{', '.join(frame.loc[frame['over_budget'] == frame['scores'], 'dataset'])}")
    lines.append(f"subsets where no score is over budget: "
                 f"{', '.join(frame.loc[frame['over_budget'] == 0, 'dataset'])}")


def conformal_stats(lines: list[str], size: int = 28, alpha: float = 0.1) -> None:
    table = load_size(size)
    section(lines, "Conformal diagnostics (marginal coverage and set size)")
    for method in ("conformal_lac", "conformal_aps"):
        rows = table[table["method"] == method].dropna(subset=["empirical_coverage"])
        if rows.empty:
            lines.append(f"{method}: no diagnostics")
            continue
        below = rows.loc[rows["empirical_coverage"] < 1 - alpha - 1e-9, "dataset"].tolist()
        lines.append(f"\n{method}: mean set size "
                     f"{rows['mean_set_size'].mean():.2f}, empirical coverage "
                     f"{rows['empirical_coverage'].mean():.3f} (target >= {1 - alpha:.2f}); "
                     f"below target on {len(below)}/{len(rows)}: {', '.join(sorted(below)) or 'none'}")
        lines.append(rows.set_index("dataset")[["mean_set_size", "empirical_coverage"]]
                     .round(3).to_markdown())


def binary_degeneracy(lines: list[str], size: int = 28) -> None:
    table = load_size(size)
    section(lines, "Binary subsets: how many scores coincide exactly")
    for dataset in BINARY:
        group = table[table["dataset"] == dataset]
        best = group["coverage_at_risk"].max()
        tied = sorted(group.loc[np.isclose(group["coverage_at_risk"], best), "method"])
        lines.append(f"- {dataset}: best coverage {best:.3f}; tied at the optimum: {', '.join(tied)}")
    # distinct coverage values per method tells us which scores are the same ranker
    for dataset in BINARY:
        group = load_size(size)
        frame = group[group["dataset"] == dataset]
        lines.append(f"\n{dataset} - coverage_at_risk and AURC per score:")
        lines.append(frame[["method", "coverage_at_risk", "aurc"]].round(4).to_markdown(index=False))


def futility(lines: list[str], size: int = 28) -> None:
    table = load_size(size)
    summaries = load_summaries(size)
    accuracy = load_accuracy(size)
    section(lines, "Futility regime")
    for dataset in ("retinamnist", "chestmnist"):
        group = table[table["dataset"] == dataset]
        best = group.sort_values("aurc").iloc[0]
        acc = float(np.mean(accuracy[dataset]["acc"])) if dataset in accuracy else float("nan")
        payload = summaries.get(dataset, {})
        lines.append(f"- {dataset}: test accuracy {acc:.3f}; best AURC {best['aurc']:.4f} "
                     f"({best['method']}); best coverage@0.1 {group['coverage_at_risk'].max():.3f}; "
                     f"median coverage across scores {group['coverage_at_risk'].median():.3f}")
        if "exact_match" in json.dumps(payload):
            lines.append(f"  exact-match risk summary present: {json.dumps(payload.get('exact_match'))[:200]}")
    chest = summaries.get("chestmnist", {})
    lines.append(f"\nchestmnist keys: {sorted(chest.keys())}")
    for key in ("risk_at_full_coverage", "exact_match", "hamming"):
        if key in chest:
            lines.append(f"chestmnist.{key} = {json.dumps(chest[key])[:400]}")
    # What if the multi-label risk is defined as exact match instead of Hamming error?
    from src import abstention as abst
    from src.metrics import coverage_at_risk

    bundles = sorted(glob.glob(str(ROOT / "outputs" / "logits"
                                         / "chestmnist_resnet18_28_seed*_logits.npz")))
    if bundles:
        primary = np.load(bundles[0], allow_pickle=False)
        test_logits = primary["test_logits"]
        test_labels = primary["test_labels"].astype(int)
        probs = abst.predict_proba_multilabel(test_logits)
        exact = np.all((probs >= 0.5).astype(int) == test_labels, axis=1)
        mean_probs = [abst.predict_proba_multilabel(np.load(b, allow_pickle=False)["test_logits"])
                      for b in bundles]
        ensemble_score = np.maximum(np.mean(mean_probs, axis=0), 0.0).max(axis=1)
        lines.append(f"\nchestmnist under an exact-match risk definition: exact-match accuracy "
                     f"{exact.mean():.3f}; coverage@0.1 = {coverage_at_risk(ensemble_score, exact, 0.1):.3f} "
                     f"(vs {summaries['chestmnist']['best_coverage']:.3f} under Hamming risk)")


def control_experiment(lines: list[str], size: int = 28, max_risk: float = 0.1) -> None:
    """Step A control: exchangeable halves of the test split (scripts/exchangeability_control.py)."""
    section(lines, "Control experiment: exchangeable halves of the test split")
    path = RESULT_DIR / "exchangeability_control.csv"
    if not path.exists():
        lines.append("not generated yet - run scripts/exchangeability_control.py")
        return
    table = pd.read_csv(path)
    deployable = table[~table["score"].str.startswith("conformal")]
    conformal = table[table["score"].str.startswith("conformal")]
    lines.append(f"- folds: {deployable['split'].nunique()} random splits x 2 directions per subset")
    lines.append(f"- deployable protocol: mean evaluation-half risk "
                 f"{deployable['risk_out'].mean():.3f} (budget {max_risk}), "
                 f"mean risk gap {deployable['risk_out'].mean() - deployable['risk_cal'].mean():+.3f}, "
                 f"folds over budget {(deployable['risk_out'] > max_risk + 1e-9).mean():.2f}")
    for name in ("conformal_lac", "conformal_aps"):
        subset = conformal[conformal["score"] == name]["coverage_out"]
        lines.append(f"- {name}: mean evaluation-half coverage {subset.mean():.3f} "
                     f"(nominal {1 - 0.1:.2f})")
    lines.append("")
    lines.append("per-subset mean evaluation-half risk:")
    lines.append(deployable.groupby("dataset")["risk_out"].mean().round(3).to_markdown())
    lines.append("")
    lines.append("validation -> test risk for comparison (same five scores): "
                 "see the table under 'Deployable protocol' above")


def split_transfer(lines: list[str], size: int = 28) -> None:
    """Validation -> test transfer: accuracy, risk and class-prior drift."""
    section(lines, "Validation vs test splits (seed 0 bundles)")
    lines.append("| subset | n_val | n_test | val acc | test acc | gap | val risk | test risk | "
                 "test/val risk | class-prior L1 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    rows = []
    for path in sorted(glob.glob(str(ROOT / "outputs" / "logits"
                                          / f"*_resnet18_{size}_seed0_logits.npz"))):
        data = np.load(path, allow_pickle=False)
        name = os.path.basename(path).split(f"_resnet18_{size}")[0]
        meta = str(data["meta"]) if "meta" in data.files else ""
        multilabel = "multi-label" in meta
        val_labels = data["val_labels"].astype(int)
        test_labels = data["test_labels"].astype(int)
        if multilabel:
            def hamming(logits_labels):
                logits, labels = logits_labels
                probs = 1.0 / (1.0 + np.exp(-logits))
                return float(1.0 - ((probs >= 0.5).astype(int) == labels).mean())

            val_acc = 1.0 - hamming((data["val_logits"], val_labels))
            test_acc = 1.0 - hamming((data["test_logits"], test_labels))
            l1 = float("nan")
        else:
            from src import abstention as abst

            val_acc = float((abst.predict_proba(data["val_logits"]).argmax(axis=1)
                             == val_labels).mean())
            test_acc = float((abst.predict_proba(data["test_logits"]).argmax(axis=1)
                              == test_labels).mean())
            k = int(max(val_labels.max(), test_labels.max())) + 1
            p_val = np.bincount(val_labels, minlength=k) / val_labels.size
            p_test = np.bincount(test_labels, minlength=k) / test_labels.size
            l1 = float(np.abs(p_val - p_test).sum() / 2)
        rows.append((name, val_labels.shape[0] if not multilabel else val_labels.shape[0],
                     test_labels.shape[0], val_acc, test_acc, val_acc - test_acc,
                     1 - val_acc, 1 - test_acc, (1 - test_acc) / (1 - val_acc), l1))
    for name, n_val, n_test, val_acc, test_acc, gap, val_risk, test_risk, ratio, l1 in rows:
        lines.append(f"| {name} | {n_val} | {n_test} | {val_acc:.3f} | {test_acc:.3f} | "
                     f"{gap:+.3f} | {val_risk:.3f} | {test_risk:.3f} | {ratio:.1f} | {l1:.3f} |")
    worst = max(rows, key=lambda r: r[7] / r[6])
    lines.append(f"\nlargest risk inflation: {worst[0]} val risk {worst[6]:.3f} -> "
                 f"test risk {worst[7]:.3f} ({worst[8]:.1f}x)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    lines: list[str] = ["# Numbers quoted in the manuscript (generated)", "",
                        f"Source: outputs/results/*.csv and *.json under {RESULT_DIR}", ""]
    sanity_table(lines)
    coverage_table(lines, 28)
    split_transfer(lines)
    control_experiment(lines)
    augrc_agreement(lines, 28)
    augrc_agreement(lines, 128)
    deployable(lines)
    conformal_stats(lines)
    binary_degeneracy(lines)
    futility(lines)
    coverage_table(lines, 128)
    out = ROOT / "paper" / "NUMBERS.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
