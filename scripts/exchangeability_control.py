"""Step A control: does the protocol hold when calibration and evaluation *are* exchangeable?

    python scripts/exchangeability_control.py

The manuscript's deployable protocol fixes a threshold (or a conformal quantile) on the
official validation split and evaluates it on the test split, and it misses the 10% risk
budget in 54 of 80 cases. This control repeats the same protocol inside the test split:
the test samples are split in half at random, one half calibrates and the other is
evaluated (both directions, R splits). Those two halves are exchangeable by
construction, so the protocol should hit its target; if it does, the failure reported in
the manuscript is a property of the validation/test pair rather than of the threshold
rule.

Writes outputs/results/exchangeability_control.csv and .../EXCHANGEABILITY.md.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import abstention as abst  # noqa: E402
from src.config import RESULT_DIR  # noqa: E402
from src.metrics import (  # noqa: E402
    evaluate_fixed_threshold,
    select_threshold_on_validation,
    select_threshold_on_validation_loss,
)

ROOT = Path(__file__).resolve().parents[1]
LOGITS = ROOT / "outputs" / "logits"
MAX_RISK = 0.1
ALPHA = 0.1
CONFORMAL = ("conformal_lac", "conformal_aps")
# The manuscript's deployable table excludes MC dropout (no validation-time score);
# the exchangeable control can include it, because both halves come from the test split.
CONTROL_SCORES = ["msp", "entropy", "temp_scaling", "margin", "ensemble"]
MULTILABEL_SCORES = ["msp", "entropy", "ensemble"]


def load_bundles(size: int = 28) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for path in sorted(glob.glob(str(LOGITS / f"*_resnet18_{size}_seed*_logits.npz"))):
        dataset = os.path.basename(path).split(f"_resnet18_{size}")[0]
        with np.load(path, allow_pickle=False) as data:
            grouped.setdefault(dataset, []).append({key: data[key] for key in data.files})
    return grouped


def _nll(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    probs = abst.predict_proba(logits / temperature)
    return float(-np.log(np.clip(probs[np.arange(labels.size), labels], 1e-12, None)).mean())


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Scalar temperature minimising the calibration-half NLL (coarse-to-fine)."""
    grid = np.concatenate([np.linspace(0.5, 5.0, 46), np.linspace(5.1, 20.0, 150)])
    best = float(grid[int(np.argmin([_nll(logits, labels, t) for t in grid]))])
    for _ in range(12):
        span = 0.2 * best if best > 1 else 0.05
        grid = np.linspace(max(0.05, best - span), best + span, 21)
        best = float(grid[int(np.argmin([_nll(logits, labels, t) for t in grid]))])
    return best


def score_table(bundles: list[dict]) -> dict:
    """Every score computable from the test split alone (temperature is added per fold)."""
    meta = str(bundles[0]["meta"]) if "meta" in bundles[0] else ""
    multilabel = "multi-label" in meta
    labels = bundles[0]["test_labels"].astype(int)
    if multilabel:
        sigmoid = lambda z: 1.0 / (1.0 + np.exp(-z))  # noqa: E731
        probs = [np.asarray(sigmoid(b["test_logits"]), dtype=np.float64) for b in bundles]
        mean = np.mean(probs, axis=0)
        clipped = np.clip(probs[0], 1e-12, 1 - 1e-12)
        binary_entropy = -(clipped * np.log(clipped)
                           + (1 - clipped) * np.log(1 - clipped)).mean(axis=1)
        loss = 1.0 - ((mean >= 0.5).astype(int) == labels).mean(axis=1)
        mc = np.asarray(bundles[0]["mc_probs"], dtype=float)
        return {
            "multilabel": True, "labels": labels, "loss": loss, "logits": None,
            "scores": {
                "msp": np.maximum(probs[0], 1 - probs[0]).mean(axis=1),
                "entropy": -binary_entropy,
                "mc_dropout": np.maximum(mc, 1 - mc).mean(axis=1),
                "ensemble": np.maximum(mean, 1 - mean).mean(axis=1),
            },
        }

    probs = [abst.predict_proba(b["test_logits"]) for b in bundles]
    mean = np.mean(probs, axis=0)
    correct = probs[0].argmax(axis=1) == labels
    order = np.argsort(-probs[0], axis=1)
    top = probs[0][np.arange(labels.size), order[:, 0]]
    runner = probs[0][np.arange(labels.size), order[:, 1]]
    return {
        "multilabel": False, "labels": labels, "correct": correct,
        "loss": 1.0 - correct.astype(float),
        "logits": np.asarray(bundles[0]["test_logits"], dtype=float),
        "probs": probs[0],
        "scores": {
            "msp": top,
            "entropy": (probs[0] * np.log(np.clip(probs[0], 1e-12, None))).sum(axis=1),
            "margin": top - runner,
            "mc_dropout": np.asarray(bundles[0]["mc_probs"], dtype=float).max(axis=1),
            "ensemble": mean.max(axis=1),
        },
    }


def conformal_halves(probs_cal, labels_cal, probs_out, alpha=ALPHA) -> dict:
    """LAC and APS quantiles from the calibration half, sets scored by size."""
    lac_qhat = abst._conformal_quantile(
        1.0 - probs_cal[np.arange(labels_cal.size), labels_cal], alpha)

    order_cal = np.argsort(-probs_cal, axis=1)
    sorted_cal = np.take_along_axis(probs_cal, order_cal, axis=1)
    ranks = np.argmax(order_cal == labels_cal[:, None], axis=1)
    cumulative = np.cumsum(sorted_cal, axis=1)[np.arange(labels_cal.size), ranks]
    aps_qhat = abst._conformal_quantile(cumulative, alpha)

    def aps_include(probs):
        order = np.argsort(-probs, axis=1)
        sorted_probs = np.take_along_axis(probs, order, axis=1)
        include_sorted = np.cumsum(sorted_probs, axis=1) <= aps_qhat
        include_sorted[:, 0] = True
        include = np.zeros_like(include_sorted, dtype=bool)
        np.put_along_axis(include, order, include_sorted, axis=1)
        return include

    lac_cal = (1.0 - probs_cal) <= lac_qhat
    lac_out = (1.0 - probs_out) <= lac_qhat
    aps_cal, aps_out = aps_include(probs_cal), aps_include(probs_out)
    return {
        "conformal_lac": {
            "confidence_cal": -lac_cal.sum(axis=1).astype(float),
            "confidence_out": -lac_out.sum(axis=1).astype(float),
            "coverage_cal": abst.empirical_coverage(lac_cal, labels_cal),
            "coverage_out": None,  # filled by the caller with the evaluation labels
            "include_out": lac_out,
        },
        "conformal_aps": {
            "confidence_cal": -aps_cal.sum(axis=1).astype(float),
            "confidence_out": -aps_out.sum(axis=1).astype(float),
            "coverage_cal": abst.empirical_coverage(aps_cal, labels_cal),
            "coverage_out": None,
            "include_out": aps_out,
        },
    }


def fold_rows(scored: dict, index_cal: np.ndarray, index_out: np.ndarray) -> list[dict]:
    labels = scored["labels"]
    loss = scored["loss"]
    scores = dict(scored["scores"])
    rows: list[dict] = []

    if not scored["multilabel"]:
        temperature = fit_temperature(scored["logits"][index_cal], labels[index_cal])
        scores["temp_scaling"] = abst.predict_proba(
            scored["logits"] / temperature).max(axis=1)
        for name, payload in conformal_halves(scored["probs"][index_cal], labels[index_cal],
                                             scored["probs"][index_out]).items():
            coverage_out = abst.empirical_coverage(payload["include_out"], labels[index_out])
            rows.append({"score": name, "coverage_cal": payload["coverage_cal"],
                         "coverage_out": coverage_out, "risk_cal": float("nan"),
                         "risk_out": float("nan"),
                         "threshold": float("nan")})

    for name, confidence in scores.items():
        if scored["multilabel"]:
            threshold = select_threshold_on_validation_loss(
                confidence[index_cal], loss[index_cal], MAX_RISK)
        else:
            threshold = select_threshold_on_validation(
                confidence[index_cal], scored["correct"][index_cal], MAX_RISK)
        accept_cal = confidence[index_cal] >= threshold
        risk_cal = float(loss[index_cal][accept_cal].mean()) if accept_cal.any() else 0.0
        coverage_out, risk_out = evaluate_fixed_threshold(
            confidence[index_out], None, threshold, loss_test=loss[index_out])
        rows.append({"score": name, "coverage_cal": float(accept_cal.mean()),
                     "coverage_out": coverage_out, "risk_cal": risk_cal,
                     "risk_out": risk_out, "threshold": threshold})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=int, default=10,
                        help="random 50/50 splits of the test set (each used both ways)")
    parser.add_argument("--size", type=int, default=28)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []
    for dataset, bundles in sorted(load_bundles(args.size).items()):
        scored = score_table(bundles)
        n = scored["labels"].shape[0]
        for split in range(args.splits):
            permutation = rng.permutation(n)
            half = n // 2
            first, second = permutation[:half], permutation[half:2 * half]
            for index_cal, index_out in ((first, second), (second, first)):
                for row in fold_rows(scored, index_cal, index_out):
                    row.update({"dataset": dataset, "split": split})
                    rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(RESULT_DIR / "exchangeability_control.csv", index=False)
    deployable = table[~table["score"].isin(CONFORMAL)].copy()
    conformal = table[table["score"].isin(CONFORMAL)].copy()
    deployable["over_budget"] = deployable["risk_out"] > MAX_RISK + 1e-9
    deployable["risk_gap"] = deployable["risk_out"] - deployable["risk_cal"]

    lines = ["# Step A control: exchangeable calibration/evaluation halves (generated)", "",
             f"{args.splits} random 50/50 splits of the **test** set per subset, each used "
             "in both directions, so the calibration and evaluation halves are "
             "exchangeable by construction. The threshold (and the conformal quantile) is "
             "selected on the calibration half only, exactly as in the manuscript's "
             "deployable protocol.", ""]

    lines += ["## Deployable protocol inside the test split", ""]
    summary = (deployable.groupby("dataset")["risk_out"]
               .agg(mean_risk="mean", median_risk="median", max_risk="max").round(3))
    summary["mean_risk_gap"] = deployable.groupby("dataset")["risk_gap"].mean().round(3)
    summary["over_budget_rate"] = deployable.groupby("dataset")["over_budget"].mean().round(2)
    lines.append(summary.to_markdown())
    lines.append("")
    lines.append(f"- mean evaluation-half risk over all folds and scores: "
                 f"**{deployable['risk_out'].mean():.3f}** (budget {MAX_RISK})")
    lines.append(f"- mean calibration-half risk at the selected thresholds: "
                 f"{deployable['risk_cal'].mean():.3f}")
    lines.append(f"- mean risk gap (evaluation - calibration): "
                 f"**{deployable['risk_gap'].mean():+.3f}**, mean absolute gap "
                 f"{deployable['risk_gap'].abs().mean():.3f}, sd {deployable['risk_gap'].std():.3f}")
    lines.append(f"- folds above the budget: {deployable['over_budget'].mean():.2f} "
                 "(expected: the guarantee is marginal, so finite-half noise puts part "
                 "of the folds above the line even when the protocol is valid)")

    lines += ["", "## Split-conformal coverage inside the test split", ""]
    conf_summary = (conformal.groupby(["dataset", "score"])["coverage_out"].mean()
                    .unstack().round(3))
    lines.append(conf_summary.to_markdown())
    lines.append("")
    for name in CONFORMAL:
        subset = conformal[conformal["score"] == name]["coverage_out"]
        lines.append(f"- {name}: mean evaluation-half coverage **{subset.mean():.3f}** "
                     f"(nominal >= {1 - ALPHA:.2f}); below nominal in "
                     f"{(subset < 1 - ALPHA - 1e-9).mean():.2f} of folds")

    lines += ["", "## Validation -> test (manuscript Table 5) vs exchangeable halves", ""]
    comparisons = []
    for path in sorted(glob.glob(str(RESULT_DIR / "*_resnet18_28_selective.csv"))):
        dataset = os.path.basename(path).split("_resnet18_28")[0]
        frame = pd.read_csv(path)
        if "risk_val_selected" not in frame.columns:
            continue
        usable = frame.dropna(subset=["risk_val_selected"])
        if usable.empty:
            continue
        # the same score set as the exchangeable control, minus MC dropout
        usable = usable[usable["method"].isin(CONTROL_SCORES)]
        if usable.empty:
            continue
        exchangeable = deployable[(deployable["dataset"] == dataset)
                                  & (deployable["score"].isin(CONTROL_SCORES))]
        comparisons.append({
            "dataset": dataset,
            "validation_to_test_mean_risk": usable["risk_val_selected"].mean(),
            "exchangeable_halves_mean_risk": exchangeable["risk_out"].mean(),
        })
    comparison = pd.DataFrame(comparisons).round(3)
    comparison["ratio"] = (comparison["validation_to_test_mean_risk"]
                           / comparison["exchangeable_halves_mean_risk"]).round(2)
    lines.append(comparison.to_markdown(index=False))
    lines.append("")
    lines.append(f"- validation -> test: mean risk {comparison['validation_to_test_mean_risk'].mean():.3f}"
                 f" over the same score set")
    lines.append(f"- exchangeable halves: mean risk {comparison['exchangeable_halves_mean_risk'].mean():.3f}")

    out = RESULT_DIR / "EXCHANGEABILITY.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten: {out}\nwritten: {RESULT_DIR / 'exchangeability_control.csv'}")

    # LaTeX fragment for the manuscript (\input from paper/main.tex).
    latex_dir = ROOT / "paper" / "tables"
    latex_dir.mkdir(parents=True, exist_ok=True)
    latex = [
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{5pt}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Subset & Val$\\rightarrow$test & Exchangeable halves & Ratio \\\\",
        "\\midrule",
    ]
    for _, row in comparison.sort_values("validation_to_test_mean_risk",
                                         ascending=False).iterrows():
        latex.append(f"\\dataset{{{row['dataset']}}} & "
                     f"{row['validation_to_test_mean_risk']:.3f} & "
                     f"{row['exchangeable_halves_mean_risk']:.3f} & "
                     f"{row['ratio']:.2f} \\\\")
    latex += ["\\bottomrule", "\\end{tabular}", "\\normalsize", "",
              f"% mean risk at the 0.10 budget over the same five scores; "
              f"nominal-target row values are generated by scripts/exchangeability_control.py",
              ""]
    (latex_dir / "table_exchangeable.tex").write_text("\n".join(latex), encoding="utf-8")
    print(f"written: {latex_dir / 'table_exchangeable.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
