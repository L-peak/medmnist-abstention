"""Independently recompute the headline numbers of the manuscript and assert them.

    python scripts/verify_paper_claims.py

The deployable-protocol numbers, the conformal coverage numbers and the binary
degeneracy check are recomputed here from the raw logit bundles with a second,
self-contained implementation (no call into src.analysis), so a silent change in the
pipeline cannot pass unnoticed. The verdict/tie counts and the AUGRC agreement are
read from the released result files and cross-checked against the values quoted in
the manuscript. Exits non-zero on the first mismatch.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
LOGITS = ROOT / "outputs" / "logits"
RESULTS = ROOT / "outputs" / "results"
MAX_RISK = 0.1

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))
    print(f"[{'ok' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def multilabel_entropy_confidence(probs: np.ndarray) -> np.ndarray:
    """-mean binary entropy per label, matching src.abstention.multilabel_confidence."""
    clipped = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1 - 1e-12)
    entropy = -(clipped * np.log(clipped) + (1 - clipped) * np.log(1 - clipped)).mean(axis=1)
    return -entropy


def coverage_at_risk(confidence: np.ndarray, loss: np.ndarray, max_risk: float) -> float:
    """Largest coverage whose accepted-set risk stays within the budget (post-hoc)."""
    order = np.argsort(-confidence, kind="mergesort")
    ordered_loss = loss[order]
    running = np.cumsum(ordered_loss) / np.arange(1, ordered_loss.size + 1)
    ok = np.nonzero(running <= max_risk)[0]
    return float((ok[-1] + 1) / ordered_loss.size) if ok.size else 0.0


def threshold_for_risk(confidence: np.ndarray, loss: np.ndarray, max_risk: float) -> float:
    """Most permissive threshold whose accepted set stays within the risk budget."""
    order = np.argsort(-confidence, kind="mergesort")
    ordered_loss = loss[order]
    running = np.cumsum(ordered_loss) / np.arange(1, ordered_loss.size + 1)
    ok = np.nonzero(running <= max_risk)[0]
    if not ok.size:
        return float("inf")
    return float(confidence[order][ok[-1]])


def bundles(size: int = 28) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for path in sorted(glob.glob(str(LOGITS / f"*_resnet18_{size}_seed*_logits.npz"))):
        dataset = os.path.basename(path).split(f"_resnet18_{size}")[0]
        grouped.setdefault(dataset, []).append(Path(path))
    return grouped


def score_bundle(paths: list[Path], size: int = 28) -> dict:
    """All eight scores for one subset, computed from the raw bundles."""
    data = [np.load(p, allow_pickle=False) for p in paths]
    meta = str(data[0]["meta"]) if "meta" in data[0].files else ""
    multilabel = "multi-label" in meta
    val_labels = data[0]["val_labels"].astype(int)
    test_labels = data[0]["test_labels"].astype(int)
    probs_val = [softmax(d["val_logits"]) for d in data]
    probs_test = [softmax(d["test_logits"]) for d in data]
    mean_val = np.mean(probs_val, axis=0)
    mean_test = np.mean(probs_test, axis=0)

    if multilabel:
        sigmoid = lambda z: 1.0 / (1.0 + np.exp(-z))  # noqa: E731
        val_probs = sigmoid(data[0]["val_logits"])
        test_probs = sigmoid(data[0]["test_logits"])
        val_loss = 1.0 - ((val_probs >= 0.5).astype(int) == val_labels).mean(axis=1)
        test_loss = 1.0 - ((test_probs >= 0.5).astype(int) == test_labels).mean(axis=1)
        scores = {
            "msp": (np.maximum(val_probs, 1 - val_probs).mean(axis=1),
                    np.maximum(test_probs, 1 - test_probs).mean(axis=1)),
            # confidence = -mean binary entropy per label (higher is safer)
            "entropy": (multilabel_entropy_confidence(val_probs),
                        multilabel_entropy_confidence(test_probs)),
            "ensemble": (np.maximum(mean_val, 1 - mean_val).mean(axis=1),
                         np.maximum(mean_test, 1 - mean_test).mean(axis=1)),
        }
        return {"multilabel": True, "scores": scores,
                "val_loss": val_loss, "test_loss": test_loss}

    def margin(probs):
        order = np.argsort(-probs, axis=1)
        top = probs[np.arange(probs.shape[0]), order[:, 0]]
        runner = probs[np.arange(probs.shape[0]), order[:, 1]]
        return top - runner

    def fit_temperature(logits, labels):
        """Scalar temperature minimising validation NLL (coarse-to-fine grid)."""
        def nll(temperature):
            scaled = softmax(logits / temperature)
            picked = scaled[np.arange(labels.size), labels]
            return float(-np.log(np.clip(picked, 1e-12, None)).mean())

        grid = np.concatenate([np.linspace(0.5, 5.0, 46), np.linspace(5.1, 20.0, 150)])
        best = min(grid, key=nll)
        for _ in range(12):
            span = 0.2 * best if best > 1 else 0.05
            grid = np.linspace(max(0.05, best - span), best + span, 21)
            best = min(grid, key=nll)
        return float(best)

    def conformal_aps(probs_cal, labels_cal, probs_out, alpha=0.1):
        order_cal = np.argsort(-probs_cal, axis=1)
        sorted_cal = np.take_along_axis(probs_cal, order_cal, axis=1)
        ranks = np.argmax(order_cal == labels_cal[:, None], axis=1)
        scores = np.cumsum(sorted_cal, axis=1)[np.arange(labels_cal.size), ranks]
        size_cal = scores.size
        k = int(np.ceil((size_cal + 1) * (1.0 - alpha)))
        qhat = float(np.sort(scores)[k - 1])
        order_out = np.argsort(-probs_out, axis=1)
        sorted_out = np.take_along_axis(probs_out, order_out, axis=1)
        include_sorted = np.cumsum(sorted_out, axis=1) <= qhat
        include_sorted[:, 0] = True
        include = np.zeros_like(include_sorted, dtype=bool)
        np.put_along_axis(include, order_out, include_sorted, axis=1)
        return include.sum(axis=1)

    val_correct = probs_val[0].argmax(axis=1) == val_labels
    test_correct = probs_test[0].argmax(axis=1) == test_labels

    # conformal quantile from the validation split, sets scored by size
    true_val_prob = probs_val[0][np.arange(val_labels.size), val_labels]
    n = true_val_prob.size
    k = int(np.ceil((n + 1) * (1.0 - 0.1)))
    qhat = float(np.sort(1.0 - true_val_prob)[k - 1])
    lac_val = ((1.0 - probs_val[0]) <= qhat).sum(axis=1)
    lac_sets = (1.0 - probs_test[0]) <= qhat
    lac_size = lac_sets.sum(axis=1)
    lac_cover = float(lac_sets[np.arange(test_labels.size), test_labels].mean())

    temperature = fit_temperature(data[0]["val_logits"], val_labels)
    temp_probs_val = softmax(data[0]["val_logits"] / temperature)
    temp_probs_test = softmax(data[0]["test_logits"] / temperature)
    aps_val = conformal_aps(probs_val[0], val_labels, probs_val[0])
    aps_test = conformal_aps(probs_val[0], val_labels, probs_test[0])

    scores = {
        "msp": (probs_val[0].max(axis=1), probs_test[0].max(axis=1)),
        # confidence = -H, so higher is safer (matches src.abstention.entropy)
        "entropy": ((probs_val[0] * np.log(np.clip(probs_val[0], 1e-12, None))).sum(axis=1),
                    (probs_test[0] * np.log(np.clip(probs_test[0], 1e-12, None))).sum(axis=1)),
        "temp_scaling": (temp_probs_val.max(axis=1), temp_probs_test.max(axis=1)),
        "margin": (margin(probs_val[0]), margin(probs_test[0])),
        "ensemble": (mean_val.max(axis=1), mean_test.max(axis=1)),
        "conformal_lac": (-lac_val.astype(float), -lac_size.astype(float)),
        "conformal_aps": (-aps_val.astype(float), -aps_test.astype(float)),
    }
    return {"multilabel": False, "scores": scores, "val_correct": val_correct,
            "test_correct": test_correct, "lac_coverage": lac_cover,
            "lac_set_size": float(lac_size.mean())}


def main() -> int:
    grouped = bundles()
    over_budget = 0
    evaluated = 0
    all_over: list[str] = []
    none_over: list[str] = []
    lac_coverages: list[float] = []
    degenerate: dict[str, list[str]] = {}

    for dataset, paths in sorted(grouped.items()):
        scored = score_bundle(paths)
        over_here = 0
        for name, (conf_val, conf_test) in scored["scores"].items():
            if conf_test is None:
                continue
            val_loss = (scored["val_loss"] if scored["multilabel"]
                        else 1.0 - scored["val_correct"].astype(float))
            test_loss = (scored["test_loss"] if scored["multilabel"]
                         else 1.0 - scored["test_correct"].astype(float))
            threshold = threshold_for_risk(conf_val, val_loss, MAX_RISK)
            accept = conf_test >= threshold
            risk = float(test_loss[accept].mean()) if accept.any() else 0.0
            evaluated += 1
            if risk > MAX_RISK + 1e-9:
                over_budget += 1
                over_here += 1
        if scored["multilabel"]:
            # every score has a validation-time value here (MC dropout has none)
            if over_here == 0:
                none_over.append(dataset)
            if over_here == len(scored["scores"]):
                all_over.append(dataset)
        else:
            if over_here == len(scored["scores"]):  # every score, MC dropout excluded
                all_over.append(dataset)
            if over_here == 0:
                none_over.append(dataset)
            lac_coverages.append(scored["lac_coverage"])

        # binary degeneracy: do the four provably equivalent scores reach identical
        # post-hoc operating points, and do the conformal scores differ from them?
        if dataset in ("pneumoniamnist", "breastmnist"):
            losses = 1.0 - scored["test_correct"].astype(float)
            coverages = {name: coverage_at_risk(conf, losses, MAX_RISK)
                         for name, (_, conf) in scored["scores"].items()}
            reference = coverages["msp"]
            equivalent = sorted(name for name, value in coverages.items()
                                if name != "msp" and abs(value - reference) < 5e-4)
            conformal = {name: round(coverages[name], 3) for name in
                         ("conformal_lac", "conformal_aps")}
            degenerate[dataset] = (equivalent, round(reference, 4), conformal)

    check("deployable protocol: 80 (subset, score) operating points evaluated",
          evaluated == 80, f"evaluated={evaluated}")
    check("deployable protocol: 54 of 80 exceed the 10% test risk budget",
          over_budget == 54, f"over_budget={over_budget}")
    check("six subsets exceed the budget for every score",
          sorted(all_over) == ["dermamnist", "octmnist", "organsmnist", "pathmnist",
                               "pneumoniamnist", "retinamnist"], ",".join(sorted(all_over)))
    check("four subsets stay within the budget for every score",
          sorted(none_over) == ["bloodmnist", "chestmnist", "organamnist", "organcmnist"],
          ",".join(sorted(none_over)))
    mean_lac = float(np.mean(lac_coverages))
    below = sum(1 for c in lac_coverages if c < 0.9 - 1e-9)
    check("LAC mean test coverage 0.816 against a nominal 0.900",
          abs(mean_lac - 0.816) < 0.002, f"mean={mean_lac:.3f}")
    check("LAC below nominal on 10 of 11 single-label subsets",
          len(lac_coverages) == 11 and below == 10, f"below={below}/{len(lac_coverages)}")
    for dataset, (equivalent, reference, conformal) in degenerate.items():
        check(f"{dataset}: entropy / temperature / margin match MSP to 5e-4 coverage",
              equivalent == ["entropy", "margin", "temp_scaling"],
              f"equivalent={equivalent}, msp={reference}")
        check(f"{dataset}: both conformal scores separate from MSP",
              all(value < reference - 0.01 for value in conformal.values()),
              f"msp={reference}, conformal={conformal}")

    # artifact-based cross-checks
    verdicts = {"ensemble_strictly_best": 0, "ensemble_tied_for_best": 0,
                "another_method_better": 0}
    significant = []
    for path in sorted(glob.glob(str(RESULTS / "*_resnet18_28_summary.json"))):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        verdicts[payload["best_method_verdict"]] += 1
        advantage = payload.get("ensemble_advantage") or {}
        if advantage.get("ci_low", -1) > 0:
            significant.append(os.path.basename(path).split("_resnet18_28")[0])
    check("ensemble: 7 strict wins, 4 ties, 1 loss at 28x28",
          verdicts["ensemble_strictly_best"] == 7 and verdicts["ensemble_tied_for_best"] == 4
          and verdicts["another_method_better"] == 1, json.dumps(verdicts))
    check("ensemble advantage significant on exactly 3 subsets",
          sorted(significant) == ["dermamnist", "organsmnist", "tissuemnist"],
          ",".join(sorted(significant)))

    # Step A control: exchangeable calibration/evaluation halves of the test split
    control_path = RESULTS / "exchangeability_control.csv"
    if control_path.exists():
        control = pd.read_csv(control_path)
        deployable = control[~control["score"].str.startswith("conformal")]
        conformal_lac = control[control["score"] == "conformal_lac"]["coverage_out"].mean()
        check("control: mean evaluation-half risk 0.097 against the 0.10 budget",
              abs(deployable["risk_out"].mean() - 0.097) < 0.003,
              f"mean={deployable['risk_out'].mean():.3f}")
        check("control: mean risk gap (evaluation - calibration) below 0.02",
              abs(deployable["risk_out"].mean() - deployable["risk_cal"].mean()) < 0.02,
              f"gap={deployable['risk_out'].mean() - deployable['risk_cal'].mean():+.3f}")
        check("control: LAC coverage 0.901 against a nominal 0.900",
              abs(conformal_lac - 0.901) < 0.005, f"lac={conformal_lac:.3f}")
    else:
        check("control: exchangeability_control.csv present", False,
              "run scripts/exchangeability_control.py")

    failures = [name for name, ok, _ in CHECKS if not ok]
    print()
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED: {', '.join(failures)}")
        return 1
    print(f"ALL {len(CHECKS)} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
