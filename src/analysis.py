"""Shared analysis engine: logits/checkpoint -> abstention comparison tables.

Design goal: the GPU box only trains and dumps a `.npz` logit bundle; everything
in this module runs on CPU (reading a bundle does not need torch).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import abstention as abst
from .config import ABSTENTION_METHODS
from .metrics import (
    augrc,
    aurc,
    aurc_from_loss,
    class_conditional_coverage,
    clinical_translation,
    coverage_at_risk,
    coverage_at_risk_from_loss,
    e_aurc,
    ece,
    evaluate_fixed_threshold,
    macro_auc,
    paired_bootstrap_coverage_difference,
    risk_coverage_curve,
    risk_coverage_curve_from_loss,
    select_threshold_on_validation,
    select_threshold_on_validation_loss,
    subset_accuracy,
)

BUNDLE_VERSION = 1


@dataclass
class LogitBundle:
    """Everything needed to score abstention methods for one trained model."""

    val_logits: np.ndarray
    val_labels: np.ndarray
    test_logits: np.ndarray
    test_labels: np.ndarray
    mc_probs: np.ndarray | None = None  # mean softmax over stochastic passes
    mc_var: np.ndarray | None = None    # per-class variance across those passes
    meta: dict = field(default_factory=dict)
    source: str = ""

    def __post_init__(self):
        for name in ("val_logits", "test_logits"):
            if np.asarray(getattr(self, name)).ndim != 2:
                raise ValueError(f"{name} must be (n_samples, n_classes)")
        # Labels are (n,) for single-label tasks and (n, n_labels) for multi-label ones,
        # so compare the sample axis rather than the flat size.
        for logits_name, labels_name in (("val_logits", "val_labels"), ("test_logits", "test_labels")):
            n_samples = np.asarray(getattr(self, logits_name)).shape[0]
            labels = np.asarray(getattr(self, labels_name))
            if labels.shape[0] != n_samples:
                raise ValueError(f"{labels_name} length mismatch: {labels.shape[0]} != {n_samples}")
        if self.mc_probs is not None:
            if np.asarray(self.mc_probs).shape != np.asarray(self.test_logits).shape:
                raise ValueError("mc_probs shape must match test_logits")
            if self.mc_var is not None and np.asarray(self.mc_var).shape != np.asarray(self.test_logits).shape:
                raise ValueError("mc_var shape must match test_logits")

    @property
    def test_probs(self) -> np.ndarray:
        return abst.predict_proba(self.test_logits)


def save_bundle(path, bundle: LogitBundle) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    empty = np.zeros(0, dtype=np.float32)
    np.savez_compressed(
        path,
        bundle_version=np.array(BUNDLE_VERSION),
        val_logits=np.asarray(bundle.val_logits, dtype=np.float32),
        val_labels=np.asarray(bundle.val_labels, dtype=np.int64),
        test_logits=np.asarray(bundle.test_logits, dtype=np.float32),
        test_labels=np.asarray(bundle.test_labels, dtype=np.int64),
        mc_probs=(np.asarray(bundle.mc_probs, dtype=np.float32) if bundle.mc_probs is not None else empty),
        mc_var=(np.asarray(bundle.mc_var, dtype=np.float32) if bundle.mc_var is not None else empty),
        meta=np.array(json.dumps(bundle.meta, default=str)),
    )
    return path


def load_bundle(path) -> LogitBundle:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        version = int(data["bundle_version"]) if "bundle_version" in data else 0
        if version != BUNDLE_VERSION:
            raise ValueError(f"{path}: bundle version {version} != {BUNDLE_VERSION}")
        mc = data["mc_probs"]
        mc_var = data["mc_var"] if "mc_var" in data else np.zeros(0, dtype=np.float32)
        return LogitBundle(
            val_logits=data["val_logits"],
            val_labels=data["val_labels"],
            test_logits=data["test_logits"],
            test_labels=data["test_labels"],
            mc_probs=(mc if mc.size else None),
            mc_var=(mc_var if mc_var.size else None),
            meta=json.loads(str(data["meta"])),
            source=str(path),
        )


def operating_point(confidence, correct, max_risk: float) -> tuple[float, float]:
    """Coverage/risk at the largest coverage whose risk stays <= max_risk."""
    coverage, risk = risk_coverage_curve(confidence, correct)
    ok = np.nonzero(risk <= max_risk)[0]
    if not ok.size:
        return 0.0, float(risk[-1])
    idx = ok[-1]
    return float(coverage[idx]), float(risk[idx])


def operating_point_from_loss(confidence, loss, max_risk: float) -> tuple[float, float]:
    coverage, risk = risk_coverage_curve_from_loss(confidence, loss)
    ok = np.nonzero(risk <= max_risk)[0]
    if not ok.size:
        return 0.0, float(risk[-1])
    idx = ok[-1]
    return float(coverage[idx]), float(risk[idx])


def _method_output(name: str, bundles: list[LogitBundle], alpha: float,
                   split: str = "test") -> dict | None:
    """Method scores for one split ("test" for reporting, "val" for threshold selection).

    Returns None when the score cannot be computed for that split - most importantly,
    MC-dropout moments were exported for the test split only, so MC dropout cannot take
    part in the validation-selected protocol without re-exporting.
    """
    primary = bundles[0]
    logits = primary.val_logits if split == "val" else primary.test_logits
    labels = np.asarray(primary.val_labels if split == "val" else primary.test_labels).astype(int)

    if _is_multilabel(primary):
        # Multi-label: conformal/temperature scaling are not implemented, and a top-2
        # margin is undefined, so the remaining scores are label-free sample aggregates.
        if name in ("msp", "entropy"):
            mode = "certainty" if name == "msp" else "entropy"
            return abst.multilabel_confidence(logits, mode)
        if name == "mc_dropout":
            if split == "val" or primary.mc_probs is None:
                return None
            mean = np.asarray(primary.mc_probs, dtype=float)
            return {
                "pred": (mean >= 0.5).astype(int),
                "confidence": np.maximum(mean, 1.0 - mean).mean(axis=1),
                "probs": mean,
            }
        if name == "ensemble":
            if len(bundles) < 2:
                return None
            key = "val_logits" if split == "val" else "test_logits"
            return abst.multilabel_ensemble([abst.predict_proba_multilabel(getattr(b, key)) for b in bundles])
        return None
    if name in ("msp", "entropy"):
        return getattr(abst, name)(logits)
    if name == "margin":
        return abst.margin_confidence(logits)
    if name == "temp_scaling":
        # On the validation split the temperature is fitted and applied in-sample, which
        # is the usual "calibrate on validation" protocol.
        return abst.temp_scaling(primary.val_logits, primary.val_labels, logits)
    if name == "mc_dropout":
        if split == "val" or primary.mc_probs is None:
            return None
        mean = np.asarray(primary.mc_probs, dtype=float)
        # Variance needs the per-class second moment, which only the training side
        # has; if it was not exported we still score by mean MSP but flag uncertainty NaN.
        if primary.mc_var is not None:
            uncertainty = np.clip(np.asarray(primary.mc_var, dtype=float).sum(axis=1) * 0.5, 0.0, None)
        else:
            uncertainty = np.full(mean.shape[0], np.nan)
        return {
            "pred": mean.argmax(axis=1),
            "confidence": mean.max(axis=1),
            "uncertainty": uncertainty,
            "probs": mean,
        }
    if name == "ensemble":
        if len(bundles) < 2:
            return None
        probabilities = [abst.predict_proba(b.val_logits if split == "val" else b.test_logits)
                         for b in bundles]
        return abst.ensemble_confidence(probabilities)
    if name == "conformal_lac":
        return abst.conformal_lac(abst.predict_proba(primary.val_logits), primary.val_labels,
                                  abst.predict_proba(logits), alpha=alpha)
    if name == "conformal_aps":
        return abst.conformal_aps(abst.predict_proba(primary.val_logits), primary.val_labels,
                                  abst.predict_proba(logits), alpha=alpha)
    raise ValueError(f"unknown method: {name}")


def _is_multilabel(bundle: LogitBundle) -> bool:
    return str(bundle.meta.get("task", "")) == "multi-label"


def _skip_reason(name: str, bundle: LogitBundle) -> str:
    if _is_multilabel(bundle) and name in ("temp_scaling", "conformal_lac",
                                           "conformal_aps", "margin"):
        return "not defined for multi-label subsets"
    if name == "mc_dropout":
        return "needs mc_probs (retrain with --mc-samples > 0)"
    return "needs >= 2 model bundles (train extra seeds)"


def run_analysis(
    bundles: list[LogitBundle],
    methods: list[str] | None = None,
    alpha: float = 0.1,
    max_risk: float = 0.1,
) -> tuple[pd.DataFrame, dict, dict, list[dict]]:
    """Compare abstention methods across one or more trained models.

    Returns (table, curves, summary, skipped). `skipped` records methods that could
    not run and why - never silently dropped.
    """
    if not bundles:
        raise ValueError("no bundles given")
    methods = list(methods or ABSTENTION_METHODS)
    primary = bundles[0]
    y_test = np.asarray(primary.test_labels).astype(int)
    multilabel = _is_multilabel(primary)
    if multilabel:
        test_probs = abst.predict_proba_multilabel(primary.test_logits)
        correct = np.all((test_probs >= 0.5).astype(int) == y_test, axis=1)
        # Per-sample Hamming loss: the risk actually used for the curve.
        sample_loss = 1.0 - ((test_probs >= 0.5).astype(int) == y_test).mean(axis=1)
        # Oracle: rank samples by ascending loss (= descending -loss).
        oracle_aurc = aurc_from_loss(-sample_loss, sample_loss)
    else:
        correct = np.asarray(primary.test_logits).argmax(axis=1) == y_test
        sample_loss = None
        oracle_aurc = None

    rows, curves, skipped, confidences = [], {}, [], {}
    for name in methods:
        out = _method_output(name, bundles, alpha)
        if out is None:
            skipped.append({"method": name, "reason": _skip_reason(name, primary)})
            continue
        confidence = np.asarray(out["confidence"], dtype=float)
        n_samples = y_test.shape[0]
        if confidence.shape != (n_samples,):
            raise ValueError(f"{name}: confidence shape {confidence.shape} != {(n_samples,)}")
        row = {
            "method": name,
            "n_models": len(bundles) if name == "ensemble" else 1,
            "aurc": (aurc_from_loss(confidence, sample_loss) if multilabel
                     else aurc(confidence, correct)),
            "e_aurc": ((aurc_from_loss(confidence, sample_loss) - oracle_aurc) if multilabel
                       else e_aurc(confidence, correct)),
            "coverage_at_risk": (coverage_at_risk_from_loss(confidence, sample_loss, max_risk)
                                 if multilabel
                                 else coverage_at_risk(confidence, correct, max_risk)),
            # AUGRC (Traub et al., NeurIPS 2024) replaces/complements AURC, which was
            # shown to over-weight high-confidence failures and to break monotonicity.
            "augrc": (augrc(confidence, loss=sample_loss) if multilabel
                      else augrc(confidence, correct=correct)),
        }
        if "probs" in out and not multilabel:
            row["ece"] = ece(out["probs"], y_test)
        if "set_size" in out:
            row["mean_set_size"] = float(np.mean(out["set_size"]))
            row["empirical_coverage"] = abst.empirical_coverage(out["include"], y_test)
        # Deployable protocol: pick the threshold on the validation split, then apply it
        # unchanged to the test split. The `coverage_at_risk` above is a post-hoc,
        # oracle-selected operating point reported for comparability only.
        val_out = _method_output(name, bundles, alpha, split="val")
        if val_out is not None:
            conf_val = np.asarray(val_out["confidence"], dtype=float)
            if multilabel:
                val_probs = abst.predict_proba_multilabel(primary.val_logits)
                y_val = np.asarray(primary.val_labels).astype(int)
                val_loss = 1.0 - ((val_probs >= 0.5).astype(int) == y_val).mean(axis=1)
                threshold = select_threshold_on_validation_loss(conf_val, val_loss, max_risk)
                coverage_fixed, risk_fixed = evaluate_fixed_threshold(
                    confidence, correct, threshold, loss_test=sample_loss)
            else:
                y_val_int = np.asarray(primary.val_labels).astype(int)
                val_correct = np.asarray(primary.val_logits).argmax(axis=1) == y_val_int
                threshold = select_threshold_on_validation(conf_val, val_correct, max_risk)
                coverage_fixed, risk_fixed = evaluate_fixed_threshold(confidence, correct, threshold)
            row["threshold_val_selected"] = threshold
            row["coverage_val_selected"] = coverage_fixed
            row["risk_val_selected"] = risk_fixed
        rows.append(row)
        confidences[name] = confidence
        curves[name] = np.stack(risk_coverage_curve_from_loss(confidence, sample_loss)
                                if multilabel else risk_coverage_curve(confidence, correct))

    if not rows:
        raise RuntimeError(f"no method could run with these inputs; skipped={skipped}")

    table = pd.DataFrame(rows).sort_values("aurc").reset_index(drop=True)
    best_name = str(table.iloc[0]["method"])
    best_conf = np.asarray(_method_output(best_name, bundles, alpha)["confidence"], dtype=float)
    if multilabel:
        best_coverage, best_risk = operating_point_from_loss(best_conf, sample_loss, max_risk)
    else:
        best_coverage, best_risk = operating_point(best_conf, correct, max_risk)

    # Separate "strictly best" from "tied at the risk budget": several subsets reach
    # coverage 1.0 with every score, which is not evidence that the winner is better.
    tolerance = 1e-9
    best_coverage_value = float(table["coverage_at_risk"].max())
    winners = sorted(table.loc[table["coverage_at_risk"] >= best_coverage_value - tolerance, "method"])
    if winners == ["ensemble"]:
        verdict = "ensemble_strictly_best"
    elif "ensemble" in winners:
        verdict = "ensemble_tied_for_best"
    else:
        verdict = "another_method_better"

    # Paired bootstrap of the ensemble advantage over the best single-model score.
    singletons = table[table["method"].isin(
        ["msp", "entropy", "temp_scaling", "margin", "mc_dropout"])]
    ensemble_advantage = None
    if "ensemble" in confidences and len(singletons):
        runner_up = str(singletons.iloc[0]["method"])
        if not multilabel:
            ensemble_advantage = paired_bootstrap_coverage_difference(
                confidences["ensemble"], confidences[runner_up], correct, max_risk=max_risk)
            ensemble_advantage["comparison"] = f"ensemble_vs_{runner_up}"

    summary = {
        "sources": [b.source for b in bundles],
        "meta": primary.meta,
        "n_test": int(y_test.shape[0]),
        "test_accuracy": float(correct.mean()),
        "methods_run": [r["method"] for r in rows],
        "methods_skipped": skipped,
        "best_method": best_name,
        "best_coverage": best_coverage_value,
        "best_method_verdict": verdict,
        "tied_methods": winners,
        "ensemble_advantage": ensemble_advantage,
        "operating_point": {
            "max_risk": max_risk,
            "coverage": best_coverage,
            "risk": best_risk,
            "clinical_translation": clinical_translation(best_risk, best_coverage),
        },
        "class_conditional_example": class_conditional_coverage(
            best_conf, correct,
            np.asarray(primary.test_logits).argmax(axis=1) if multilabel else y_test,
            float(np.median(best_conf)),
        ),
    }
    if multilabel:
        probs = abst.predict_proba_multilabel(primary.test_logits)
        summary["metric_note"] = (
            "multi-label: risk = per-sample Hamming error (1 - label accuracy); "
            "exact-match statistics are reported separately"
        )
        summary["macro_auc"] = macro_auc(y_test, probs)
        summary["subset_accuracy"] = subset_accuracy(y_test, probs)
        summary["exact_match_correct_rate"] = float(correct.mean())
        summary["hamming_risk_at_full_coverage"] = float(sample_loss.mean())
    return table, curves, summary, skipped
