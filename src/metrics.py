"""Risk-coverage metrics for selective prediction.

Pure NumPy on purpose: this module is unit-testable without torch and without
downloading any data.
"""
from __future__ import annotations

import numpy as np


def _as_1d(x) -> np.ndarray:
    return np.asarray(x).reshape(-1)


def risk_coverage_curve(confidence, correct):
    """Return (coverage, risk) when samples are accepted by descending confidence."""
    confidence = _as_1d(confidence).astype(float)
    correct = _as_1d(correct).astype(bool)
    if confidence.shape != correct.shape:
        raise ValueError("confidence and correct must have the same length")
    if confidence.size == 0:
        raise ValueError("empty input")
    order = np.argsort(-confidence, kind="mergesort")
    hits = correct[order]
    n = hits.size
    k = np.arange(1, n + 1)
    coverage = k / n
    risk = 1.0 - np.cumsum(hits) / k
    return coverage, risk


def risk_coverage_curve_from_loss(confidence, loss):
    """Risk-coverage curve for a continuous per-sample loss.

    Needed for multi-label tasks: exact-match correctness is dominated by the hardest
    labels, so a per-sample Hamming loss gives a far more informative curve.
    """
    confidence = _as_1d(confidence).astype(float)
    loss = _as_1d(loss).astype(float)
    if confidence.shape != loss.shape:
        raise ValueError("confidence and loss must have the same length")
    if confidence.size == 0:
        raise ValueError("empty input")
    order = np.argsort(-confidence, kind="mergesort")
    ordered = loss[order]
    n = ordered.size
    k = np.arange(1, n + 1)
    return k / n, np.cumsum(ordered) / k


def aurc_from_loss(confidence, loss) -> float:
    coverage, risk = risk_coverage_curve_from_loss(confidence, loss)
    return _trapezoid(risk, coverage)


def coverage_at_risk_from_loss(confidence, loss, max_risk: float = 0.1) -> float:
    coverage, risk = risk_coverage_curve_from_loss(confidence, loss)
    ok = np.nonzero(risk <= max_risk)[0]
    return float(coverage[ok[-1]]) if ok.size else 0.0


def _trapezoid(y, x) -> float:
    fn = getattr(np, "trapezoid", None) or np.trapz  # NumPy 2.x renamed trapz
    return float(fn(y, x))


def aurc(confidence, correct) -> float:
    """Area under the risk-coverage curve. Lower is better."""
    coverage, risk = risk_coverage_curve(confidence, correct)
    return _trapezoid(risk, coverage)


def oracle_aurc(correct) -> float:
    """AURC of a perfect ranker: the irreducible floor for this test set."""
    correct = _as_1d(correct).astype(float)
    n = correct.size
    k = np.arange(1, n + 1)
    risk = 1.0 - np.cumsum(np.sort(correct)[::-1]) / k
    return _trapezoid(risk, k / n)


def e_aurc(confidence, correct) -> float:
    """Excess AURC over the oracle ranking. Lower is better."""
    return aurc(confidence, correct) - oracle_aurc(correct)


def coverage_at_risk(confidence, correct, max_risk: float = 0.1) -> float:
    """Largest coverage whose risk stays at or below `max_risk` (0 if impossible)."""
    coverage, risk = risk_coverage_curve(confidence, correct)
    ok = np.nonzero(risk <= max_risk)[0]
    return float(coverage[ok[-1]]) if ok.size else 0.0


def select_threshold_on_validation(confidence_val, correct_val, max_risk: float = 0.1) -> float:
    """Pick an abstention threshold on a validation/calibration split.

    Returns the smallest confidence value (i.e. the most permissive threshold) whose
    risk on the validation split stays at or below `max_risk`. If even the single most
    confident validation sample exceeds the budget, returns +inf (accept nothing).
    """
    confidence_val = _as_1d(confidence_val).astype(float)
    correct_val = _as_1d(correct_val).astype(bool)
    coverage, risk = risk_coverage_curve(confidence_val, correct_val)
    ok = np.nonzero(risk <= max_risk)[0]
    if not ok.size:
        return float("inf")
    order = np.argsort(-confidence_val, kind="mergesort")
    return float(confidence_val[order][ok[-1]])


def select_threshold_on_validation_loss(confidence_val, loss_val, max_risk: float = 0.1) -> float:
    """Validation threshold selection for a continuous per-sample loss (multi-label).

    `select_threshold_on_validation` thresholds a 0/1 error indicator; for multi-label
    tasks the reported risk is the mean Hamming error of the accepted set, so the
    selection must optimise the same quantity or the two protocols disagree.
    """
    confidence_val = _as_1d(confidence_val).astype(float)
    loss_val = _as_1d(loss_val).astype(float)
    coverage, risk = risk_coverage_curve_from_loss(confidence_val, loss_val)
    ok = np.nonzero(risk <= max_risk)[0]
    if not ok.size:
        return float("inf")
    order = np.argsort(-confidence_val, kind="mergesort")
    return float(confidence_val[order][ok[-1]])


def evaluate_fixed_threshold(confidence_test, correct_test, threshold: float,
                             loss_test=None) -> tuple[float, float]:
    """Coverage and risk on the test split when the threshold was fixed elsewhere."""
    confidence_test = _as_1d(confidence_test).astype(float)
    if loss_test is None:
        loss_test = 1.0 - _as_1d(correct_test).astype(float)
    else:
        loss_test = _as_1d(loss_test).astype(float)
    accept = confidence_test >= threshold
    coverage = float(accept.mean())
    risk = float(loss_test[accept].mean()) if accept.any() else 0.0
    return coverage, risk


def paired_bootstrap_coverage_difference(confidence_a, confidence_b, correct,
                                         max_risk: float = 0.1, n_boot: int = 2000,
                                         seed: int = 0) -> dict:
    """Paired bootstrap CI for coverage(a) - coverage(b) on the same test samples.

    Both operating points are re-selected inside each bootstrap replicate, so the CI
    reflects resampling of the test set rather than a fixed threshold.
    """
    rng = np.random.default_rng(seed)
    confidence_a = _as_1d(confidence_a).astype(float)
    confidence_b = _as_1d(confidence_b).astype(float)
    correct = _as_1d(correct).astype(bool)
    n = correct.size
    deltas = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        sample = rng.integers(0, n, size=n)
        deltas[index] = (coverage_at_risk(confidence_a[sample], correct[sample], max_risk)
                         - coverage_at_risk(confidence_b[sample], correct[sample], max_risk))
    point = coverage_at_risk(confidence_a, correct, max_risk) - coverage_at_risk(confidence_b, correct, max_risk)
    return {
        "delta": float(point),
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
        "prob_positive": float((deltas > 0).mean()),
        "n_boot": int(n_boot),
    }


def ece(probs, labels, n_bins: int = 15) -> float:
    """Top-label expected calibration error (equal-width bins)."""
    probs = np.asarray(probs, dtype=float)
    labels = _as_1d(labels).astype(int)
    if probs.ndim != 2 or probs.shape[0] != labels.size:
        raise ValueError("probs must be (n_samples, n_classes)")
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if not mask.any():
            continue
        total += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(total)


def class_conditional_coverage(confidence, correct, labels, threshold: float) -> dict:
    """Abstention is 'accept if confidence >= threshold'; report per-class acceptance."""
    confidence = _as_1d(confidence).astype(float)
    correct = _as_1d(correct).astype(bool)
    labels = _as_1d(labels).astype(int)
    accept = confidence >= threshold
    out = {}
    for cls in np.unique(labels):
        mask = labels == cls
        kept = accept[mask]
        out[int(cls)] = {
            "n": int(mask.sum()),
            "coverage": float(kept.mean()) if mask.any() else float("nan"),
            "risk_on_accepted": float(1.0 - correct[mask][kept].mean()) if kept.any() else float("nan"),
        }
    return out


def clinical_translation(risk: float, coverage: float, n_patients: int = 100) -> dict:
    """Restate a risk/coverage pair in per-patient terms for the discussion section."""
    return {
        "coverage": float(coverage),
        "risk": float(risk),
        "miss_per_100": float(risk * n_patients),
        "deferred_per_100": float((1.0 - coverage) * n_patients),
    }


def macro_auc(labels, probs) -> float:
    """Macro ROC-AUC over labels, skipping labels with a single class present.

    This is the metric MedMNIST uses for multi-label subsets (e.g. chestmnist).
    """
    from sklearn.metrics import roc_auc_score

    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)
    scores = []
    for index in range(labels.shape[1]):
        column = labels[:, index]
        if column.min() == column.max():
            continue
        scores.append(roc_auc_score(column, probs[:, index]))
    return float(np.mean(scores)) if scores else float("nan")


def augrc_curve(confidence, loss):
    """Generalized risk-coverage curve of Traub et al. (NeurIPS 2024).

    x: coverage - fraction of accepted samples, ordered by descending confidence.
    y: cumulative loss rate among the accepted samples (the "silent failure" rate).
    The curve starts at (0, 0) and ends at (1, mean loss), so a perfect ranker
    keeps the area near 0 and a random ranker gives 0.5 * mean loss.
    AURC over-weights high-confidence failures and breaks monotonicity; AUGRC is the
    area under this curve and is bounded by [0, 1/2] for 0/1 errors.
    """
    confidence = _as_1d(confidence).astype(float)
    loss = _as_1d(loss).astype(float)
    if confidence.shape != loss.shape or confidence.size == 0:
        raise ValueError("confidence and loss must be non-empty and the same shape")
    # Descending confidence: the most confident samples are accepted first, and the
    # curve accumulates the failures hidden among them (equation 6 of the paper).
    order = np.argsort(-confidence, kind="mergesort")
    n = loss.size
    coverage = np.concatenate([[0.0], np.arange(1, n + 1) / n])
    cumulative = np.concatenate([[0.0], np.cumsum(loss[order]) / n])
    return coverage, cumulative


def augrc(confidence, correct=None, loss=None) -> float:
    """Area under the generalized risk-coverage curve. Lower is better."""
    if loss is None:
        if correct is None:
            raise ValueError("pass either correct or loss")
        loss = 1.0 - _as_1d(correct).astype(float)
    coverage, curve = augrc_curve(confidence, loss)
    return _trapezoid(curve, coverage)


def augrc_closed_form(confidence, correct) -> float:
    """Equation 7 of Traub et al. for 0/1 errors.

    AUGRC = (1 - AUROC_f) * acc * (1 - acc) + 0.5 * (1 - acc)^2
    AUROC_f is the AUROC of a failure score (higher = more failure-like), so we pass
    -confidence. Used to cross-check `augrc` in the smoke test.
    """
    from sklearn.metrics import roc_auc_score

    correct = _as_1d(correct).astype(int)
    confidence = _as_1d(confidence).astype(float)
    failure = 1 - correct
    if failure.min() == failure.max():
        return float("nan")
    auroc_f = roc_auc_score(failure, -confidence)
    acc = float(correct.mean())
    return float((1 - auroc_f) * acc * (1 - acc) + 0.5 * (1 - acc) ** 2)


def subset_accuracy(labels, probs, threshold: float = 0.5) -> float:
    """Exact-match ratio for multi-label predictions."""
    labels = np.asarray(labels, dtype=int)
    predicted = (np.asarray(probs, dtype=float) >= threshold).astype(int)
    return float(np.all(predicted == labels, axis=1).mean())
