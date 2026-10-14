"""Abstention / uncertainty scoring methods.

Every method returns a dict with at least:
    pred        predicted class (n,)
    confidence  score where HIGHER = more trustworthy (used to rank for abstention)
For the conformal methods we also return `set_size` and `in_set` (empirical coverage).
"""
from __future__ import annotations

import numpy as np
import torch

from .models import enable_mc_dropout


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def predict_proba(logits: np.ndarray) -> np.ndarray:
    return _softmax(logits)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))


def predict_proba_multilabel(logits: np.ndarray) -> np.ndarray:
    """Per-label probabilities for multi-label heads."""
    return sigmoid(logits)


def multilabel_confidence(logits: np.ndarray, mode: str = "certainty") -> dict:
    """Sample-level confidence for a multi-label prediction.

    `certainty`: mean over labels of max(p, 1-p) - how decisive each label is.
    `entropy`:   negative mean binary entropy per label.
    Both are label-free, so they can be used as an abstention score at deployment.
    """
    probs = sigmoid(logits)
    if mode == "certainty":
        confidence = np.maximum(probs, 1.0 - probs).mean(axis=1)
    elif mode == "entropy":
        clipped = np.clip(probs, 1e-12, 1 - 1e-12)
        entropy = -(clipped * np.log(clipped) + (1 - clipped) * np.log(1 - clipped)).mean(axis=1)
        confidence = -entropy
    else:
        raise ValueError(f"unknown multilabel confidence mode: {mode}")
    return {"pred": (probs >= 0.5).astype(int), "confidence": confidence, "probs": probs}


def multilabel_ensemble(all_probs: list) -> dict:
    """Ensemble confidence for multi-label: mean per-label probability, disagreement = std."""
    stack = np.stack([np.asarray(p, dtype=np.float64) for p in all_probs], axis=0)
    mean = stack.mean(axis=0)
    return {
        "pred": (mean >= 0.5).astype(int),
        "confidence": np.maximum(mean, 1.0 - mean).mean(axis=1),
        "disagreement": stack.std(axis=0).mean(axis=1),
        "probs": mean,
        "n_models": len(all_probs),
    }


def msp(logits: np.ndarray) -> dict:
    probs = _softmax(logits)
    return {"pred": probs.argmax(axis=1), "confidence": probs.max(axis=1), "probs": probs}


def entropy(logits: np.ndarray) -> dict:
    probs = _softmax(logits)
    ent = -(probs * np.log(np.clip(probs, 1e-12, None))).sum(axis=1)
    return {"pred": probs.argmax(axis=1), "confidence": -ent, "probs": probs}


def fit_temperature(logits_val: np.ndarray, labels_val: np.ndarray, grid=None) -> float:
    """Pick the temperature minimising validation NLL (simple grid search)."""
    if grid is None:
        grid = np.concatenate([np.linspace(0.5, 5.0, 46), np.linspace(5.5, 20.0, 30)])
    labels_val = np.asarray(labels_val).astype(int)
    best_t, best_nll = 1.0, np.inf
    for temperature in grid:
        probs = _softmax(np.asarray(logits_val) / temperature)
        nll = -np.log(np.clip(probs[np.arange(labels_val.size), labels_val], 1e-12, None)).mean()
        if nll < best_nll:
            best_t, best_nll = float(temperature), float(nll)
    return best_t


def temp_scaling(logits_val: np.ndarray, labels_val: np.ndarray, logits_test: np.ndarray) -> dict:
    temperature = fit_temperature(logits_val, labels_val)
    probs = _softmax(np.asarray(logits_test) / temperature)
    return {"pred": probs.argmax(axis=1), "confidence": probs.max(axis=1), "probs": probs, "temperature": temperature}


@torch.no_grad()
def mc_dropout(model, loader, device: torch.device, n_samples: int = 20, seed: int = 0,
               task: str = "single") -> dict:
    """Monte-Carlo dropout. Returns mean probabilities and their variance."""
    torch.manual_seed(seed)
    model.eval()
    enable_mc_dropout(model)
    prob_fn = predict_proba_multilabel if task == "multi-label" else _softmax
    accum, accum_sq, n_batches = None, None, 0
    for _ in range(n_samples):
        batch_probs = []
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            batch_probs.append(prob_fn(model(images).detach().float().cpu().numpy()))
        probs = np.concatenate(batch_probs, axis=0)
        accum = probs if accum is None else accum + probs
        accum_sq = probs ** 2 if accum_sq is None else accum_sq + probs ** 2
        n_batches += 1
    mean = accum / n_batches
    var = np.clip(accum_sq / n_batches - mean ** 2, 0.0, None)
    model.eval()
    # Cheap, monotone uncertainty proxy: total predictive variance across classes.
    # Replace with a proper entropy decomposition if the paper reports mutual information.
    mutual_info = np.clip(var.sum(axis=1) * 0.5, 0.0, None)
    if task == "multi-label":
        confidence = np.maximum(mean, 1.0 - mean).mean(axis=1)
        prediction = (mean >= 0.5).astype(int)
    else:
        confidence = mean.max(axis=1)
        prediction = mean.argmax(axis=1)
    return {
        "pred": prediction,
        "confidence": confidence,
        "uncertainty": mutual_info,
        "probs": mean,
        "probs_var": var,
        "n_samples": n_samples,
    }


def ensemble_confidence(all_probs: list) -> dict:
    """Combine per-model probability matrices into an ensemble confidence."""
    stack = np.stack([np.asarray(p, dtype=np.float64) for p in all_probs], axis=0)
    mean = stack.mean(axis=0)
    disagreement = stack.std(axis=0).mean(axis=1)
    return {
        "pred": mean.argmax(axis=1),
        "confidence": mean.max(axis=1),
        "disagreement": disagreement,
        "probs": mean,
        "n_models": len(all_probs),
    }


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample conformal quantile: ceil((n+1)(1-alpha))/n empirical quantile."""
    scores = np.sort(np.asarray(scores, dtype=float).reshape(-1))
    n = scores.size
    if n == 0:
        raise ValueError("empty calibration set")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float("inf")
    return float(scores[k - 1])


def conformal_lac(probs_cal: np.ndarray, labels_cal: np.ndarray, probs_test: np.ndarray, alpha: float = 0.1) -> dict:
    """Least-ambiguous-set (a.k.a. '1 - p_true' score) split conformal."""
    probs_cal = np.asarray(probs_cal, dtype=float)
    labels_cal = np.asarray(labels_cal).astype(int)
    scores = 1.0 - probs_cal[np.arange(labels_cal.size), labels_cal]
    qhat = _conformal_quantile(scores, alpha)
    probs_test = np.asarray(probs_test, dtype=float)
    include = (1.0 - probs_test) <= qhat
    set_size = include.sum(axis=1)
    pred = probs_test.argmax(axis=1)
    # Confidence = margin of the top class over the best excluded class (higher = safer to accept)
    order = np.argsort(-probs_test, axis=1)
    top = probs_test[np.arange(probs_test.shape[0]), order[:, 0]]
    runner = probs_test[np.arange(probs_test.shape[0]), order[:, 1]] if probs_test.shape[1] > 1 else np.zeros_like(top)
    return {
        "pred": pred,
        # The conformal object for LAC is the prediction-set size: a smaller set means a
        # more decisive (safer to accept) prediction. `-set_size` therefore orders samples
        # exactly as the conformal construction does, and matches how APS is scored below.
        "confidence": -set_size.astype(float),
        # The top-1/top-2 margin is a *different* confidence score; it is exposed separately
        # (see `margin_confidence`) and reported as its own baseline rather than as "LAC".
        "margin": top - runner,
        "set_size": set_size,
        "include": include,
        "qhat": qhat,
        "alpha": alpha,
    }


def margin_confidence(logits: np.ndarray) -> dict:
    """Top-1 minus top-2 probability. Monotone in MSP for K=2, close to it for larger K."""
    probs = _softmax(logits)
    order = np.argsort(-probs, axis=1)
    top = probs[np.arange(probs.shape[0]), order[:, 0]]
    runner = (probs[np.arange(probs.shape[0]), order[:, 1]]
              if probs.shape[1] > 1 else np.zeros_like(top))
    return {"pred": probs.argmax(axis=1), "confidence": top - runner, "probs": probs}


def conformal_aps(probs_cal: np.ndarray, labels_cal: np.ndarray, probs_test: np.ndarray, alpha: float = 0.1) -> dict:
    """Adaptive prediction sets (Romano et al. 2020), deterministic variant.

    Nonconformity score = cumulative probability mass up to and including the true class
    when classes are sorted by descending probability.
    """
    probs_cal = np.asarray(probs_cal, dtype=float)
    labels_cal = np.asarray(labels_cal).astype(int)
    order_cal = np.argsort(-probs_cal, axis=1)
    sorted_cal = np.take_along_axis(probs_cal, order_cal, axis=1)
    ranks = np.argmax(order_cal == labels_cal[:, None], axis=1)
    scores = np.cumsum(sorted_cal, axis=1)[np.arange(labels_cal.size), ranks]
    qhat = _conformal_quantile(scores, alpha)

    probs_test = np.asarray(probs_test, dtype=float)
    order = np.argsort(-probs_test, axis=1)
    sorted_test = np.take_along_axis(probs_test, order, axis=1)
    cum = np.cumsum(sorted_test, axis=1)
    include_sorted = cum <= qhat
    include_sorted[:, 0] = True  # always keep the top class
    include = np.zeros_like(include_sorted, dtype=bool)
    np.put_along_axis(include, order, include_sorted, axis=1)
    return {
        "pred": probs_test.argmax(axis=1),
        "confidence": -include.sum(axis=1).astype(float),  # smaller sets are safer
        "set_size": include.sum(axis=1),
        "include": include,
        "qhat": qhat,
        "alpha": alpha,
    }


def empirical_coverage(include: np.ndarray, labels: np.ndarray) -> float:
    include = np.asarray(include)
    labels = np.asarray(labels).astype(int)
    return float(include[np.arange(labels.size), labels].mean())
