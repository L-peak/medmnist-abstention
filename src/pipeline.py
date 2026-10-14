"""One training run end to end: train -> evaluate -> export a logit bundle.

Shared by `scripts/run_baseline.py` (local, interactive) and
`scripts/kaggle_run_all.py` (batch, resumable) so the two can never drift.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from . import abstention as abst
from .analysis import LogitBundle, save_bundle
from .config import CKPT_DIR, LOGIT_DIR, RESULT_DIR, SEED
from .data import build_loaders
from .metrics import macro_auc, subset_accuracy
from .models import build_model, count_parameters, resolve_device
from .train import evaluate, save_checkpoint, seed_everything, train_model


def stem_for(dataset: str, model_name: str, size: int) -> str:
    return f"{dataset}_{model_name}_{size}"


def bundle_path_for(dataset: str, model_name: str, size: int, seed: int) -> Path:
    return LOGIT_DIR / f"{stem_for(dataset, model_name, size)}_seed{seed}_logits.npz"


def train_and_export(
    dataset: str,
    model_name: str = "resnet18",
    size: int = 28,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = SEED,
    mc_samples: int = 20,
    device=None,
    task: str | None = None,
    num_workers: int = 0,
    save_ckpt: bool = True,
    binary_collapse: bool = True,
    verbose: bool = True,
) -> dict:
    """Train one model and export the logit bundle used by the CPU-side analysis."""
    seed_everything(seed)
    dev = resolve_device(device)
    train_loader, val_loader, test_loader, meta = build_loaders(
        dataset, size=size, batch_size=batch_size, task=task, num_workers=num_workers,
        binary_collapse=binary_collapse,
    )
    model = build_model(model_name, meta.n_channels, meta.n_classes, size).to(dev)
    if verbose:
        print(f"[{dataset} seed={seed}] device={dev} task={meta.task} classes={meta.n_classes} "
              f"params={count_parameters(model):,}")

    started = time.time()
    history = train_model(model, train_loader, val_loader, meta, dev,
                          epochs=epochs, lr=lr, verbose=verbose)
    val_logits, val_labels = evaluate(model, val_loader, meta, dev)
    test_logits, test_labels = evaluate(model, test_loader, meta, dev)
    macro_auc_value = None
    if meta.task == "multi-label":
        probs = abst.predict_proba_multilabel(test_logits)
        macro_auc_value = macro_auc(test_labels, probs)
        test_acc = subset_accuracy(test_labels, probs)
    else:
        test_acc = float((test_logits.argmax(axis=1) == test_labels).mean())

    mc_probs = mc_var = None
    if mc_samples > 0:
        mc = abst.mc_dropout(model, test_loader, dev, n_samples=mc_samples, seed=seed, task=meta.task)
        mc_probs, mc_var = mc["probs"], mc["probs_var"]

    stem = stem_for(dataset, model_name, size)
    ckpt_path = None
    if save_ckpt:
        ckpt_path = CKPT_DIR / f"{stem}_seed{seed}.pt"
        save_checkpoint(ckpt_path, model, meta,
                        extra={"history": history, "test_acc": test_acc,
                               "seed": seed, "epochs": epochs})

    bundle = LogitBundle(
        val_logits=val_logits,
        val_labels=np.asarray(val_labels).astype(int),
        test_logits=test_logits,
        test_labels=np.asarray(test_labels).astype(int),
        mc_probs=mc_probs,
        mc_var=mc_var,
        meta={"dataset": meta.flag, "task": meta.task, "n_classes": meta.n_classes,
              "model": model_name, "size": size, "seed": seed, "epochs": epochs,
              "test_acc": test_acc},
        source=f"{stem}_seed{seed}",
    )
    bundle_path = save_bundle(bundle_path_for(dataset, model_name, size, seed), bundle)

    summary = {
        "dataset": dataset,
        "model": model_name,
        "size": size,
        "seed": seed,
        "epochs": epochs,
        "task": meta.task,
        "n_classes": meta.n_classes,
        "n_test": int(test_labels.size),
        "test_acc": test_acc,
        "macro_auc": macro_auc_value,
        "minutes": (time.time() - started) / 60.0,
        "bundle_path": str(bundle_path),
        "ckpt_path": str(ckpt_path) if ckpt_path else None,
    }
    (RESULT_DIR / f"{stem}_seed{seed}_baseline.json").write_text(
        json.dumps({**summary, "history": history}, indent=2), encoding="utf-8")
    if verbose:
        extra = f" macro_auc={macro_auc_value:.4f}" if macro_auc_value is not None else ""
        print(f"[{dataset} seed={seed}] test_acc={test_acc:.4f}{extra} in {summary['minutes']:.1f} min -> {bundle_path.name}")
    return summary
