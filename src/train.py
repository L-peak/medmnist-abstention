"""Training / evaluation helpers (AMP-safe on Turing GPUs such as T4)."""
from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def loss_for(meta) -> nn.Module:
    """Cross-entropy for single-label tasks; BCE for multi-label."""
    if meta.task == "multi-label":
        return nn.BCEWithLogitsLoss()
    return nn.CrossEntropyLoss()


def _loss_value(criterion, logits, targets, task: str):
    if task == "multi-label":
        return criterion(logits, targets.float())
    return criterion(logits, targets.long())


@torch.no_grad()
def evaluate(model: nn.Module, loader, meta, device: torch.device):
    """Return (logits, labels) as NumPy arrays."""
    model.eval()
    logits_all, labels_all = [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        out = model(images)
        logits_all.append(out.detach().float().cpu().numpy())
        if meta.task == "multi-label":
            labels_all.append(targets.numpy().astype(np.float32))
        else:
            labels_all.append(targets.numpy().reshape(-1).astype(np.int64))
    return np.concatenate(logits_all, axis=0), np.concatenate(labels_all, axis=0)


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    meta,
    device: torch.device,
    epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    verbose: bool = True,
):
    """Train with fp16 autocast when on CUDA.

    Turing (T4) has no bf16/TF32 support, so we deliberately use float16 + GradScaler.
    """
    criterion = loss_for(meta)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(1, epochs))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimiser.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(images)
                loss = _loss_value(criterion, logits, targets, meta.task)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            running += float(loss.detach()) * images.size(0)
            seen += images.size(0)
        scheduler.step()

        val_logits, val_labels = evaluate(model, val_loader, meta, device)
        if meta.task == "multi-label":
            from .metrics import macro_auc
            val_acc = macro_auc(val_labels, 1.0 / (1.0 + np.exp(-val_logits)))
        else:
            val_acc = float((val_logits.argmax(axis=1) == val_labels).mean())
        history.append({"epoch": epoch, "train_loss": running / max(1, seen), "val_acc": val_acc})
        if verbose:
            print(f"epoch {epoch:3d} | train_loss {running / max(1, seen):.4f} | val_acc {val_acc:.4f}")
    return history


def save_checkpoint(path, model: nn.Module, meta, extra: dict | None = None) -> None:
    payload = {"state_dict": model.state_dict(), "meta": vars(meta)}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path, model: nn.Module, device: torch.device) -> dict:
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    return payload
