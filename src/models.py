"""Backbones: a MedMNIST-adapted ResNet-18 and a tiny CPU-friendly CNN."""
from __future__ import annotations

import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    """Fast sanity-check model. Has dropout so MC-dropout is meaningful."""

    def __init__(self, n_channels: int = 3, n_classes: int = 10, width: int = 32, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(n_channels, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1), nn.BatchNorm2d(width * 2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width * 2, width * 4, 3, padding=1), nn.BatchNorm2d(width * 4), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(width * 4, n_classes)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.head(self.dropout(x))


def resnet18_medmnist(n_channels: int = 3, n_classes: int = 10, image_size: int = 28) -> nn.Module:
    """ResNet-18 tuned for small inputs (MedMNIST's standard recipe).

    For <=64px inputs the 7x7/stride-2 stem and the max-pool are removed, otherwise
    the spatial grid collapses before the residual stages.
    """
    try:
        from torchvision.models import resnet18
    except ImportError as exc:  # torch-only installs can still run SmallCNN
        raise ImportError("resnet18 needs torchvision: pip install torchvision") from exc
    model = resnet18(weights=None, num_classes=n_classes)
    if image_size <= 64:
        model.conv1 = nn.Conv2d(n_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    else:
        model.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model


def build_model(name: str, n_channels: int, n_classes: int, image_size: int) -> nn.Module:
    if name == "resnet18":
        return resnet18_medmnist(n_channels, n_classes, image_size)
    if name == "smallcnn":
        return SmallCNN(n_channels, n_classes)
    raise ValueError(f"unknown model: {name}")


def enable_mc_dropout(model: nn.Module) -> None:
    """Turn dropout on while keeping BatchNorm in eval mode.

    Plain `model.train()` would also update BatchNorm running statistics on the
    test set, which silently corrupts MC-dropout estimates.
    """
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
        elif isinstance(module, nn.Dropout):
            module.train()


def disable_dropout(model: nn.Module) -> None:
    model.eval()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def resolve_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
