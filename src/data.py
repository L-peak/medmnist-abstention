"""MedMNIST loading plus a dependency-free synthetic fallback.

Real data:  pip install medmnist   (downloads .npz from Zenodo into data/)
Synthetic:  used by scripts/smoke_test.py so the pipeline is verifiable
            without network access, without the medmnist package and without a GPU.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
import torch
from torch.utils.data import DataLoader, Dataset

from .config import DATA_DIR, N_CHANNELS, SEED


class ToTensorNormalize:
    """HWC uint8/RGB -> CHW float tensor in [-1, 1].

    Deliberately torchvision-free so the package imports on torch-only installs
    (the smoke test must run on a bare CPU box).
    """

    def __init__(self, mean: float = 0.5, std: float = 0.5):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = arr[:, :, None]
        arr = arr.astype(np.float32)
        if arr.max(initial=0.0) > 1.0:
            arr = arr / 255.0
        tensor = torch.from_numpy(np.array(arr.transpose(2, 0, 1), copy=True))
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(N_CHANNELS, 1, 1)
        return (tensor - self.mean) / self.std


@dataclass
class DatasetMeta:
    flag: str
    n_classes: int
    n_channels: int
    task: str  # "multi-class" | "binary" | "multi-label"
    size: int


def _import_medmnist():
    try:
        import medmnist
        from medmnist import INFO
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "`medmnist` is required for real data. Install it with: pip install medmnist"
        ) from exc
    return medmnist, INFO


def dataset_source(flag: str, size: int) -> tuple[str | None, str | None]:
    """(url, md5) for a specific MedMNIST / MedMNIST+ resolution."""
    _, INFO = _import_medmnist()
    suffix = "" if size == 28 else f"_{size}"
    info = INFO[flag]
    return info.get(f"url{suffix}"), info.get(f"MD5{suffix}")


def file_md5(path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_dataset_file(
    flag: str,
    size: int = 28,
    root=None,
    url: str | None = None,
    md5: str | None = None,
    retries: int = 6,
    verbose: bool = True,
):
    """Make sure `<flag>[_size].npz` exists and matches its MD5.

    MedMNIST's own downloader gives up on the first network hiccup, which is fatal
    for the multi-GB MedMNIST+ files (pathmnist_128.npz is 4.3 GB). This wrapper
    retries, resumes a partial file when the server supports ranges, and verifies
    the MD5 so a truncated file cannot silently poison a run.
    """
    if url is None or md5 is None:
        url, md5 = dataset_source(flag, size)
    if not url:
        return None
    suffix = "" if size == 28 else f"_{size}"
    target = Path(root or DATA_DIR) / f"{flag}{suffix}.npz"

    if target.exists() and (not md5 or file_md5(target) == md5):
        return target
    target.parent.mkdir(parents=True, exist_ok=True)

    last_error = None
    for attempt in range(1, retries + 1):
        start = target.stat().st_size if target.exists() else 0
        headers = {"Range": f"bytes={start}-"} if start else {}
        try:
            with requests.get(url, stream=True, timeout=120, headers=headers) as response:
                response.raise_for_status()
                resumed = response.status_code == 206 and start > 0
                mode = "ab" if resumed else "wb"
                if not resumed:
                    start = 0
                if verbose:
                    print(f"[download] {target.name} attempt {attempt}/{retries} "
                          f"({'resume' if resumed else 'restart'} at {start / 1e6:.1f} MB)")
                written = start
                with open(target, mode) as handle:
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                            written += len(chunk)
            if not md5 or file_md5(target) == md5:
                if verbose:
                    print(f"[download] {target.name} ok ({target.stat().st_size / 1e6:.1f} MB)")
                return target
            last_error = f"md5 mismatch after {written / 1e6:.1f} MB"
        except Exception as exc:  # noqa: BLE001 - network flakiness
            last_error = repr(exc)[:200]
        if verbose:
            print(f"[download] {target.name} attempt {attempt} failed: {last_error}")
        time.sleep(5)
    raise RuntimeError(f"could not download {target.name} from {url}: {last_error}")


def dataset_meta(flag: str, size: int = 28, task: str | None = None) -> DatasetMeta:
    """Resolve number of classes/task from the MedMNIST registry.

    `INFO[flag]["task"]` is the authoritative signal, e.g. "multi-label, binary-class"
    for chestmnist or "multi-class" for pathmnist. `label` is usually a flat
    {index: name} dict (14 entries for chestmnist), but older MedMNIST layouts nest
    it as {"multi-label": {...}, "binary-class": {...}}, so both are handled.
    """
    _, INFO = _import_medmnist()
    info = INFO[flag]
    labels = info["label"]
    declared = str(info.get("task", "")).lower()

    if isinstance(labels, dict) and labels and all(isinstance(v, dict) for v in labels.values()):
        key = ("binary-class" if task == "binary" and "binary-class" in labels
               else ("multi-label" if "multi-label" in labels else next(iter(labels))))
        n_classes = len(labels[key])
        default_task = ("multi-label" if key == "multi-label"
                        else ("binary" if n_classes == 2 else "multi-class"))
    else:
        n_classes = len(labels)
        if "multi-label" in declared:
            default_task = "multi-label"
        elif "binary" in declared and n_classes == 2:
            default_task = "binary"
        else:
            default_task = "multi-class"

    resolved = task or default_task
    if resolved == "binary" and n_classes != 2:
        raise ValueError(f"{flag}: requested binary task but {n_classes} labels found")

    # We always load with as_rgb=True, so every subset arrives as 3-channel.
    return DatasetMeta(flag=flag, n_classes=n_classes, n_channels=N_CHANNELS, task=resolved, size=size)


def collapsed_meta(meta: DatasetMeta) -> DatasetMeta:
    """Meta for a multi-label subset collapsed to a single 'any finding' binary task.

    Without this the model would be built with the multi-label class count (e.g. 14)
    while the targets are 0/1 - a shape mismatch that only shows up on chestmnist.
    """
    if meta.task != "multi-label":
        return meta
    return DatasetMeta(flag=meta.flag, n_classes=2, n_channels=meta.n_channels,
                       task="binary", size=meta.size)


def _transform() -> ToTensorNormalize:
    return ToTensorNormalize()


def _target_of(meta: DatasetMeta, raw) -> int:
    """MedMNIST returns np arrays; multi-label subsets need an argmax."""
    if meta.task == "multi-class":
        arr = np.asarray(raw).reshape(-1)
        return int(arr.argmax()) if arr.size > 1 else int(arr[0])
    return int(np.asarray(raw).reshape(-1)[0])


class AnyAbnormality(Dataset):
    """Collapse a multi-label subset (e.g. chestmnist) to a binary 'any finding' target."""

    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, target = self.base[index]
        label = int(np.asarray(target).reshape(-1).max() > 0)
        return image, label


class SingleLabelTarget(Dataset):
    """MedMNIST returns labels shaped (1,); flatten to a scalar.

    Without this, CrossEntropyLoss receives a (N, 1) target: it happens to compute
    the right number here, but the semantics are ambiguous and a later shape change
    would break silently.
    """

    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, target = self.base[index]
        return image, int(np.asarray(target).reshape(-1)[0])


class MultiLabelTarget(Dataset):
    """Keep the raw multi-hot vector as float32 (multi-label subsets such as chestmnist)."""

    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, target = self.base[index]
        vector = np.asarray(target, dtype=np.float32).reshape(-1)
        return image, torch.from_numpy(vector)


def build_loaders(
    flag: str,
    size: int = 28,
    batch_size: int = 128,
    num_workers: int = 0,
    task: str | None = None,
    binary_collapse: bool = True,
):
    """Return (train, val, test, meta) for a MedMNIST 2D subset."""
    medmnist, INFO = _import_medmnist()
    meta = dataset_meta(flag, size, task)
    is_multilabel = meta.task == "multi-label"
    if is_multilabel and binary_collapse:
        meta = collapsed_meta(meta)
    # MedMNIST+ (size != 28) files are multi-GB; fetch them ourselves with retries.
    ensure_dataset_file(flag, size)
    cls = getattr(medmnist, INFO[flag]["python_class"])

    def make(split: str) -> Dataset:
        kwargs = dict(split=split, download=True, transform=_transform(), root=str(DATA_DIR), as_rgb=True)
        try:
            dataset = cls(size=size, **kwargs)
        except TypeError:  # older medmnist releases: no `size` argument
            dataset = cls(**kwargs)
        if is_multilabel:
            dataset = AnyAbnormality(dataset) if binary_collapse else MultiLabelTarget(dataset)
        else:
            dataset = SingleLabelTarget(dataset)
        return dataset

    train, val, test = (make(s) for s in ("train", "val", "test"))
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    if num_workers > 0:
        # Without persistent_workers, every epoch re-spawns the worker processes.
        # On Windows (spawn start method) each spawn re-imports torch: measured 60x
        # slower end-to-end for a 2-epoch run on a 546-image subset.
        common.update(persistent_workers=True, prefetch_factor=2)
    return (
        DataLoader(train, shuffle=True, drop_last=False, **common),
        DataLoader(val, shuffle=False, **common),
        DataLoader(test, shuffle=False, **common),
        meta,
    )


# --------------------------------------------------------------------------------------
# Synthetic fallback
# --------------------------------------------------------------------------------------


class SyntheticMedMNIST(Dataset):
    """Separable blobs + label noise.

    Tuned so the task is clearly learnable (accuracy well above chance) but not
    saturated: a saturated task would make every abstention method look identical
    and the smoke test would stop being informative.

    The class prototypes (`centres`) define the task and MUST be shared across
    splits - generating independent prototypes per split makes train/val/test
    three different tasks and every metric meaningless.
    """

    def __init__(
        self,
        n: int = 512,
        size: int = 28,
        n_classes: int = 3,
        seed: int = SEED,
        noise: float = 0.05,
        jitter: float = 0.6,
        centres: torch.Tensor | None = None,
    ):
        rng = np.random.default_rng(seed)
        if centres is None:
            centres = synthetic_centres(n_classes, size, seed=seed)
        centres = np.asarray(centres, dtype=np.float32)
        labels = rng.integers(0, n_classes, size=n)
        images = centres[labels] + rng.normal(0.0, jitter, size=(n, N_CHANNELS, size, size))
        flip = rng.random(n) < noise
        labels = np.where(flip, (labels + 1) % n_classes, labels)
        self.images = torch.tensor(images, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return self.labels.numel()

    def __getitem__(self, index: int):
        return self.images[index], int(self.labels[index])


def synthetic_centres(n_classes: int = 3, size: int = 28, seed: int = SEED, separation: float = 3.0) -> np.ndarray:
    """Class prototypes shared by every split of the synthetic task."""
    rng = np.random.default_rng(seed)
    return (rng.normal(0.0, 1.0, size=(n_classes, N_CHANNELS, size, size)) * separation).astype(np.float32)


def synthetic_loaders(n_classes: int = 3, size: int = 28, batch_size: int = 64):
    meta = DatasetMeta(flag="synthetic", n_classes=n_classes, n_channels=N_CHANNELS, task="multi-class", size=size)
    centres = synthetic_centres(n_classes, size, seed=SEED)  # one task, three sample draws
    train = SyntheticMedMNIST(768, size, n_classes, seed=SEED, centres=centres)
    val = SyntheticMedMNIST(256, size, n_classes, seed=SEED + 1, centres=centres)
    test = SyntheticMedMNIST(256, size, n_classes, seed=SEED + 2, centres=centres)
    common = dict(batch_size=batch_size, num_workers=0)
    return (
        DataLoader(train, shuffle=True, drop_last=False, **common),
        DataLoader(val, shuffle=False, **common),
        DataLoader(test, shuffle=False, **common),
        meta,
    )
