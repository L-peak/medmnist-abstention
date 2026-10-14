"""Central configuration: dataset registry, method registry, paths."""
from __future__ import annotations

from pathlib import Path
import os

SEED = 0

ROOT = Path(__file__).resolve().parents[1]
# Overridable so Kaggle runs can keep the multi-GB .npz files outside
# /kaggle/working (only /kaggle/working is saved as kernel output).
DATA_DIR = Path(os.environ.get("MEDMNIST_DATA_DIR", ROOT / "data"))
OUT_DIR = ROOT / "outputs"
CKPT_DIR = OUT_DIR / "checkpoints"
RESULT_DIR = OUT_DIR / "results"
LOGIT_DIR = OUT_DIR / "logits"
FIG_DIR = OUT_DIR / "figures"

for _path in (DATA_DIR, CKPT_DIR, RESULT_DIR, LOGIT_DIR, FIG_DIR):
    _path.mkdir(parents=True, exist_ok=True)

# MedMNIST v2 ships 12 2D subsets (plus 6 3D ones, out of scope here).
ALL_2D_DATASETS = [
    "pathmnist",       # colon pathology, 9-class
    "chestmnist",      # chest X-ray, multi-label / binary
    "dermamnist",      # dermoscopy, 7-class
    "octmnist",        # retinal OCT, 4-class
    "pneumoniamnist",  # chest X-ray, binary
    "retinamnist",     # fundus, ordinal 5-class
    "breastmnist",     # breast ultrasound, binary
    "bloodmnist",      # blood smear, 8-class
    "tissuemnist",     # kidney cortex microscopy, 8-class
    "organamnist",     # abdominal CT, 11-class
    "organcmnist",     # abdominal CT, 11-class
    "organsmnist",     # abdominal CT, 11-class
]

# Pilot subset agreed in the plan: one pathology + one radiography dataset.
PILOT_DATASETS = ["pathmnist", "chestmnist"]

# Images come as RGB (MedMNIST expands grayscale when as_rgb=True).
N_CHANNELS = 3

ABSTENTION_METHODS = [
    "msp",             # max softmax probability (baseline)
    "entropy",         # predictive entropy
    "temp_scaling",    # temperature-scaled MSP (calibrated on val split)
    "margin",          # top-1 minus top-2 probability (non-conformal margin score)
    "mc_dropout",      # Monte-Carlo dropout
    "ensemble",        # deep ensemble
    "conformal_lac",   # split conformal LAC, scored by prediction-set size
    "conformal_aps",   # split conformal APS, scored by prediction-set size
]

# Multi-label subsets (MedMNIST v2). Collapsed to a binary "any finding" task by
# default; the collapse is applied in data.build_loaders.
MULTILABEL_FLAGS = {"chestmnist"}
