"""Train one model on a MedMNIST 2D subset and export a logit bundle.

Thin CLI over `src.pipeline.train_and_export`. The bundle it writes is everything
the CPU-side analysis needs, so this is the only step that has to run on a GPU.
For a whole grid of runs on Kaggle use `scripts/kaggle_run_all.py` instead.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import SEED  # noqa: E402
from src.pipeline import train_and_export  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="pathmnist")
    parser.add_argument("--model", default="resnet18", choices=["resnet18", "smallcnn"])
    parser.add_argument("--size", type=int, default=28)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--task", default=None, help="force task for multi-task subsets (binary|multi-label)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=SEED,
                        help="vary this across runs to build a real deep ensemble")
    parser.add_argument("--mc-samples", type=int, default=20,
                        help="stochastic passes exported for MC-dropout (0 disables)")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-ckpt", action="store_true", help="skip writing the .pt checkpoint")
    parser.add_argument("--multi-label", action="store_true",
                        help="keep multi-label targets as-is (e.g. chestmnist's 14 labels) "
                             "instead of collapsing to 'any abnormality'")
    args = parser.parse_args()

    summary = train_and_export(
        dataset=args.dataset,
        model_name=args.model,
        size=args.size,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        mc_samples=args.mc_samples,
        device=args.device,
        task=args.task,
        num_workers=args.num_workers,
        save_ckpt=not args.no_ckpt,
        binary_collapse=not args.multi_label,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
