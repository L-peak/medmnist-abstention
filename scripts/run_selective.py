"""Compare abstention methods from logit bundles (CPU-only) or from a checkpoint.

Typical split:
    # on the GPU box (e.g. Kaggle T4 x2)
    python scripts/run_baseline.py --dataset pathmnist --seed 0
    python scripts/run_baseline.py --dataset pathmnist --seed 1
    # back on the laptop (no GPU needed)
    python scripts/run_selective.py --stem pathmnist_resnet18_28
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis import LogitBundle, load_bundle, run_analysis  # noqa: E402
from src.config import ABSTENTION_METHODS, CKPT_DIR, LOGIT_DIR, RESULT_DIR, SEED  # noqa: E402


def _bundles_from_disk(pattern: str) -> list:
    paths = sorted(Path(p) for p in glob.glob(pattern))
    if not paths:
        raise SystemExit(f"no logit bundles match: {pattern}")
    return [load_bundle(p) for p in paths]


def _stem_from_source(source: str) -> str:
    """`.../pathmnist_resnet18_28_seed1_logits.npz` -> `pathmnist_resnet18_28`."""
    name = Path(source).name
    for suffix in ("_logits.npz", ".npz"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return re.sub(r"_seed\d+$", "", name)


def _bundles_from_checkpoint(args) -> list:
    """Fallback when no bundle was exported: rerun the checkpoint to get logits."""
    from src import abstention as abst
    from src.data import build_loaders
    from src.models import build_model, resolve_device
    from src.train import evaluate, load_checkpoint, seed_everything

    seed_everything(SEED)
    device = resolve_device(args.device)
    _, val_loader, test_loader, meta = build_loaders(
        args.dataset, size=args.size, batch_size=args.batch_size, task=args.task
    )
    stem = f"{args.dataset}_{args.model}_{args.size}"
    ckpt = CKPT_DIR / f"{stem}.pt"
    model = build_model(args.model, meta.n_channels, meta.n_classes, args.size).to(device)
    if ckpt.exists():
        load_checkpoint(ckpt, model, device)
        print(f"loaded {ckpt}")
    else:
        print(f"[warn] {ckpt} not found - using an untrained model")
    val_logits, val_labels = evaluate(model, val_loader, meta, device)
    test_logits, test_labels = evaluate(model, test_loader, meta, device)
    mc_probs = None
    mc_var = None
    if args.mc_samples > 0:
        mc = abst.mc_dropout(model, test_loader, device, n_samples=args.mc_samples)
        mc_probs, mc_var = mc["probs"], mc["probs_var"]
    return [LogitBundle(val_logits, val_labels, test_logits, test_labels,
                        mc_probs=mc_probs, mc_var=mc_var, meta=vars(meta), source=stem)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stem", default=None,
                        help="bundle stem, e.g. pathmnist_resnet18_28 (loads every exported seed)")
    parser.add_argument("--logits", default=None, help="explicit glob of .npz bundles")
    parser.add_argument("--dataset", default="pathmnist")
    parser.add_argument("--model", default="resnet18", choices=["resnet18", "smallcnn"])
    parser.add_argument("--size", type=int, default=28)
    parser.add_argument("--task", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=0.1, help="miscoverage level for conformal methods")
    parser.add_argument("--max-risk", type=float, default=0.1, help="risk budget for coverage@risk")
    parser.add_argument("--mc-samples", type=int, default=20, help="checkpoint mode only")
    parser.add_argument("--methods", default=None, help="comma-separated subset of methods")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.logits or args.stem:
        pattern = args.logits or str(LOGIT_DIR / f"{args.stem}_seed*_logits.npz")
        bundles = _bundles_from_disk(pattern)
        save_stem = args.stem or _stem_from_source(bundles[0].source)
    else:
        bundles = _bundles_from_checkpoint(args)
        save_stem = f"{args.dataset}_{args.model}_{args.size}"
    print(f"loaded {len(bundles)} bundle(s): {[Path(b.source).name for b in bundles]}")

    methods = args.methods.split(",") if args.methods else ABSTENTION_METHODS
    table, curves, summary, skipped = run_analysis(
        bundles, methods=methods, alpha=args.alpha, max_risk=args.max_risk
    )

    print()
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    for item in skipped:
        print(f"[skipped] {item['method']}: {item['reason']}")

    csv_path = RESULT_DIR / f"{save_stem}_selective.csv"
    npz_path = RESULT_DIR / f"{save_stem}_risk_coverage.npz"
    json_path = RESULT_DIR / f"{save_stem}_summary.json"
    table.to_csv(csv_path, index=False)
    np.savez_compressed(npz_path, **curves)
    json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    point = summary["operating_point"]
    translation = point["clinical_translation"]
    print(f"\nbest={summary['best_method']} | risk<={point['max_risk']}: coverage={point['coverage']:.3f} "
          f"risk={point['risk']:.3f} | miss/100={translation['miss_per_100']:.1f} "
          f"deferred/100={translation['deferred_per_100']:.1f}")
    print(f"saved: {csv_path}\nsaved: {npz_path}\nsaved: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
