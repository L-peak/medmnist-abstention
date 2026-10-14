"""One-shot Kaggle batch runner: train a grid of (dataset, seed) runs.

Built around Kaggle's free GPU quota (30 h/week, up to 9 h per session):

  * resumable     - a run whose bundle already exists is skipped, so you can
                    restart the session and continue where it stopped
  * time-budgeted - stops cleanly before the session limit instead of being killed
  * self-contained- zips every bundle into one file for a single download, then
                    you analyse everything locally on CPU

Example (Kaggle notebook cell):
    !python scripts/kaggle_run_all.py --max-minutes 480 --zip
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import LOGIT_DIR, OUT_DIR, RESULT_DIR  # noqa: E402
from src.pipeline import bundle_path_for, train_and_export  # noqa: E402

# Modality-diverse default grid with test sets big enough for stable curves.
DEFAULT_DATASETS = "pathmnist,bloodmnist,dermamnist,octmnist,organamnist,retinamnist,pneumoniamnist,breastmnist"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=DEFAULT_DATASETS,
                        help="comma-separated MedMNIST flags")
    parser.add_argument("--seeds", default="0,1,2", help="comma-separated seeds (>=2 enables the ensemble)")
    parser.add_argument("--model", default="resnet18", choices=["resnet18", "smallcnn"])
    parser.add_argument("--size", type=int, default=28)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--mc-samples", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="0 is fastest here: MedMNIST lives in RAM and worker "
                             "respawn costs more than it saves (esp. on Windows)")
    parser.add_argument("--max-minutes", type=float, default=480.0,
                        help="stop starting new runs after this many minutes (Kaggle session limit is 9 h)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--keep-checkpoints", action="store_true",
                        help="also write .pt files (costs ~45 MB per run)")
    parser.add_argument("--multi-label", action="store_true",
                        help="train multi-label subsets with their raw targets (chestmnist: 14 labels)")
    parser.add_argument("--no-skip-existing", action="store_true",
                        help="retrain runs that already have a bundle")
    parser.add_argument("--zip", action="store_true", help="zip all bundles at the end")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    plan = [(d, s) for d in datasets for s in seeds]

    try:
        import torch
        available = torch.cuda.is_available()
        print(f"torch {torch.__version__} | cuda_available={available} "
              f"| device_count={torch.cuda.device_count()}")
        if available:
            print(f"gpu0: {torch.cuda.get_device_name(0)}")
    except Exception as exc:  # torch is required for training anyway
        print(f"[warn] torch probe failed: {exc}")

    print(f"plan: {len(plan)} runs ({len(datasets)} datasets x {len(seeds)} seeds), "
          f"epochs={args.epochs}, model={args.model}, time budget={args.max_minutes:.0f} min")
    if args.dry_run:
        for dataset, seed in plan:
            state = "have" if bundle_path_for(dataset, args.model, args.size, seed).exists() else "todo"
            print(f"  [{state}] {dataset} seed={seed}")
        return 0

    started = time.time()
    manifest: list[dict] = []
    completed_minutes: list[float] = []
    stopped_early = False

    for index, (dataset, seed) in enumerate(plan, start=1):
        target = bundle_path_for(dataset, args.model, args.size, seed)
        if target.exists() and not args.no_skip_existing:
            print(f"[{index}/{len(plan)}] skip existing {target.name}")
            manifest.append({"dataset": dataset, "seed": seed, "status": "skipped",
                             "bundle": str(target)})
            continue

        elapsed = (time.time() - started) / 60.0
        # Reserve time for the run in progress: use the average of finished runs.
        reserve = sum(completed_minutes) / len(completed_minutes) if completed_minutes else 0.0
        if elapsed + reserve > args.max_minutes:
            print(f"[{index}/{len(plan)}] time budget reached ({elapsed:.0f} min elapsed) - stopping cleanly")
            stopped_early = True
            break

        print(f"[{index}/{len(plan)}] training {dataset} seed={seed} (elapsed {elapsed:.0f} min)")
        try:
            summary = train_and_export(
                dataset=dataset,
                model_name=args.model,
                size=args.size,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                seed=seed,
                mc_samples=args.mc_samples,
                device=args.device,
                num_workers=args.num_workers,
                save_ckpt=args.keep_checkpoints,
                binary_collapse=not args.multi_label,
                verbose=True,
            )
            completed_minutes.append(summary["minutes"])
            manifest.append({"dataset": dataset, "seed": seed, "status": "ok", **summary})
        except Exception as exc:  # keep going: one bad subset must not kill the session
            print(f"[error] {dataset} seed={seed}: {exc!r}")
            manifest.append({"dataset": dataset, "seed": seed, "status": "error",
                             "error": repr(exc)[:400]})

    remaining = [(d, s) for d, s in plan
                 if not any(m["dataset"] == d and m["seed"] == s for m in manifest)]
    for dataset, seed in remaining:
        manifest.append({"dataset": dataset, "seed": seed, "status": "not_started"})

    total_minutes = (time.time() - started) / 60.0
    ok = sum(1 for m in manifest if m["status"] == "ok")
    skipped = sum(1 for m in manifest if m["status"] == "skipped")
    errors = sum(1 for m in manifest if m["status"] == "error")
    print(f"\ndone: ok={ok} skipped={skipped} errors={errors} "
          f"not_started={len(remaining)} in {total_minutes:.1f} min")

    manifest_path = OUT_DIR / "kaggle_manifest.json"
    manifest_path.write_text(json.dumps(
        {"model": args.model, "size": args.size, "epochs": args.epochs,
         "max_minutes": args.max_minutes, "total_minutes": total_minutes,
         "stopped_early": stopped_early, "runs": manifest}, indent=2), encoding="utf-8")
    print(f"manifest: {manifest_path}")

    if args.zip:
        bundles = sorted(LOGIT_DIR.glob("*.npz"))
        if bundles:
            archive = RESULT_DIR / "logit_bundles"
            archive_path = shutil.make_archive(str(archive), "zip", root_dir=str(LOGIT_DIR))
            size_mb = Path(archive_path).stat().st_size / 1e6
            print(f"zip: {archive_path} ({len(bundles)} bundles, {size_mb:.1f} MB) <- download this")
        else:
            print("[warn] no bundles to zip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
