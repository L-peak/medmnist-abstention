"""Kaggle kernel entry point (executed by "Save & Run All" / `kernels push`).

The project code arrives as a read-only Kaggle Dataset mounted under
/kaggle/input/. This script copies it into /kaggle/working (so outputs are kept
as kernel output), installs `medmnist`, then runs the whole training grid.

Kaggle environment (from /docs/notebooks): T4 x2, 4 CPU cores, 29 GB RAM,
20 GB auto-saved disk, 12 h max execution for GPU sessions.
"""
from __future__ import annotations

import os
import json
import shutil
import subprocess
import sys
from pathlib import Path

WORK = Path("/kaggle/working")
PROJECT = WORK / "project"

# Keep the multi-GB MedMNIST .npz files out of /kaggle/working: only that folder is
# saved as kernel output, and a 7 GB output makes downloading the results painful.
os.environ.setdefault("MEDMNIST_DATA_DIR", "/kaggle/temp/medmnist")

# Replaced by kaggle/launch.py at push time (`--run-args`).
# 11 h default: leaves an hour of margin inside the 12 h session cap.
RUN_ARGS_JSON = '["--max-minutes", "660", "--seeds", "0,1,2", "--zip"]'


def find_code_root() -> Path | None:
    """Locate the dataset copy of the project.

    The mount layout is not stable across Kaggle versions: we have seen both
    /kaggle/input/<slug>/src and /kaggle/input/datasets/<...>/src, so search
    recursively instead of assuming a depth.
    """
    for marker in Path("/kaggle/input").rglob("src/pipeline.py"):
        return marker.parents[1]
    return None


def describe_input(max_depth: int = 3) -> str:
    root = Path("/kaggle/input")
    lines = [f"contents of {root} (depth <= {max_depth}):"]
    for path in sorted(root.rglob("*")):
        depth = len(path.relative_to(root).parts)
        if depth > max_depth:
            continue
        lines.append("  " * depth + path.name + ("/" if path.is_dir() else ""))
    return "\n".join(lines)


def main() -> int:
    code_root = find_code_root()
    if code_root is None:
        print(describe_input())
        raise SystemExit("code dataset not found - attach it as a notebook input")
    print(f"code root: {code_root}")

    if PROJECT.exists():
        shutil.rmtree(PROJECT)
    shutil.copytree(code_root, PROJECT)
    os.chdir(PROJECT)
    print(f"working dir: {Path.cwd()}")

    # `medmnist` is not in Kaggle's image; this needs Internet enabled in the
    # notebook settings (which in turn requires phone verification).
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "medmnist"], check=False)

    import torch
    print(f"torch {torch.__version__} | cuda={torch.cuda.is_available()} "
          f"| gpus={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            print(f"  gpu{index}: {torch.cuda.get_device_name(index)}")

    args = os.environ.get("RUN_ARGS", "").split() or json.loads(RUN_ARGS_JSON)
    command = [sys.executable, "scripts/kaggle_run_all.py", *args]
    print("running:", " ".join(command))
    subprocess.run(command, check=True)

    bundles = sorted((PROJECT / "outputs" / "logits").glob("*.npz"))
    print(f"\nfinished with {len(bundles)} bundle(s) under {PROJECT / 'outputs'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
