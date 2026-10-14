"""Drive Kaggle from this laptop: upload the code, push the run, fetch results.

Needs the Kaggle API token at %USERPROFILE%\\.kaggle\\kaggle.json
(Kaggle -> avatar -> Settings -> API -> Create New API Token) and:
    pip install kaggle

Usage:
    python kaggle/launch.py --push        # stage code -> dataset -> start the GPU run
    python kaggle/launch.py --status      # poll the run status
    python kaggle/launch.py --download    # pull kernel output into outputs/kaggle_download
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KAGGLE_DIR = PROJECT_ROOT / "kaggle"
STAGE = KAGGLE_DIR / "_stage"

DATASET_SLUG = "medmnist-abstention-code"
DEFAULT_KERNEL_SLUG = "medmnist-abstention-run"
CODE_ITEMS = ["src", "scripts", "README.md", "requirements.txt"]
SKIP_PARTS = {"__pycache__", "outputs", "data", "kaggle"}


def _api():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise SystemExit("missing dependency: pip install kaggle") from exc
    api = KaggleApi()
    api.authenticate()
    return api


def _retry(label: str, call, attempts: int = 4, delay: float = 6.0):
    """Retry an API call: api.kaggle.com drops connections intermittently."""
    last = None
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except Exception as exc:  # noqa: BLE001 - transient SSL/timeout errors
            last = exc
            print(f"[retry {attempt}/{attempts}] {label}: {type(exc).__name__}")
            time.sleep(delay)
    raise SystemExit(f"{label} failed after {attempts} attempts: {last}")


def _cli_status(kernel: str) -> str:
    """Fallback: the REST path through kagglesdk is flakier than the CLI on some networks."""
    result = subprocess.run(["kaggle", "kernels", "status", kernel],
                            capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip()[:300])
    return result.stdout.strip()


def _username(override: str | None = None) -> str:
    if override:
        return override
    token = Path(os.path.expanduser("~")) / ".kaggle" / "kaggle.json"
    if token.exists():
        return json.loads(token.read_text(encoding="utf-8"))["username"]
    # New-style KGAT_ access token: the username comes from the authenticated API.
    values = getattr(_api(), "config_values", {}) or {}
    username = values.get("username")
    if not username:
        raise SystemExit(
            "could not determine the Kaggle username - pass --username explicitly "
            "(token expected at ~/.kaggle/access_token)"
        )
    return username


def _ignore(_dir, names):
    return [n for n in names if n in SKIP_PARTS]


RUN_ARGS_MARKER = re.compile(r"^RUN_ARGS_JSON = .*$", re.MULTILINE)


def stage(username: str, run_args: str | None = None,
          kernel_slug: str = DEFAULT_KERNEL_SLUG) -> tuple[Path, Path]:
    """Build the two upload folders: the code dataset and the kernel."""
    if STAGE.exists():
        shutil.rmtree(STAGE)
    dataset_dir = STAGE / "dataset"
    kernel_dir = STAGE / "kernel"
    dataset_dir.mkdir(parents=True)
    kernel_dir.mkdir(parents=True)

    for item in CODE_ITEMS:
        source = PROJECT_ROOT / item
        if not source.exists():
            continue
        target = dataset_dir / item
        if source.is_dir():
            shutil.copytree(source, target, ignore=_ignore)
        else:
            shutil.copy2(source, target)
    (dataset_dir / "dataset-metadata.json").write_text(json.dumps({
        "title": "MedMNIST abstention code",
        "id": f"{username}/{DATASET_SLUG}",
        "licenses": [{"name": "other"}],
    }, indent=2), encoding="utf-8")

    run_source = (KAGGLE_DIR / "kernel" / "run.py").read_text(encoding="utf-8")
    if run_args:
        replacement = "RUN_ARGS_JSON = " + repr(json.dumps(run_args.split()))
        run_source, count = RUN_ARGS_MARKER.subn(replacement, run_source)
        if count != 1:
            raise SystemExit("marker RUN_ARGS_JSON not found in kaggle/kernel/run.py")
        print(f"kernel run args: {run_args}")
    (kernel_dir / "run.py").write_text(run_source, encoding="utf-8")
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{username}/{kernel_slug}",
        "title": kernel_slug.replace("-", " "),
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [f"{username}/{DATASET_SLUG}"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }, indent=2), encoding="utf-8")
    return dataset_dir, kernel_dir


def push(args) -> int:
    username = _username()
    api = _api()
    dataset_dir, kernel_dir = stage(username, args.run_args, args.kernel_slug)

    existing = False
    try:
        response = api.dataset_list_files(f"{username}/{DATASET_SLUG}")
        # An empty file list means the dataset does not exist yet.
        existing = bool(getattr(response, "files", None))
    except Exception:
        existing = False
    if existing and not args.new_dataset:
        print("dataset exists -> uploading a new version")
        _retry("dataset_create_version",
               lambda: api.dataset_create_version(str(dataset_dir), "update code", quiet=False,
                                                  convert_to_csv=False, dir_mode="tar"))
    else:
        print("creating dataset")
        # dir_mode="tar" is essential: the default ("skip") silently drops folders,
        # which would leave the kernel without src/ and scripts/.
        _retry("dataset_create_new",
               lambda: api.dataset_create_new(str(dataset_dir), public=False, quiet=False,
                                              convert_to_csv=False, dir_mode="tar"))

    print("pushing kernel (this starts the GPU run)")
    _retry("kernels_push", lambda: api.kernels_push(str(kernel_dir)))
    print(f"kernel: https://www.kaggle.com/code/{username}/{args.kernel_slug}")
    print("check progress with: python kaggle/launch.py --status")
    return 0


def status(args) -> int:
    username = _username()
    api = _api()
    kernel = f"{username}/{args.kernel_slug}"
    try:
        state = _retry("kernels_status", lambda: api.kernels_status(kernel), attempts=3, delay=5)
    except SystemExit:
        print("[warn] Python API unavailable, falling back to the kaggle CLI")
        print(_cli_status(kernel))
        return 0
    print(json.dumps(state, indent=2, default=str))
    return 0


def download(args) -> int:
    username = _username()
    api = _api()
    out = PROJECT_ROOT / "outputs" / "kaggle_download"
    if args.kernel_slug != DEFAULT_KERNEL_SLUG:
        out = PROJECT_ROOT / "outputs" / f"kaggle_download_{args.kernel_slug}"
    out.mkdir(parents=True, exist_ok=True)
    _retry("kernels_output",
           lambda: api.kernels_output(f"{username}/{args.kernel_slug}", path=str(out),
                                      file_pattern=getattr(args, "file_pattern", None)))
    print(f"downloaded kernel output to {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--push", action="store_true", help="upload code and start the run")
    group.add_argument("--status", action="store_true", help="poll the run status")
    group.add_argument("--download", action="store_true", help="fetch kernel output")
    group.add_argument("--stage-only", action="store_true",
                       help="just build kaggle/_stage and print the paths")
    parser.add_argument("--new-dataset", action="store_true",
                        help="force creating the dataset instead of a new version")
    parser.add_argument("--username", default=None,
                        help="override the detected Kaggle username")
    parser.add_argument("--run-args", default=None,
                        help="args forwarded to scripts/kaggle_run_all.py (with --push)")
    parser.add_argument("--kernel-slug", default=DEFAULT_KERNEL_SLUG,
                        help="use a different slug to run several batches in parallel "
                             "(pushing the same slug cancels the previous run)")
    parser.add_argument("--file-pattern", default=None,
                        help="with --download: regex; only matching output files are fetched. "
                             "Without it you also pull the downloaded MedMNIST data (several GB).")
    args = parser.parse_args()

    if args.stage_only:
        dataset_dir, kernel_dir = stage(_username(args.username), args.run_args, args.kernel_slug)
        print(f"dataset: {dataset_dir}\nkernel:  {kernel_dir}")
        return 0
    if args.push:
        return push(args)
    if args.status:
        return status(args)
    if args.download:
        return download(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
