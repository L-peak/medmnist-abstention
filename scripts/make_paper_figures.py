"""Build the multi-panel risk-coverage figure for the paper (vector PDF + PNG).

Runs standalone (numpy + matplotlib only) so it never shares a process with torch -
that combination aborts on Windows/conda with a duplicate libiomp5md.dll error.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PANELS = [
    ("octmnist_resnet18_28", "(a) OCT, 28$\\times$28: best score wins, APS loses"),
    ("octmnist_resnet18_128", "(b) OCT, 128$\\times$128: same ranking, better operating point"),
    ("retinamnist_resnet18_28", "(c) Retina, 28$\\times$28: futility regime"),
    ("pneumoniamnist_resnet18_28", "(d) Pneumonia, 28$\\times$28: binary degeneracy"),
]

METHOD_ORDER = ["msp", "entropy", "temp_scaling", "mc_dropout",
                "ensemble", "conformal_lac", "conformal_aps"]


def _results_table(results: Path, size: int = 28):
    """Per-dataset x per-method table, assembled from the run_selective CSVs."""
    import pandas as pd

    frames = []
    for path in sorted(results.glob(f"*_resnet18_{size}_selective.csv")):
        dataset = path.name.split(f"_resnet18_{size}")[0]
        frame = pd.read_csv(path)
        frame["dataset"] = dataset
        frames.append(frame)
    if not frames:
        raise SystemExit(f"no *_resnet18_{size}_selective.csv under {results}")
    return pd.concat(frames, ignore_index=True)


def figure_heatmap(results: Path, out_dir: Path, size: int = 28,
                   max_risk: float = 0.1) -> None:
    """Coverage at the risk budget for every dataset x score - the whole benchmark at a glance."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = _results_table(results, size)
    pivot = table.pivot_table(index="dataset", columns="method",
                              values="coverage_at_risk", aggfunc="mean")
    pivot = pivot.reindex(columns=[m for m in METHOD_ORDER if m in pivot.columns])
    pivot = pivot.sort_values("ensemble", ascending=False)

    fig, ax = plt.subplots(figsize=(6.2, 4.6), dpi=300)
    data = pivot.to_numpy(dtype=float)
    image = ax.imshow(data, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(pivot.shape[1]), pivot.columns, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(pivot.shape[0]), pivot.index, fontsize=8)
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            value = data[row, col]
            label = "n/a" if np.isnan(value) else f"{value:.2f}"
            ax.text(col, row, label, ha="center", va="center", fontsize=6.5,
                    color="black" if not np.isnan(value) else "gray")
    ax.set_title(f"Coverage at risk $\\leq$ {max_risk:.1f} ({size}$\\times${size})", fontsize=9)
    ax.set_xlabel("abstention score", fontsize=8)
    fig.colorbar(image, ax=ax, shrink=0.8, label="coverage (higher is better)")
    fig.tight_layout()
    fig.savefig(out_dir / "Figure_3_heatmap.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "Figure_3_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_dir / 'Figure_3_heatmap.pdf'}")


def figure_futility(results: Path, out_dir: Path, size: int = 28,
                    max_risk: float = 0.1) -> None:
    """Backbone accuracy vs the coverage the best score can offer - quantifies the futility regime."""
    import json

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = _results_table(results, size)
    best = (table.sort_values("aurc").groupby("dataset", as_index=False).first()
            [["dataset", "method", "coverage_at_risk"]])

    accuracy = {}
    for path in results.glob(f"*_resnet18_{size}_seed*_baseline.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        accuracy.setdefault(payload["dataset"], []).append(payload["test_acc"])

    points = []
    for _, row in best.iterrows():
        accs = accuracy.get(row["dataset"])
        if not accs:
            continue
        points.append((float(np.mean(accs)), float(row["coverage_at_risk"]), row["dataset"]))
    points.sort()

    fig, ax = plt.subplots(figsize=(5.6, 4.2), dpi=300)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.scatter(xs, ys, s=28, zorder=3)
    # Manual label offsets: the top-right cluster (bloodmnist / organamnist /
    # organcmnist / pathmnist / pneumoniamnist / breastmnist) sits within a few pixels,
    # where automatic placement collides. Negative x offsets put a label to the left.
    LABEL_OFFSETS = {
        "chestmnist": (6, 5), "bloodmnist": (7, -11), "organamnist": (7, 5),
        "organcmnist": (-38, -4), "pathmnist": (-42, -4), "pneumoniamnist": (-56, -3),
        "breastmnist": (7, -11), "organsmnist": (7, 4), "dermamnist": (9, 7),
        "octmnist": (9, -13), "tissuemnist": (7, 4), "retinamnist": (7, -4),
    }
    for x, y, name in points:
        ax.annotate(name, (x, y), textcoords="offset points",
                    xytext=LABEL_OFFSETS.get(name, (5, 4)), fontsize=6.5)
    ax.margins(x=0.14, y=0.10)
    ax.axhline(1.0, color="gray", linewidth=0.6, linestyle="--")
    ax.text(0.98, 0.02, "dashed line: full coverage is safe",
            transform=ax.transAxes, fontsize=6.5, color="gray", ha="right")
    ax.set_xlabel(f"backbone test accuracy ({size}$\\times${size})", fontsize=8)
    ax.set_ylabel(f"coverage at risk $\\leq$ {max_risk:.1f}", fontsize=8)
    ax.set_title("How much abstention buys, per subset", fontsize=9)
    ax.grid(alpha=0.3, linewidth=0.4)
    ax.tick_params(labelsize=7.5)
    fig.tight_layout()
    fig.savefig(out_dir / "Figure_5_accuracy_vs_coverage.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "Figure_5_accuracy_vs_coverage.png", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_dir / 'Figure_5_accuracy_vs_coverage.pdf'}")


def figure_split_shift(logits_dir: Path, out_dir: Path, size: int = 28) -> None:
    """Validation-to-test transfer: class-prior drift vs the accuracy gap.

    MedMNIST uses the official source split where one exists, so the validation and
    test subsets are not exchangeable; this figure shows the drift that breaks
    validation-calibrated thresholds and conformal quantiles.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = []
    for path in sorted(logits_dir.glob(f"*_resnet18_{size}_seed0_logits.npz")):
        with np.load(path, allow_pickle=False) as data:
            meta = str(data["meta"]) if "meta" in data.files else ""
            if "multi-label" in meta:
                continue
            val_labels = data["val_labels"].astype(int)
            test_labels = data["test_labels"].astype(int)
            val_logits = data["val_logits"]
            test_logits = data["test_logits"]
        name = path.name.split(f"_resnet18_{size}")[0]
        n_classes = int(max(val_labels.max(), test_labels.max())) + 1
        p_val = np.bincount(val_labels, minlength=n_classes) / val_labels.size
        p_test = np.bincount(test_labels, minlength=n_classes) / test_labels.size
        drift = float(np.abs(p_val - p_test).sum() / 2)
        val_acc = float((val_logits.argmax(axis=1) == val_labels).mean())
        test_acc = float((test_logits.argmax(axis=1) == test_labels).mean())
        points.append((drift, val_acc - test_acc, name))

    if not points:
        print("no logit bundles found; skipping the split-shift figure")
        return

    fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=300)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.scatter(xs, ys, s=26, zorder=3)
    offsets = {"octmnist": (6, -3), "pneumoniamnist": (-58, 4), "organsmnist": (8, 4),
               "pathmnist": (8, -3), "organcmnist": (-52, -11), "organamnist": (-50, 6),
               "retinamnist": (8, -3), "breastmnist": (-52, -4), "dermamnist": (-56, 7),
               "bloodmnist": (7, 1), "tissuemnist": (7, -11)}
    for x, y, name in points:
        ax.annotate(name, (x, y), textcoords="offset points",
                    xytext=offsets.get(name, (5, 4)), fontsize=6.5)
    ax.axhline(0.0, color="gray", linewidth=0.6, linestyle="--")
    ax.margins(x=0.12, y=0.14)
    ax.set_xlabel("class-prior drift between validation and test (total variation)", fontsize=8)
    ax.set_ylabel("validation accuracy $-$ test accuracy", fontsize=8)
    ax.set_title("Why validation-calibrated thresholds do not transfer", fontsize=9)
    ax.grid(alpha=0.3, linewidth=0.4)
    ax.tick_params(labelsize=7.5)
    fig.tight_layout()
    fig.savefig(out_dir / "Figure_4_split_shift.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "Figure_4_split_shift.png", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_dir / 'Figure_4_split_shift.pdf'}")


def figure_pipeline(out_dir: Path) -> None:
    """Task/pipeline schematic (paper Figure 1)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(6.4, 2.9), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    boxes = [
        (0.15, 1.6, 1.7, 1.1, "MedMNIST v2\n12 2D subsets\n11 modalities"),
        (2.15, 1.6, 1.7, 1.1, "ResNet-18\n28$\\times$28 and 128$\\times$128\n3 / 2 seeds"),
        (4.15, 1.6, 1.7, 1.1, "export logit bundles\nval/test logits, labels,\nMC-dropout moments"),
        (6.15, 1.6, 1.7, 1.1, "8 abstention scores\nsoftmax, calibration,\nensembles, conformal"),
        (8.15, 1.6, 1.7, 1.1, "risk--coverage\nAURC / AUGRC\ncoverage@risk"),
    ]
    for x, y, width, height, label in boxes:
        ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.06",
                                    linewidth=0.8, edgecolor="black", facecolor="#eef3fa"))
        ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=6.5)
    for index in range(len(boxes) - 1):
        start = boxes[index][0] + boxes[index][2]
        end = boxes[index + 1][0]
        ax.add_patch(FancyArrowPatch((start + 0.02, 2.15), (end - 0.02, 2.15),
                                     arrowstyle="->", mutation_scale=8, linewidth=0.8))
    ax.text(5.0, 3.55, "all analysis after training runs on CPU from the exported logits",
            ha="center", fontsize=6.5, style="italic", color="#444444")
    ax.text(5.0, 0.85, "GPU only trains; the benchmark is reproducible on a free compute tier",
            ha="center", fontsize=6.5, style="italic", color="#444444")
    fig.tight_layout()
    fig.savefig(out_dir / "Figure_1_pipeline.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "Figure_1_pipeline.png", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_dir / 'Figure_1_pipeline.pdf'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="outputs/results")
    parser.add_argument("--out-dir", default="paper/figures")
    parser.add_argument("--stem", default="Figure_2_risk_coverage")
    parser.add_argument("--max-coverage", type=float, default=1.0)
    parser.add_argument("--figure", default="all",
                        choices=["all", "panels", "heatmap", "futility", "pipeline",
                                 "split_shift"],
                        help="which figure(s) to build")
    parser.add_argument("--logits", default="outputs/logits")
    parser.add_argument("--max-risk", type=float, default=0.1)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.figure in ("all", "pipeline"):
        figure_pipeline(out_dir)
    if args.figure in ("all", "heatmap"):
        figure_heatmap(Path(args.results), out_dir, max_risk=args.max_risk)
    if args.figure in ("all", "futility"):
        figure_futility(Path(args.results), out_dir, max_risk=args.max_risk)
    if args.figure in ("all", "split_shift"):
        figure_split_shift(Path(args.logits), out_dir)
    if args.figure != "all" and args.figure != "panels":
        return 0

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = Path(args.results)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), dpi=300)
    for ax, (stem, title) in zip(axes.ravel(), PANELS):
        path = results / f"{stem}_risk_coverage.npz"
        if not path.exists():
            raise SystemExit(f"missing curve file: {path} (run scripts/run_selective.py first)")
        with np.load(path, allow_pickle=False) as data:
            curves = {key: data[key] for key in data.files}
        order = sorted(curves, key=lambda name: float(np.trapezoid(curves[name][1], curves[name][0])))
        for name in order:
            coverage, risk = curves[name]
            keep = coverage <= args.max_coverage
            area = float(np.trapezoid(risk, coverage))
            ax.plot(coverage[keep], risk[keep], linewidth=1.1, label=f"{name}  {area:.3f}")
        ax.set_title(title, fontsize=8.5)
        ax.set_xlim(0.0, args.max_coverage)
        ax.set_ylim(bottom=0.0)
        ax.grid(alpha=0.3, linewidth=0.4)
        ax.tick_params(labelsize=7.5)
        ax.legend(fontsize=6, frameon=False, loc="upper left", handlelength=1.4)
    for ax in axes[1, :]:
        ax.set_xlabel("coverage (accepted fraction)", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("risk on accepted cases", fontsize=8)
    fig.tight_layout()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{args.stem}.pdf"
    png_path = out_dir / f"{args.stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {pdf_path}")
    print(f"saved: {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
