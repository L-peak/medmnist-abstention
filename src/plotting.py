"""Paper-ready risk-coverage figures (matplotlib, non-interactive backend)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_risk_coverage(curves: dict, out_path, title: str | None = None,
                       max_coverage: float = 1.0, dpi: int = 300) -> Path:
    """Draw one risk-coverage curve per method.

    `curves` maps method name -> array of shape (2, n_points) holding
    (coverage, risk), exactly what run_analysis() returns.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.0), dpi=dpi)

    order = sorted(curves, key=lambda name: float(np.trapezoid(curves[name][1], curves[name][0])))
    for name in order:
        data = np.asarray(curves[name], dtype=float)
        coverage, risk = data[0], data[1]
        keep = coverage <= max_coverage
        label = f"{name} (AURC={np.trapezoid(risk, coverage):.3f})"
        ax.plot(coverage[keep], risk[keep], linewidth=1.4, label=label)

    ax.set_xlabel("coverage (fraction of cases accepted)")
    ax.set_ylabel("risk (1 - accuracy on accepted cases)")
    ax.set_xlim(0.0, max_coverage)
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.3, linewidth=0.5)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
