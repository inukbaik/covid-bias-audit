"""Shared matplotlib style and figure helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

# Okabe-Ito colour-blind-safe palette.
PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "sky": "#56B4E9",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "grey": "#7F7F7F",
}
COLORS = list(PALETTE.values())
DPI = 200


def apply_style() -> None:
    """Set the package-wide rcParams (idempotent)."""
    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": matplotlib.cycler(color=COLORS),
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.axisbelow": True,
            "font.size": 10,
            "axes.titlesize": 11,
            "legend.frameon": False,
        }
    )


def new_figure(width: float = 7.0, height: float = 4.5, **kwargs):
    """Create a styled figure/axes pair."""
    apply_style()
    return plt.subplots(figsize=(width, height), **kwargs)


def plain_log_axis(ax, axis: str = "x", ticks=(0.2, 0.3, 0.5, 1, 2, 3, 5, 10)) -> None:
    """Log-scale axis with plain decimal tick labels and no minor labels."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    target = ax.xaxis if axis == "x" else ax.yaxis
    lo, hi = ax.get_xlim() if axis == "x" else ax.get_ylim()
    shown = [t for t in ticks if lo <= t <= hi] or list(ticks)
    target.set_major_locator(FixedLocator(shown))
    target.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    target.set_minor_formatter(NullFormatter())


def save(fig, path: Path) -> Path:
    """Save a figure (creating parents) and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path
