"""Shared utilities for the overnight Sheep HapMap Phase-1 pipeline.

Guarantees:
    * headless matplotlib (Agg) is selected before pyplot is imported;
    * ``plt.show`` is a no-op everywhere in the sheep pipeline;
    * every figure is saved as PNG (300 DPI) + PDF under
      ``results/sheep_cross_species/plots/``;
    * plotting errors are caught and logged, never crash the training loop;
    * all progress/metrics are written to stdout AND
      ``results/sheep_cross_species/execution.log``.
"""

from __future__ import annotations

import os
import traceback

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402

plt.show = lambda *args, **kwargs: None  # hard no-op for unattended runs

# Publisher-friendly Type 42 fonts (embed TrueType, searchable text).
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

PLOTS_DIR = "results/sheep_cross_species/plots"
EXEC_LOG = "results/sheep_cross_species/execution.log"


class TeeLogger:
    """Print to stdout and append the same lines to the execution log."""

    def __init__(self, path: str = EXEC_LOG):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self._fh = open(path, "a", buffering=1)

    def log(self, msg: str = "") -> None:
        text = str(msg)
        print(text, flush=True)
        self._fh.write(text + "\n")
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


def save_figure(fig, stem: str, logger: TeeLogger | None = None,
                plots_dir: str = PLOTS_DIR) -> None:
    """Save PNG (300 dpi) + PDF; never raise."""
    try:
        os.makedirs(plots_dir, exist_ok=True)
        png = os.path.join(plots_dir, stem + ".png")
        pdf = os.path.join(plots_dir, stem + ".pdf")
        fig.savefig(png, dpi=300, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
        if logger is not None:
            logger.log(f"[plots] saved {png} + {pdf}")
    except Exception:
        if logger is not None:
            logger.log(f"[plots] ERROR saving {stem}:\n{traceback.format_exc()}")
    finally:
        try:
            plt.close(fig)
        except Exception:
            pass


def guarded_plot(fn, label: str, logger: TeeLogger | None = None) -> None:
    """Run a plotting function; log (never propagate) any failure."""
    try:
        fn()
    except Exception:
        if logger is not None:
            logger.log(f"[plots] ERROR in {label}:\n{traceback.format_exc()}")
        else:
            traceback.print_exc()
