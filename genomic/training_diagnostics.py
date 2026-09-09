"""Model-neutral convergence-history instrumentation for the RQ1 pilots.

Records per-epoch ``train_loss`` / ``val_loss`` / ``learning_rate`` via a
Lightning callback, then summarises them into the ceiling diagnostics used to
decide whether the current training ceilings bind.

This instrumentation must not alter training behaviour: it only reads logged
metrics and the early-stopping state.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Optional

import numpy as np
from pytorch_lightning import Callback

#: best_epoch / max_epochs ratio above which a non-early-stopped run is
#: considered suspicious of being ceiling-limited (used together with the
#: final-10% val_loss trend, never alone).
CEILING_RATIO_THRESHOLD = 0.80
#: fraction of val_loss observations used for the tail-trend slope.
TAIL_FRACTION = 0.10

HISTORY_FIELDS = ["epoch", "train_loss", "val_loss", "learning_rate"]


def _to_float(v):
    if v is None:
        return None
    if hasattr(v, "item"):
        v = v.item()
    return float(v)


def _logged(trainer, key: str) -> Optional[float]:
    v = trainer.callback_metrics.get(key)
    if v is None:
        v = trainer.logged_metrics.get(key)
    return _to_float(v)


def _current_lr(trainer) -> Optional[float]:
    optimizers = getattr(trainer, "optimizers", None)
    if not optimizers:
        return None
    for pg in optimizers[0].param_groups:
        return _to_float(pg.get("lr"))
    return None


class ConvergenceHistoryCallback(Callback):
    """Capture epoch-level convergence history and early-stopping outcome.

    Produces one row per completed epoch with ``epoch`` (0-based Lightning
    epoch index), ``train_loss``, ``val_loss`` (both ``None`` when unavailable)
    and ``learning_rate`` (read from the optimizer at epoch start).
    """

    def __init__(self):
        self.history: List[dict] = []
        # Number of completed training epochs (set in ``on_train_end``).
        self.stopped_epoch: Optional[int] = None
        self.early_stopping_triggered: bool = False

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        self.history.append({
            "epoch": int(trainer.current_epoch),
            "train_loss": None,
            "val_loss": None,
            "learning_rate": _current_lr(trainer),
        })

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or not self.history:
            return
        self.history[-1]["train_loss"] = _logged(trainer, "train_loss")

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or not self.history:
            return
        self.history[-1]["val_loss"] = _logged(trainer, "val_loss")

    def on_train_end(self, trainer, pl_module):
        esc = getattr(trainer, "early_stopping_callback", None)
        # EarlyStopping status comes from the actual callback state: Lightning
        # sets ``stopped_epoch`` (0-based) only when the callback itself caused
        # the stop; with patience >= 1 it can never fire at epoch 0.
        self.early_stopping_triggered = bool(
            esc is not None and int(getattr(esc, "stopped_epoch", 0)) > 0
        )
        # Number of completed training epochs: the history records exactly one
        # row per completed epoch, so it is the source of truth (robust to any
        # non-EarlyStopping early termination, not just natural completion).
        self.stopped_epoch = len(self.history)

    def rows(self) -> List[dict]:
        return [
            {k: r.get(k) for k in HISTORY_FIELDS}
            for r in self.history
        ]


def save_history_csv(rows: List[dict], path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in HISTORY_FIELDS})
    return path


def save_summary_json(summary: dict, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    clean = {}
    for k, v in summary.items():
        if hasattr(v, "item"):
            v = v.item()
        clean[k] = v
    with open(path, "w") as f:
        json.dump(clean, f, indent=2, sort_keys=True)
    return path


def tail_slope(val_losses, fraction: float = TAIL_FRACTION) -> Optional[float]:
    """Least-squares slope of the final ``fraction`` of non-None val_loss values.

    A negative slope only indicates that validation loss is *on average*
    decreasing over the tail. It is NOT by itself proof of meaningful
    improvement: the final pilot decision must also inspect the magnitude of
    the slope and the actual loss curve (a tiny drift on a noisy plateau is
    not a real descent).
    """
    vals = [v for v in val_losses if v is not None]
    if len(vals) < 2:
        return None
    k = max(2, int(round(len(vals) * fraction)))
    tail = np.asarray(vals[-k:], dtype=np.float64)
    x = np.arange(len(tail), dtype=np.float64)
    if np.ptp(x) == 0:
        return 0.0
    slope, _ = np.polyfit(x, tail, 1)
    return float(slope)


def classify_ceiling(early_stopping_triggered: bool,
                     best_epoch_ratio: Optional[float],
                     tail_slope_value: Optional[float]) -> str:
    """Classify a run as CONVERGED / UNCERTAIN / CEILING_SUSPICIOUS.

    ``CEILING_SUSPICIOUS`` uses the tail slope only as a coarse "still
    decreasing" signal. A negative slope alone is NOT proof of meaningful
    improvement; the human decision must additionally inspect the slope
    magnitude and the full loss curve before adjusting any training ceiling.
    """
    if early_stopping_triggered:
        return "CONVERGED"
    if best_epoch_ratio is None or tail_slope_value is None:
        return "UNCERTAIN"
    if best_epoch_ratio >= CEILING_RATIO_THRESHOLD and tail_slope_value < 0.0:
        return "CEILING_SUSPICIOUS"
    return "UNCERTAIN"


def summarize_convergence(history: List[dict], *, best_epoch: Optional[int],
                          stopped_epoch: Optional[int], max_epochs: int,
                          early_stopping_triggered: bool) -> dict:
    """Compute the pilot diagnostics from a convergence history.

    ``best_epoch`` is 1-based (number of completed epochs); ``history`` rows are
    keyed by the 0-based Lightning epoch index.
    """
    best_epoch_ratio = (best_epoch / max_epochs) if (best_epoch is not None and max_epochs) else None
    epochs_since_best = (max_epochs - best_epoch) if best_epoch is not None else None

    val_losses = [r.get("val_loss") for r in history]
    slope = tail_slope(val_losses)

    lr_at_best_epoch = None
    if best_epoch is not None:
        idx = best_epoch - 1  # 1-based -> 0-based
        if 0 <= idx < len(history):
            lr_at_best_epoch = history[idx].get("learning_rate")

    return {
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "max_epochs": max_epochs,
        "best_epoch_ratio": best_epoch_ratio,
        "epochs_since_best": epochs_since_best,
        "early_stopping_triggered": early_stopping_triggered,
        "lr_at_best_epoch": lr_at_best_epoch,
        "tail_slope": slope,
        "ceiling_status": classify_ceiling(early_stopping_triggered, best_epoch_ratio, slope),
    }
