"""Model-neutral RQ1 configuration selection and checkpoint utilities.

The frozen PRIMARY RQ1 selection policy ranks configurations by:

    1. mean Macro-F1 across all folds (primary)
    2. mean Balanced Accuracy (first tie-break)
    3. mean Accuracy (second tie-break)
    4. deterministic configuration order (original list order)

This is deliberately independent of any training/checkpoint metric (``val_loss``)
and of any legacy "best fold" score (GE / self-consistency KNN).
"""

from __future__ import annotations

import re
from typing import List, Tuple

SELECTION_KEYS = ("Macro_F1 (mean)", "Balanced_Accuracy (mean)", "Accuracy (mean)")


def select_best_configuration(results: List[dict]) -> Tuple[int, dict]:
    """Return ``(best_index, best_result)`` per the frozen RQ1 policy.

    Results are expected to be result dicts exposing the common metric fields
    (``Macro_F1 (mean)``, ``Balanced_Accuracy (mean)``, ``Accuracy (mean)``).
    Missing metrics are treated as ``-inf``. Ties fall through the keys in order
    and finally to the original list order (deterministic, first wins).
    """
    if not results:
        raise ValueError("select_best_configuration requires at least one result.")

    def score(r: dict):
        return tuple(float(r.get(k, float("-inf"))) for k in SELECTION_KEYS)

    best_index = max(range(len(results)), key=lambda i: score(results[i]))
    return best_index, results[best_index]


def best_epoch_from_checkpoint_path(ckpt_path: str) -> int:
    """Return the 1-based number of completed training epochs for a checkpoint.

    Preferred source of truth is the zero-based ``epoch`` stored in the
    Lightning checkpoint metadata; the returned value is ``stored_epoch + 1``
    (so checkpoint epoch 0 -> 1 completed epoch, epoch 99 -> 100). If the
    checkpoint cannot be loaded (e.g. a non-existent path in a test), fall back
    to parsing the ``epoch=NNNN`` field from the filename, which is explicitly
    controlled by this project (``filename='{epoch:04d}-{val_loss:.4f}'``).
    """
    try:
        import torch

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "epoch" in ckpt:
            return int(ckpt["epoch"]) + 1
    except Exception:
        pass

    m = re.search(r"epoch=(\d+)", ckpt_path)
    if m is None:
        raise ValueError(f"Cannot determine best epoch from checkpoint: {ckpt_path!r}")
    return int(m.group(1)) + 1
