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
    """Parse the 0-based Lightning epoch index from a checkpoint filename.

    Lightning stores ``trainer.current_epoch`` (0-based) in the ``{epoch}``
    filename field. The returned value is the 1-based training-epoch count
    (``parsed + 1``), so ``epoch=0000`` -> ``1``.
    """
    m = re.search(r"epoch=(\d+)", ckpt_path)
    if m is None:
        raise ValueError(f"Cannot parse epoch from checkpoint path: {ckpt_path!r}")
    return int(m.group(1)) + 1
