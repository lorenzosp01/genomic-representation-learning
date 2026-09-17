"""Downstream panel evaluation and aggregation for RQ3/RQ4 (protocol-v2).

The downstream evaluator is frozen for every panel size and every ranking
method:

    RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

It is fitted strictly on the fold TRAINING partition and evaluated on the fold
VALIDATION partition. These helpers never accept, load or index the locked
test set: a panel is an explicit list of markers of the training partition and
the evaluation rows are the fold's validation rows.

Classification metrics are computed over the canonical ``np.arange(n_classes)``
label set (34 for the primary breed cohort) so Macro-F1 / Balanced Accuracy are
comparable across panel sizes even when a very small panel never predicts some
breed in a given validation fold.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .classification_metrics import compute_classification_metrics

#: frozen downstream-evaluator hyperparameters (protocol-v2)
RF_N_ESTIMATORS = 200
RF_RANDOM_STATE = 42


def _check_xy(X, y, name_X: str, name_y: str):
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2:
        raise ValueError(f"{name_X} must be a 2D array (samples x markers).")
    if y.shape != (X.shape[0],):
        raise ValueError(
            f"{name_y} must have shape ({X.shape[0]},) matching {name_X} rows, "
            f"got {y.shape}."
        )
    return X, y


def _check_selected_snp_indices(
    selected_snp_indices, n_features: int
) -> np.ndarray:
    """Validate an explicit panel: 1D, non-empty, unique, in-range integer indices."""
    idx = np.asarray(selected_snp_indices)
    if idx.ndim != 1:
        raise ValueError(
            f"selected_snp_indices must be 1D (K,), got shape {idx.shape}."
        )
    if idx.size == 0:
        raise ValueError("selected_snp_indices must contain at least one marker.")
    if not np.issubdtype(idx.dtype, np.integer):
        raise ValueError("selected_snp_indices must be integer feature indices.")
    if idx.min() < 0 or idx.max() >= n_features:
        raise ValueError(
            f"selected_snp_indices out of range [0, {n_features}): "
            f"min={int(idx.min())}, max={int(idx.max())}."
        )
    if np.unique(idx).size != idx.size:
        raise ValueError("selected_snp_indices contains duplicate indices.")
    return idx.astype(np.int64)


def evaluate_panel(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    selected_snp_indices: np.ndarray,
    random_state: int = 42,
    *,
    n_classes: Optional[int] = None,
) -> Dict[str, float]:
    """Fit the frozen downstream RF on a panel and evaluate it on validation.

    Parameters
    ----------
    X_train, y_train:
        Fold training dosages ``(N_train, P)`` and integer breed labels.
    X_val, y_val:
        Fold validation dosages ``(N_val, P)`` and integer breed labels.
    selected_snp_indices:
        ``(K,)`` integer indices into the post-QC feature axis (P).
    random_state:
        Random state of the downstream RF (default 42; fixed by protocol-v2).
    n_classes:
        Optional canonical class count. If ``None`` it is inferred as
        ``max(max(y_train), max(y_val)) + 1``. The primary cohort always passes
        the explicit 34.

    Returns
    -------
    dict with ``macro_f1``, ``balanced_accuracy``, ``accuracy`` (floats in
    [0, 1], computed over the canonical ``np.arange(n_classes)`` label set).
    """
    X_train, y_train = _check_xy(X_train, y_train, "X_train", "y_train")
    X_val, y_val = _check_xy(X_val, y_val, "X_val", "y_val")
    if X_train.shape[1] != X_val.shape[1]:
        raise ValueError(
            f"X_train has {X_train.shape[1]} markers but X_val has "
            f"{X_val.shape[1]}."
        )

    idx = _check_selected_snp_indices(selected_snp_indices, X_train.shape[1])

    if n_classes is None:
        n_classes = int(max(y_train.max(), y_val.max())) + 1
    labels = np.arange(int(n_classes), dtype=np.int64)

    X_tr = X_train[:, idx]
    X_v = X_val[:, idx]

    clf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS, random_state=random_state, n_jobs=-1
    )
    clf.fit(X_tr, y_train)
    y_pred = clf.predict(X_v)

    metrics = compute_classification_metrics(y_val, y_pred, labels=labels)
    return {
        "macro_f1": float(metrics["macro_f1"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "accuracy": float(metrics["accuracy"]),
    }


def evaluate_full_panel(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    random_state: int = 42,
    *,
    n_classes: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate the downstream RF on every post-QC marker of the fold (S_full).

    Equivalent to :func:`evaluate_panel` with ``selected_snp_indices`` covering
    all available columns; the full panel is never a selected subset.
    """
    X_train_arr = np.asarray(X_train)
    if X_train_arr.ndim != 2:
        raise ValueError("X_train must be a 2D array (samples x markers).")
    all_indices = np.arange(X_train_arr.shape[1], dtype=np.int64)
    return evaluate_panel(
        X_train, y_train, X_val, y_val, all_indices,
        random_state=random_state, n_classes=n_classes,
    )


def aggregate_panel_records(
    records: Sequence[dict],
    *,
    skip_methods: Sequence[str] = ("full",),
) -> List[dict]:
    """Aggregate per-(method, K) records into mean/std across observations.

    Standard deviations use ``ddof=1`` (sample std) when more than one
    observation is available, otherwise 0. ``n_observations`` is the number of
    folds (deterministic methods) or folds x random replicates (random
    baseline). Rows whose method is in ``skip_methods`` are ignored, so a
    per-fold full-panel baseline with differing marker counts is never averaged
    into a single pseudo-K row.
    """
    groups: Dict[tuple, List[dict]] = {}
    for r in records:
        if r["method"] in skip_methods:
            continue
        key = (str(r["method"]), int(r["K"]))
        groups.setdefault(key, []).append(r)

    rows: List[dict] = []
    for (method, k), recs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        row = {"method": method, "K": int(k), "n_observations": len(recs)}
        for metric in ("macro_f1", "balanced_accuracy", "accuracy"):
            vals = np.asarray([float(r[metric]) for r in recs], dtype=np.float64)
            if not np.all(np.isfinite(vals)):
                raise ValueError(
                    f"non-finite {metric} values for method={method}, K={k}: "
                    f"{vals.tolist()}"
                )
            ddof = 1 if vals.size > 1 else 0
            row[f"{metric}_mean"] = float(vals.mean())
            row[f"{metric}_std"] = float(vals.std(ddof=ddof))
        rows.append(row)
    return rows


def retention_threshold(s_full_macro_f1: float, epsilon: float) -> float:
    """Predefined retention threshold ``(1 - epsilon) * S_full``."""
    if not np.isfinite(s_full_macro_f1):
        raise ValueError("s_full_macro_f1 must be finite.")
    if not 0.0 <= epsilon < 1.0:
        raise ValueError("epsilon must be in [0, 1).")
    return (1.0 - float(epsilon)) * float(s_full_macro_f1)


def minimum_panel_size(
    method: str,
    aggregates: Sequence[dict],
    s_full_macro_f1: float,
    *,
    epsilon: float,
    k_grid: Sequence[int],
) -> Optional[int]:
    """Smallest K in ``k_grid`` retaining ``(1 - epsilon) * S_full`` Macro-F1.

    Criterion fixed before interpreting results: a method's aggregated
    ``macro_f1_mean`` at K must be ``>= (1 - epsilon) * S_full`` (mean S_full
    across folds). Returns ``None`` when no K in the grid qualifies; never
    invents a panel size outside the frozen grid.
    """
    threshold = retention_threshold(s_full_macro_f1, epsilon)
    by_k = {
        int(r["K"]): float(r["macro_f1_mean"])
        for r in aggregates
        if r["method"] == method
    }
    for k in sorted(int(k) for k in k_grid):
        if k in by_k and by_k[k] >= threshold:
            return k
    return None
