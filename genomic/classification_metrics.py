"""Model-neutral breed-classification metrics shared by VAE and contrastive.

Primary metric: Macro-F1. Secondary: Balanced Accuracy, Accuracy. Diagnostic:
per-class recall and confusion matrix.

All metrics are computed over an explicit canonical label set so that a
confusion matrix always has the full vocabulary dimensions (34 x 34 for the
primary breed task) and per-class recall is ordered by canonical class index.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score


def compute_classification_metrics(y_true, y_pred, *, labels) -> Dict:
    """Compute common classification metrics over a fixed canonical label set.

    Parameters
    ----------
    y_true, y_pred:
        Integer arrays of true / predicted class labels.
    labels:
        Canonical class indices (e.g. ``np.arange(34)`` for the primary breed
        task), fixing confusion-matrix dimensions and per-class-recall ordering.
        Classes with zero predicted (or zero true) samples still occupy their
        row/column.

    Returns
    -------
    dict with keys:
        ``accuracy`` (float),
        ``macro_f1`` (float),
        ``balanced_accuracy`` (float, mean of per-class recall over ``labels``),
        ``per_class_recall`` (float64 array, length ``len(labels)``),
        ``confusion_matrix`` (int64 array, ``(len(labels), len(labels))``),
        ``labels`` (int64 array).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.asarray(labels, dtype=np.int64)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(
        f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    )
    per_class_recall = recall_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    ).astype(np.float64)
    balanced_accuracy = float(np.mean(per_class_recall))

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
        "per_class_recall": per_class_recall,
        "confusion_matrix": cm.astype(np.int64),
        "labels": labels,
    }


def aggregate_folds(fold_metrics: Sequence[Dict]) -> Dict:
    """Aggregate per-fold metric dicts into mean/std + summed confusion matrix.

    Scalars (``accuracy``, ``macro_f1``, ``balanced_accuracy``) -> mean and std.
    ``per_class_recall`` -> mean/std across folds (breed-aligned canonical order).
    ``confusion_matrix`` -> elementwise sum across folds (raw counts, not
    normalized).
    """
    acc = np.array([m["accuracy"] for m in fold_metrics], dtype=np.float64)
    mf = np.array([m["macro_f1"] for m in fold_metrics], dtype=np.float64)
    ba = np.array([m["balanced_accuracy"] for m in fold_metrics], dtype=np.float64)
    pcr = np.stack([m["per_class_recall"] for m in fold_metrics])
    cms = np.stack([m["confusion_matrix"] for m in fold_metrics])

    return {
        "accuracy_mean": float(acc.mean()),
        "accuracy_std": float(acc.std()),
        "macro_f1_mean": float(mf.mean()),
        "macro_f1_std": float(mf.std()),
        "balanced_accuracy_mean": float(ba.mean()),
        "balanced_accuracy_std": float(ba.std()),
        "per_class_recall_mean": pcr.mean(axis=0),
        "per_class_recall_std": pcr.std(axis=0),
        "confusion_matrix_sum": cms.sum(axis=0),
    }
