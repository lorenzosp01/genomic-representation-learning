"""Fold data loading for the contrastive model (protocol-v2 shared layer).

The contrastive data layer no longer performs any cohort/individual/marker QC,
imputation, MAF/LD pruning, breed filtering, label fitting, or train/val/test
splitting. All of that is owned by the shared ``genomic.*`` layer
(``genomic.experiment_data.GenomicExperimentData`` / ``FoldData``).

This module only converts already-prepared per-fold dosage matrices (values in
{0, 1, 2}, no NaNs) into tensors and DataLoaders. The dosage -> 4-channel
one-hot + allele-flip/mask augmentation remains model-specific and lives in
``GeneticAugmentation`` (inside ``ContrastiveGeneticModel``), applied AFTER this
layer.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset


def fold_tensors(fold_data):
    """Convert a :class:`~genomic.experiment_data.FoldData` into torch tensors.

    Returns ``(X_train, y_train, X_val, y_val)`` where ``X_*`` are float32
    dosage tensors (values in {0, 1, 2}) and ``y_*`` are int64 canonical labels.
    """
    X_train = torch.FloatTensor(fold_data.X_train)
    y_train = torch.LongTensor(fold_data.y_train)
    X_val = torch.FloatTensor(fold_data.X_val)
    y_val = torch.LongTensor(fold_data.y_val)
    return X_train, y_train, X_val, y_val


def build_fold_dataloaders(fold_data, batch_size, *, num_workers=4, pin_memory=True):
    """Build shuffled train / unshuffled validation DataLoaders for one fold.

    ``batch_size`` is the caller's responsibility (the runner bounds it to
    prevent OOM for the full-batch N-pair objective).
    """
    X_train, y_train, X_val, y_val = fold_tensors(fold_data)
    train_dl = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_dl = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_dl, val_dl
