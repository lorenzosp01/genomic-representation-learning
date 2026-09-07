"""Fold data preparation for the VMGP model (protocol-v2 shared layer).

The VAE data layer no longer performs any cohort/individual/marker QC,
imputation, MAF/LD pruning, breed filtering, label fitting, or train/val/test
splitting. All of that is owned by the shared ``genomic.*`` layer
(``genomic.experiment_data.GenomicExperimentData`` / ``FoldData``).

This module converts already-prepared per-fold dosage matrices (values in
{0, 1, 2}, no NaNs) into the 5-tensor batch format the VMGP LightningModule
expects, ``(x, y_breed, y_continent, y_caseina, y_attitudine)``, and (optionally)
derives the VAE-specific auxiliary labels (continent / caseina / attitudine) for
the exact same source rows supplied by ``FoldData``. Auxiliary labels never drop
samples and never influence cohort, split, preprocessing or the canonical breed
labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from genomic.experiment_data import FoldData
from genomic.preprocessing import GenomicPreprocessor


# ---------------------------------------------------------------------------
# Auxiliary label helpers (preserved from the legacy data module)
# ---------------------------------------------------------------------------
def load_metadata(metadata_path: Optional[str]) -> Optional[pd.DataFrame]:
    if metadata_path is None:
        return None
    return pd.read_csv(metadata_path)


def match_breed_to_metadata(breed_code: str, metadata_df: Optional[pd.DataFrame]) -> Dict:
    """Return ``{'continent': ..., 'caseina': ...}`` (Nones when unmatched)."""
    if metadata_df is None:
        return {"continent": None, "caseina": None}

    match = metadata_df[metadata_df["Breed"] == breed_code]
    if len(match) == 0:
        match = metadata_df[metadata_df["Breed"].str.startswith(breed_code + "_")]
    if len(match) == 0:
        prefix = breed_code.split("_")[0] if "_" in breed_code else breed_code
        match = metadata_df[metadata_df["Breed"] == prefix]

    if len(match) > 0:
        row = match.iloc[0]
        caseina = row.get("Caseina", None)
        if caseina is None or (isinstance(caseina, float) and np.isnan(caseina)) or str(caseina) == "":
            caseina = None
        return {"continent": row.get("Continent", None), "caseina": caseina}
    return {"continent": None, "caseina": None}


def get_attitudine(breed_code: str, attitudine_mapping: Optional[Dict]) -> str:
    """Map a breed to its attitudine category (default ``ALTRO``)."""
    if attitudine_mapping is None:
        return "ALTRO"
    prefix = breed_code.split("_")[0] if "_" in breed_code else breed_code
    for attitudine, breeds in attitudine_mapping.items():
        if prefix in breeds:
            return attitudine
    return "ALTRO"


def derive_auxiliary(
    breeds,
    metadata_df: Optional[pd.DataFrame],
    attitudine_mapping: Optional[Dict],
    use_continent: bool,
    use_caseina: bool,
    use_attitudine: bool,
):
    """Derive dense auxiliary labels for a row-aligned set of breed strings.

    Returns ``(y_continent, y_caseina, y_attitudine, class_names_continent,
    class_names_caseina, class_names_attitudine)``. Unavailable values are
    ``-1`` (continent/caseina) or mapped to ``ALTRO`` (attitudine). Samples are
    never dropped.
    """
    n = len(breeds)
    cont_raw = [None] * n
    cas_raw = [None] * n
    att_raw = [None] * n
    for i, b in enumerate(breeds):
        m = match_breed_to_metadata(b, metadata_df)
        cont_raw[i] = m["continent"]
        cas_raw[i] = m["caseina"]
        att_raw[i] = get_attitudine(b, attitudine_mapping)

    class_names_continent = sorted({c for c in cont_raw if c is not None})
    class_names_caseina = sorted({c for c in cas_raw if c is not None})
    class_names_attitudine = sorted({a for a in att_raw if a is not None})

    y_continent = np.full(n, -1, dtype=np.int64)
    y_caseina = np.full(n, -1, dtype=np.int64)
    y_attitudine = np.full(n, -1, dtype=np.int64)

    if use_continent:
        for i, c in enumerate(cont_raw):
            if c is not None:
                y_continent[i] = class_names_continent.index(c)
    if use_caseina:
        for i, c in enumerate(cas_raw):
            if c is not None:
                y_caseina[i] = class_names_caseina.index(c)
    if use_attitudine:
        for i, a in enumerate(att_raw):
            y_attitudine[i] = class_names_attitudine.index(a)

    return (
        y_continent, y_caseina, y_attitudine,
        class_names_continent, class_names_caseina, class_names_attitudine,
    )


# ---------------------------------------------------------------------------
# Fold data container + builder
# ---------------------------------------------------------------------------
@dataclass
class VAEFoldData:
    """Prepared train/validation data for one VMGP CV fold.

    Breed labels are the canonical dense labels from the shared layer. Auxiliary
    labels are derived for the same source rows. ``preprocessor`` was fit on
    this fold's training rows only and carries the SNP identity chain.
    """

    fold: int
    preprocessor: GenomicPreprocessor
    num_snps: int
    num_classes_breed: int
    class_names_breed: List[str]
    num_classes_continent: int
    num_classes_caseina: int
    num_classes_attitudine: int
    class_names_continent: List[str]
    class_names_caseina: List[str]
    class_names_attitudine: List[str]
    X_train: np.ndarray
    y_breed_train: np.ndarray
    y_continent_train: np.ndarray
    y_caseina_train: np.ndarray
    y_attitudine_train: np.ndarray
    X_val: np.ndarray
    y_breed_val: np.ndarray
    y_continent_val: np.ndarray
    y_caseina_val: np.ndarray
    y_attitudine_val: np.ndarray
    train_source_index: np.ndarray
    val_source_index: np.ndarray
    train_breed: np.ndarray
    val_breed: np.ndarray

    @property
    def retained_snp_indices(self) -> np.ndarray:
        return self.preprocessor.retained_snp_indices_


def build_vae_fold(
    fold_data: FoldData,
    *,
    class_names_breed: List[str],
    metadata_path: Optional[str] = None,
    attitudine_mapping: Optional[Dict] = None,
    use_continent: bool = False,
    use_caseina: bool = False,
    use_attitudine: bool = False,
) -> VAEFoldData:
    """Build the VAE-specific fold data from a shared ``FoldData``.

    The breed labels and dosage matrices come straight from ``fold_data``; the
    auxiliary labels are derived for exactly ``fold_data.train_breed`` /
    ``fold_data.val_breed`` (never dropping samples).
    """
    metadata_df = load_metadata(metadata_path)

    (yc_tr, ycas_tr, yatt_tr, names_cont, names_cas, names_att) = derive_auxiliary(
        fold_data.train_breed, metadata_df, attitudine_mapping,
        use_continent, use_caseina, use_attitudine,
    )
    (yc_va, ycas_va, yatt_va, _, _, _) = derive_auxiliary(
        fold_data.val_breed, metadata_df, attitudine_mapping,
        use_continent, use_caseina, use_attitudine,
    )

    return VAEFoldData(
        fold=fold_data.fold,
        preprocessor=fold_data.preprocessor,
        num_snps=int(fold_data.X_train.shape[1]),
        num_classes_breed=len(class_names_breed),
        class_names_breed=list(class_names_breed),
        num_classes_continent=len(names_cont) if use_continent else 0,
        num_classes_caseina=len(names_cas) if use_caseina else 0,
        num_classes_attitudine=len(names_att) if use_attitudine else 0,
        class_names_continent=names_cont if use_continent else [],
        class_names_caseina=names_cas if use_caseina else [],
        class_names_attitudine=names_att if use_attitudine else [],
        X_train=fold_data.X_train,
        y_breed_train=np.asarray(fold_data.y_train, dtype=np.int64),
        y_continent_train=yc_tr,
        y_caseina_train=ycas_tr,
        y_attitudine_train=yatt_tr,
        X_val=fold_data.X_val,
        y_breed_val=np.asarray(fold_data.y_val, dtype=np.int64),
        y_continent_val=yc_va,
        y_caseina_val=ycas_va,
        y_attitudine_val=yatt_va,
        train_source_index=fold_data.train_source_index,
        val_source_index=fold_data.val_source_index,
        train_breed=fold_data.train_breed,
        val_breed=fold_data.val_breed,
    )


# ---------------------------------------------------------------------------
# Tensor / DataLoader construction
# ---------------------------------------------------------------------------
def build_fold_datasets(vfd: VAEFoldData):
    """Build the 5-tensor (train, val) TensorDatasets for one fold."""
    train_ds = TensorDataset(
        torch.FloatTensor(vfd.X_train),
        torch.LongTensor(vfd.y_breed_train),
        torch.LongTensor(vfd.y_continent_train),
        torch.LongTensor(vfd.y_caseina_train),
        torch.LongTensor(vfd.y_attitudine_train),
    )
    val_ds = TensorDataset(
        torch.FloatTensor(vfd.X_val),
        torch.LongTensor(vfd.y_breed_val),
        torch.LongTensor(vfd.y_continent_val),
        torch.LongTensor(vfd.y_caseina_val),
        torch.LongTensor(vfd.y_attitudine_val),
    )
    return train_ds, val_ds


def build_fold_dataloaders(vfd: VAEFoldData, batch_size: int, *, num_workers=4, pin_memory=True):
    """Shuffled train / unshuffled validation DataLoaders for one fold."""
    train_ds, val_ds = build_fold_datasets(vfd)
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return train_dl, val_dl
