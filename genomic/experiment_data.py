"""Model-neutral shared experiment-data assembly.

This layer ties the protocol-v2 :class:`~genomic.splitting.SplitIndex` to the
raw PLINK genotype matrix and the train-only
:class:`~genomic.preprocessing.GenomicPreprocessor`, producing per-fold
train/validation dosage matrices plus source and SNP identity.

Design invariants (frozen protocol):

* Source row identity is ``SplitIndex.source_index`` (original PLINK row). No
  sorting, no row-order reconstruction, no resetting of source identity.
* Preprocessing is fit on the fold's TRAINING rows only; validation is only ever
  transformed. Different folds may retain different SNP panels.
* The output is always the shared dosage representation (values in {0, 1, 2},
  no NaNs). Model-specific representations (VMGP dosage, contrastive 4-channel
  one-hot + augmentation) are built downstream, never here.
* The locked test set is not reachable from the fold loader; it is available
  only through the explicit ``SplitIndex.test_indices()`` (final-evaluation flow
  is out of scope for this layer).
* Every fold keeps its fitted preprocessor so transformed feature index ->
  raw SNP index -> ``.bim`` SNP ID remains available for RQ3/RQ4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Optional

import numpy as np
from bed_reader import open_bed

from .labels import CanonicalLabelMapper
from .preprocessing import GenomicPreprocessor, load_bim
from .splitting import SplitIndex


def load_genotype_matrix(bed_path: str, fam_path: Optional[str] = None) -> np.ndarray:
    """Read the raw PLINK genotype matrix in ``.fam`` row order.

    Returns ``float32`` ``(n_samples, n_snps)`` with missing genotypes as
    ``NaN``. Row ``i`` corresponds to ``.fam`` row ``i``, so
    ``SplitIndex.source_index`` indexes directly into the returned rows.

    If ``fam_path`` is provided the sample count is checked and the matrix is
    transposed if ``bed_reader`` returned it transposed (defensive, matching the
    existing cohort loader). If ``fam_path`` is ``None`` the matrix is returned
    as-is (sample-major, the ``bed_reader`` default).
    """
    G = open_bed(bed_path, count_A1=True)
    X = G.read().astype(np.float32)
    if fam_path is not None:
        fam = np.loadtxt(fam_path, dtype=str, ndmin=2)
        n_samples = fam.shape[0]
        if X.ndim != 2:
            raise ValueError(f"Unexpected genotype matrix shape: {X.shape}")
        if X.shape[0] != n_samples and X.shape[1] == n_samples:
            X = X.T
        if X.shape[0] != n_samples:
            raise ValueError(
                f"Genotype matrix has {X.shape[0]} rows but .fam has "
                f"{n_samples} samples; cannot align rows."
            )
    return X


@dataclass
class FoldData:
    """Train/validation data for one CV fold, preprocessed train-only.

    All SNP arrays are the shared dosage representation (float32 in {0, 1, 2}).
    ``preprocessor`` was fit on this fold's training rows only and carries the
    SNP identity chain (``retained_snp_indices_`` / ``retained_snp_metadata_``).
    """

    split: SplitIndex
    fold: int
    preprocessor: GenomicPreprocessor
    train_positions: np.ndarray  # positions into the eligible cohort
    val_positions: np.ndarray
    X_train: np.ndarray
    X_val: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray

    # ------------------------------------------------------------------
    # Source identity (source_index -> raw PLINK row -> FID/IID/sample_id/breed)
    # ------------------------------------------------------------------
    @property
    def train_source_index(self) -> np.ndarray:
        return self.split.source_index[self.train_positions]

    @property
    def val_source_index(self) -> np.ndarray:
        return self.split.source_index[self.val_positions]

    @property
    def train_sample_id(self) -> np.ndarray:
        return self.split.sample_id[self.train_positions]

    @property
    def val_sample_id(self) -> np.ndarray:
        return self.split.sample_id[self.val_positions]

    @property
    def train_breed(self) -> np.ndarray:
        return self.split.breed[self.train_positions]

    @property
    def val_breed(self) -> np.ndarray:
        return self.split.breed[self.val_positions]

    # ------------------------------------------------------------------
    # SNP identity (transformed feature index -> raw SNP index -> .bim SNP ID)
    # ------------------------------------------------------------------
    @property
    def n_features(self) -> int:
        return int(self.X_train.shape[1])

    @property
    def retained_snp_indices(self) -> np.ndarray:
        return self.preprocessor.retained_snp_indices_

    @property
    def retained_snp_metadata(self) -> Optional[Dict[str, np.ndarray]]:
        return self.preprocessor.retained_snp_metadata_


class GenomicExperimentData:
    """Assemble per-fold experiment data from a protocol-v2 split.

    The raw genotype matrix is loaded/kept once (eligible rows only); each call
    to :meth:`fold_data` fits a fresh ``GenomicPreprocessor`` on that fold's
    training rows and transforms train + validation. No test data is ever
    touched by the fold loader.
    """

    def __init__(
        self,
        split: SplitIndex,
        raw_X: np.ndarray,
        *,
        snp_metadata: Optional[Dict[str, np.ndarray]] = None,
        preprocessor_kwargs: Optional[Dict] = None,
        label_mapper: Optional[CanonicalLabelMapper] = None,
    ):
        """
        Parameters
        ----------
        split:
            The persisted protocol-v2 ``SplitIndex``.
        raw_X:
            Raw genotype matrix ``(n_raw_samples, P)`` float32 in ``.fam`` row
            order with missing values as ``NaN``. Row ``i`` is ``.fam`` row ``i``.
        snp_metadata:
            Optional ``load_bim``-style metadata dict (length = raw SNP count).
        preprocessor_kwargs:
            Optional overrides for ``GenomicPreprocessor`` (defaults already
            match the frozen protocol).
        label_mapper:
            Optional canonical breed mapper; defaults to one built from the
            split's breeds (sorted-alphabetical dense 0..K-1).
        """
        raw_X = np.asarray(raw_X, dtype=np.float32)
        if raw_X.ndim != 2:
            raise ValueError("raw_X must be a 2D array (samples x SNPs).")
        n_raw = raw_X.shape[0]
        if int(split.source_index.max()) >= n_raw:
            raise ValueError(
                "split.source_index references rows outside the provided "
                f"genotype matrix (matrix has {n_raw} rows)."
            )

        self.split = split
        self.preprocessor_kwargs = dict(preprocessor_kwargs or {})
        self._snp_metadata = snp_metadata
        self.label_mapper = label_mapper or CanonicalLabelMapper.from_breeds(split.breed)

        # Keep only the eligible cohort, in split order, aligned with the split
        # arrays by position. source_index -> raw row is preserved.
        self._X = raw_X[split.source_index]
        self._y = self.label_mapper.encode(split.breed)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_plink(
        cls,
        split: SplitIndex,
        bed_path: str,
        *,
        fam_path: Optional[str] = None,
        bim_path: Optional[str] = None,
        preprocessor_kwargs: Optional[Dict] = None,
        label_mapper: Optional[CanonicalLabelMapper] = None,
    ) -> "GenomicExperimentData":
        """Build from PLINK files (canonical raw load path)."""
        if fam_path is None:
            fam_path = bed_path.replace(".bed", ".fam")
        raw_X = load_genotype_matrix(bed_path, fam_path=fam_path)
        snp_metadata = load_bim(bim_path) if bim_path is not None else None
        return cls(
            split,
            raw_X,
            snp_metadata=snp_metadata,
            preprocessor_kwargs=preprocessor_kwargs,
            label_mapper=label_mapper,
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def n_samples(self) -> int:
        return int(self._X.shape[0])

    @property
    def n_snps_raw(self) -> int:
        return int(self._X.shape[1])

    @property
    def n_classes(self) -> int:
        return self.label_mapper.n_classes

    @property
    def class_names(self):
        return self.label_mapper.classes

    def y_all(self) -> np.ndarray:
        """Dense canonical labels for every retained (eligible) sample."""
        return self._y

    # ------------------------------------------------------------------
    # Fold data
    # ------------------------------------------------------------------
    def fold_data(self, k: int) -> FoldData:
        """Fit-on-train fold data for fold ``k`` (0-based)."""
        if not 0 <= k < self.split.n_folds:
            raise ValueError(f"fold {k} out of range [0, {self.split.n_folds})")

        dev = self.split.outer_split == "development"
        train_mask = dev & (self.split.fold != k)
        val_mask = dev & (self.split.fold == k)
        train_pos = np.flatnonzero(train_mask)
        val_pos = np.flatnonzero(val_mask)

        pre = GenomicPreprocessor(**self.preprocessor_kwargs)
        pre.fit(self._X[train_pos], snp_metadata=self._snp_metadata)

        X_train = pre.transform(self._X[train_pos])
        X_val = pre.transform(self._X[val_pos])

        return FoldData(
            split=self.split,
            fold=k,
            preprocessor=pre,
            train_positions=train_pos,
            val_positions=val_pos,
            X_train=X_train,
            X_val=X_val,
            y_train=self._y[train_pos],
            y_val=self._y[val_pos],
        )

    def iter_folds(self) -> Iterator[FoldData]:
        for k in range(self.split.n_folds):
            yield self.fold_data(k)
