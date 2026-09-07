"""Train-fitted genomic marker preprocessing (shared across models).

This layer implements the frozen marker-preprocessing protocol and is
intentionally leakage-safe: every fitted quantity (marker-missingness mask,
MAF, mode-imputation values, LD panel) is learned from ``fit(X_train)`` only.
Validation/test data may only be passed to ``transform``.

The output is always an exact categorical dosage matrix with values in
{0, 1, 2} and no NaNs. Model-specific input representations (VMGP dosage,
contrastive 4-channel one-hot, classical dosage) are built *downstream* of this
layer, never inside it.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def load_bim(bim_path: str) -> Dict[str, np.ndarray]:
    """Load PLINK ``.bim`` metadata preserving raw SNP column order.

    Returns a dict with keys ``raw_snp_index``, ``chromosome``, ``snp_id``,
    ``genetic_distance``, ``position``, ``allele_1``, ``allele_2``. Values are
    preserved verbatim (as strings); no biological interpretation is applied
    (in particular, chromosome values are not dropped even though the
    genetic-distance and base-pair-position fields are zero in this dataset).
    """
    bim = np.loadtxt(bim_path, dtype=str, ndmin=2)
    if bim.shape[0] == 0:
        raise ValueError(f"Empty .bim file: {bim_path}")
    if bim.shape[1] < 6:
        raise ValueError(f".bim file has fewer than 6 columns: {bim_path}")
    n = bim.shape[0]
    return {
        "raw_snp_index": np.arange(n, dtype=np.int64),
        "chromosome": bim[:, 0],
        "snp_id": bim[:, 1],
        "genetic_distance": bim[:, 2],
        "position": bim[:, 3],
        "allele_1": bim[:, 4],
        "allele_2": bim[:, 5],
    }


def _check_genotype_values(X: np.ndarray) -> None:
    valid = (X == 0) | (X == 1) | (X == 2) | np.isnan(X)
    if not np.all(valid):
        n_bad = int(np.count_nonzero(~valid))
        raise ValueError(
            f"Input contains {n_bad} genotype value(s) outside {{0, 1, 2, NaN}}."
        )


def _prune_ld_mask(X: np.ndarray, window: int, r2_thresh: float) -> np.ndarray:
    """Greedy LD pruning over marker order (returns a local boolean keep-mask).

    Preserves the current mathematical semantics: per-SNP z-score
    standardization, then for each SNP ``i`` remove SNPs ``j`` in
    ``[i+1, i+window)`` whose squared correlation ``r^2 > r2_thresh``.
    """
    n, p = X.shape
    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    to_remove = np.zeros(p, dtype=bool)
    for i in range(p):
        if to_remove[i]:
            continue
        end = min(i + window, p)
        if end <= i + 1:
            continue
        r2 = (X_std[:, i] @ X_std[:, i + 1:end] / n) ** 2
        high = np.nonzero(r2 > r2_thresh)[0] + i + 1
        to_remove[high] = True
    return ~to_remove


def _select_metadata(
    snp_metadata: Dict[str, np.ndarray],
    retained_raw: np.ndarray,
    n_features: int,
) -> Dict[str, np.ndarray]:
    if not isinstance(snp_metadata, dict):
        raise ValueError("snp_metadata must be a dict of equal-length arrays.")
    out: Dict[str, np.ndarray] = {}
    for key, arr in snp_metadata.items():
        arr = np.asarray(arr)
        if arr.shape[0] != n_features:
            raise ValueError(
                f"snp_metadata['{key}'] has {arr.shape[0]} rows, expected {n_features}."
            )
        out[key] = arr[retained_raw]
    return out


class GenomicPreprocessor:
    """Fit-once, transform-many genomic marker preprocessing.

    Fitting order (train only):
        1. marker missingness filtering (``missing_rate < marker_missingness_threshold``)
        2. MAF filtering on observed genotypes (``maf >= maf_threshold``)
        3. mode imputation (lowest-genotype tie-break)
        4. LD pruning (``r^2 > ld_r2_threshold``, window ``ld_window``)

    All masks and value arrays are indexed in RAW SNP feature space (length
    ``n_features_in_``); ``retained_snp_indices_`` holds the raw column indices
    of the final panel, so ``transform`` output column ``k`` corresponds to raw
    column ``retained_snp_indices_[k]``.
    """

    def __init__(
        self,
        marker_missingness_threshold: float = 0.10,
        maf_threshold: float = 0.01,
        ld_window: int = 50,
        ld_r2_threshold: float = 0.2,
    ):
        if not 0.0 < marker_missingness_threshold < 1.0:
            raise ValueError("marker_missingness_threshold must be in (0, 1)")
        if not 0.0 <= maf_threshold < 0.5:
            raise ValueError("maf_threshold must be in [0, 0.5)")
        if ld_window < 1:
            raise ValueError("ld_window must be >= 1")
        if not 0.0 <= ld_r2_threshold < 1.0:
            raise ValueError("ld_r2_threshold must be in [0, 1)")
        self.marker_missingness_threshold = float(marker_missingness_threshold)
        self.maf_threshold = float(maf_threshold)
        self.ld_window = int(ld_window)
        self.ld_r2_threshold = float(ld_r2_threshold)

    # ------------------------------------------------------------------
    def fit(self, X_train, snp_metadata: Optional[Dict[str, np.ndarray]] = None):
        """Learn all preprocessing state from ``X_train`` only.

        ``snp_metadata`` (optional) must be a dict of arrays each of length
        ``X_train.shape[1]`` (raw SNP count), aligned to raw PLINK column order.
        """
        X = np.asarray(X_train, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("X_train must be a 2D array (samples x SNPs).")
        _check_genotype_values(X)
        n_samples, p = X.shape
        if p == 0:
            raise ValueError("X_train has zero features.")
        self.n_features_in_ = p
        self.original_snp_indices_ = np.arange(p, dtype=np.int64)

        # --- 1. marker missingness ------------------------------------
        rates = np.isnan(X).mean(axis=0)
        miss_mask = rates < self.marker_missingness_threshold
        self.marker_missingness_rates_ = rates.astype(np.float64)
        self.marker_missingness_mask_ = miss_mask
        if not miss_mask.any():
            raise ValueError("No SNP survives marker-missingness filtering.")

        miss_idx = np.flatnonzero(miss_mask)
        X_miss = X[:, miss_mask]

        # --- 2. MAF on observed genotypes (before imputation) ---------
        obs_count = np.sum(~np.isnan(X_miss), axis=0).astype(np.float64)
        obs_sum = np.nansum(X_miss, axis=0).astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            freq = np.where(obs_count > 0, obs_sum / (2.0 * obs_count), np.nan)
        maf = np.minimum(freq, 1.0 - freq)
        maf_keep = maf >= self.maf_threshold  # stage-local (over missingness survivors)

        self.maf_values_ = np.full(p, np.nan, dtype=np.float64)
        self.maf_values_[miss_idx] = maf
        self.maf_mask_ = np.zeros(p, dtype=bool)
        self.maf_mask_[miss_idx[maf_keep]] = True
        if not self.maf_mask_.any():
            raise ValueError("No SNP survives MAF filtering.")

        maf_idx = miss_idx[maf_keep]  # raw indices surviving missingness + MAF
        X_maf = X_miss[:, maf_keep]

        # --- 3. mode imputation (lowest-genotype tie-break) -----------
        c0 = np.sum(X_maf == 0, axis=0)
        c1 = np.sum(X_maf == 1, axis=0)
        c2 = np.sum(X_maf == 2, axis=0)
        if np.any((c0 + c1 + c2) == 0):
            raise ValueError(
                "A retained SNP has no observed training genotype; "
                "cannot derive an imputation mode."
            )
        mode = np.argmax(np.stack([c0, c1, c2], axis=0), axis=0).astype(np.float32)
        self.imputation_values_ = np.full(p, np.nan, dtype=np.float32)
        self.imputation_values_[maf_idx] = mode

        X_imp = np.where(np.isnan(X_maf), mode, X_maf)

        # --- 4. LD pruning --------------------------------------------
        ld_keep_local = _prune_ld_mask(X_imp, window=self.ld_window, r2_thresh=self.ld_r2_threshold)
        if not ld_keep_local.any():
            raise ValueError("No SNP survives LD pruning.")

        retained_raw = maf_idx[ld_keep_local]
        self.ld_mask_ = np.zeros(p, dtype=bool)
        self.ld_mask_[retained_raw] = True
        self.retained_snp_indices_ = retained_raw

        self.retained_snp_metadata_ = None
        if snp_metadata is not None:
            self.retained_snp_metadata_ = _select_metadata(snp_metadata, retained_raw, p)

        return self

    # ------------------------------------------------------------------
    def transform(self, X):
        """Apply the fitted panel and train-learned imputation to ``X``."""
        self._check_fitted()
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array (samples x SNPs).")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}."
            )
        _check_genotype_values(X)
        X_sel = X[:, self.retained_snp_indices_]
        impute_vals = self.imputation_values_[self.retained_snp_indices_]
        return np.where(np.isnan(X_sel), impute_vals, X_sel)

    # ------------------------------------------------------------------
    def fit_transform(self, X_train, snp_metadata: Optional[Dict[str, np.ndarray]] = None):
        self.fit(X_train, snp_metadata=snp_metadata)
        return self.transform(X_train)

    def _check_fitted(self):
        if not hasattr(self, "retained_snp_indices_"):
            raise RuntimeError("GenomicPreprocessor.transform called before fit.")
