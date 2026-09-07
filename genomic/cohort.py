"""Pre-split sample-level cohort eligibility.

Individual (sample-level) missingness is a deterministic per-sample quality
rule and is conceptually distinct from the train-fitted marker preprocessing
that will live in ``genomic/preprocessing.py``.
"""

from __future__ import annotations

import numpy as np
from bed_reader import open_bed


def compute_individual_missingness(bed_path: str, fam_path: str) -> np.ndarray:
    """Per-sample missing-genotype rate, in ``.fam`` row order.

    Missing genotypes are read as ``NaN`` (bed_reader default). The rate for
    individual ``i`` is ``nan_count_i / total_snp_count`` and depends only on
    that individual's own genotypes, never on other samples.
    """
    G = open_bed(bed_path, count_A1=True)
    X = G.read().astype(np.float32)
    fam = np.loadtxt(fam_path, dtype=str, ndmin=2)
    if X.shape[1] == len(fam):
        X = X.T
    n_snps = X.shape[1]
    return np.isnan(X).sum(axis=1) / n_snps


def individual_qc_mask(missingness: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean keep-mask: keep a sample iff ``missingness < threshold``.

    This encodes the frozen rule ``individual_missingness_rule == "<"``.
    """
    return missingness < threshold
