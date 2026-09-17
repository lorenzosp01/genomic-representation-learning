"""Locus-wise F_ST (Weir & Cockerham 1984) for protocol-v2 breed panels.

Computes the multi-population theta estimator for every biallelic diploid
marker from allele dosages in {0, 1, 2}, for unequal subpopulation sample sizes:

    theta = a / (a + b + c)

with the variance components ``a``, ``b``, ``c`` of Weir & Cockerham (1984).
The implementation is fully vectorized across markers.

Protocol-v2 note: this is a pure locus-wise statistic. It performs no data
selection, no preprocessing and no locked-test access; the caller is
responsible for fitting/ranking with training-only information.
"""

from __future__ import annotations

import numpy as np

#: denominator below which theta is reported as 0 (monomorphic markers etc.)
_DENOM_EPS = 1e-8


def weir_cockerham_fst(X: np.ndarray, y: np.ndarray, n_classes: int) -> np.ndarray:
    """Weir & Cockerham (1984) multi-population F_ST for every marker.

    Parameters
    ----------
    X:
        ``(N, P)`` allele dosages in ``{0, 1, 2}`` (no missing values).
    y:
        ``(N,)`` integer subpopulation labels in ``[0, n_classes)``.
    n_classes:
        Total number of classes; classes without samples are ignored.

    Returns
    -------
    ``(P,)`` float64 array with ``theta`` in ``[0, 1]``. Monomorphic markers
    and markers whose variance-component denominator is ``<= 1e-8`` yield 0.

    Notes
    -----
    With ``n_i`` the sample size of subpopulation ``i``, ``r`` the number of
    populated classes and ``n_total = sum(n_i)``:

        n_bar = n_total / r
        n_c   = (n_total - sum(n_i**2) / n_total) / (r - 1)
        p_i   = mean(X_i) / 2          (sample allele frequency)
        h_i   = mean(X_i == 1)         (observed heterozygosity)
        p_bar = sum(n_i * p_i) / n_total
        s2    = sum(n_i * (p_i - p_bar)**2) / ((r - 1) * n_bar)
        h_bar = sum(n_i * h_i) / n_total

        a = (n_bar / n_c) * (s2 - (1 / (n_bar - 1)) *
                             (p_bar * (1 - p_bar)
                              - ((r - 1) / r) * s2 - 0.25 * h_bar))
        b = (n_bar / (n_bar - 1)) * (p_bar * (1 - p_bar)
                                     - ((r - 1) / r) * s2
                                     - ((2 * n_bar - 1) / (4 * n_bar)) * h_bar)
        c = 0.5 * h_bar
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array (samples x markers).")
    n_samples, n_markers = X.shape
    if y.shape != (n_samples,):
        raise ValueError(
            f"y must have shape ({n_samples},) matching X rows, got {y.shape}."
        )
    if int(n_classes) < 1:
        raise ValueError("n_classes must be >= 1.")

    if not np.all(np.isfinite(X)):
        raise ValueError(
            "X contains non-finite dosages; expected exact {0, 1, 2} values."
        )
    if not np.all((X == 0.0) | (X == 1.0) | (X == 2.0)):
        raise ValueError(
            "X contains dosage values outside {0, 1, 2}; continuous "
            "imputation is not a valid input for F_ST."
        )

    n_classes = int(n_classes)
    y_int = np.asarray(y, dtype=np.int64)
    if y_int.size and (y_int.min() < 0 or y_int.max() >= n_classes):
        raise ValueError("y contains labels outside [0, n_classes).")

    fst = np.zeros(n_markers, dtype=np.float64)
    if n_markers == 0 or n_samples == 0:
        return fst

    counts = np.bincount(y_int, minlength=n_classes).astype(np.float64)
    populated = np.flatnonzero(counts > 0)
    r = int(populated.size)
    if r < 2:
        return fst  # differentiation is undefined for a single population

    n_i = counts[populated]
    n_total = float(n_i.sum())
    n_bar = n_total / r
    if n_bar <= 1.0:
        return fst  # n_bar - 1 divides the formula; undefined at n_bar == 1
    n_c = (n_total - float((n_i ** 2).sum()) / n_total) / (r - 1)
    if n_c <= 0.0:
        return fst

    # Compact one-hot over populated classes -> marker-vectorized sums.
    label_pos = -np.ones(n_classes, dtype=np.int64)
    label_pos[populated] = np.arange(r)
    onehot = np.zeros((n_samples, r), dtype=np.float64)
    onehot[np.arange(n_samples), label_pos[y_int]] = 1.0

    dosage_sum = onehot.T @ X                                  # (r, P)
    het_count = onehot.T @ (X == 1.0).astype(np.float64)       # (r, P)

    p_i = dosage_sum / (2.0 * n_i[:, None])
    h_i = het_count / n_i[:, None]

    p_bar = (n_i[:, None] * p_i).sum(axis=0) / n_total          # (P,)
    h_bar = (n_i[:, None] * h_i).sum(axis=0) / n_total          # (P,)
    s2 = (n_i[:, None] * (p_i - p_bar[None, :]) ** 2).sum(axis=0) / ((r - 1) * n_bar)

    ratio = (r - 1) / r
    p_var = p_bar * (1.0 - p_bar)
    within = p_var - ratio * s2

    a = (n_bar / n_c) * (s2 - (within - 0.25 * h_bar) / (n_bar - 1.0))
    b = (n_bar / (n_bar - 1.0)) * (
        p_var - ratio * s2 - ((2.0 * n_bar - 1.0) / (4.0 * n_bar)) * h_bar
    )
    c = 0.5 * h_bar

    denom = a + b + c
    ok = denom > _DENOM_EPS
    fst[ok] = a[ok] / denom[ok]
    return np.clip(fst, 0.0, 1.0)
