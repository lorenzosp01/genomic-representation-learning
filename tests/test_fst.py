"""Tests for the locus-wise Weir & Cockerham (1984) F_ST estimator."""

import numpy as np
import pytest

from genomic.fst import weir_cockerham_fst


def test_discriminative_marker_is_one():
    # Class 0 fixed for dosage 0, class 1 fixed for dosage 2 -> complete
    # differentiation, expected theta == 1.
    X = np.array([[0, 0], [0, 0], [2, 2], [2, 2]], dtype=np.float32)
    y = np.array([0, 0, 1, 1])
    fst = weir_cockerham_fst(X, y, 2)
    assert fst.shape == (2,)
    assert np.allclose(fst, 1.0)


def test_identical_marker_is_zero():
    # Same genotype distribution in every class -> no differentiation.
    X = np.array([[0], [1], [2], [0], [1], [2]], dtype=np.float32)
    y = np.array([0, 0, 0, 1, 1, 1])
    fst = weir_cockerham_fst(X, y, 2)
    assert fst[0] == pytest.approx(0.0)
    assert 0.0 <= fst[0] <= 1.0


def test_monomorphic_marker_is_zero():
    y = np.array([0, 0, 0, 1, 1, 1])

    X_zero = np.zeros((6, 2), dtype=np.float32)
    assert np.allclose(weir_cockerham_fst(X_zero, y, 2), 0.0)

    X_two = np.full((6, 1), 2.0, dtype=np.float32)
    assert weir_cockerham_fst(X_two, y, 2)[0] == pytest.approx(0.0)


def test_unequal_sample_sizes_numerically_stable():
    rng = np.random.RandomState(0)
    # Deliberately unbalanced classes (2 vs 20 individuals).
    X = np.concatenate([
        rng.randint(0, 2, size=(2, 50)),
        rng.randint(0, 3, size=(20, 50)),
    ], axis=0).astype(np.float32)
    y = np.array([0] * 2 + [1] * 20)

    fst = weir_cockerham_fst(X, y, 2)
    assert fst.shape == (50,)
    assert np.isfinite(fst).all()
    assert ((fst >= 0.0) & (fst <= 1.0)).all()


def test_output_shape_dtype_and_range():
    rng = np.random.RandomState(1)
    X = rng.randint(0, 3, size=(40, 17)).astype(np.float32)
    y = rng.randint(0, 4, size=40)

    fst = weir_cockerham_fst(X, y, 4)
    assert fst.shape == (17,)
    assert fst.dtype.kind == "f"
    assert ((fst >= 0.0) & (fst <= 1.0)).all()


def test_empty_class_is_ignored():
    # n_classes=3 but label 2 is unpopulated: must match the 2-class result.
    X = np.array([[0, 0], [0, 0], [2, 2], [2, 2]], dtype=np.float32)
    y = np.array([0, 0, 1, 1])
    assert np.allclose(weir_cockerham_fst(X, y, 2), weir_cockerham_fst(X, y, 3))


def test_single_population_returns_zero():
    X = np.array([[0], [1], [2]], dtype=np.float32)
    y = np.array([0, 0, 0])
    assert np.allclose(weir_cockerham_fst(X, y, 1), 0.0)


def test_matches_scalar_reference_implementation():
    # Cross-check the marker-vectorized code against a direct per-marker loop.
    rng = np.random.RandomState(2)
    X = rng.randint(0, 3, size=(30, 9)).astype(np.float64)
    y = rng.randint(0, 3, size=30)

    got = weir_cockerham_fst(X, y, 3)
    ref = np.array([_scalar_fst(X[:, j], y, 3) for j in range(X.shape[1])])
    assert np.allclose(got, ref)


def test_rejects_non_dosage_values():
    X = np.array([[0.0], [1.3]], dtype=np.float32)
    with pytest.raises(ValueError, match="outside"):
        weir_cockerham_fst(X, np.array([0, 1]), 2)

    X_nan = np.array([[0.0], [np.nan]], dtype=np.float32)
    with pytest.raises(ValueError, match="non-finite"):
        weir_cockerham_fst(X_nan, np.array([0, 1]), 2)


def test_rejects_shape_mismatch():
    X = np.zeros((4, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        weir_cockerham_fst(X, np.array([0, 1]), 2)


def _scalar_fst(x: np.ndarray, y: np.ndarray, n_classes: int) -> float:
    """Direct per-marker reference implementing the same W&C (1984) formulas."""
    counts = np.array([(y == c).sum() for c in range(n_classes)], dtype=np.float64)
    populated = np.flatnonzero(counts > 0)
    r = populated.size
    if r < 2:
        return 0.0
    n_i = counts[populated]
    n_total = float(n_i.sum())
    n_bar = n_total / r
    if n_bar <= 1.0:
        return 0.0
    n_c = (n_total - float((n_i ** 2).sum()) / n_total) / (r - 1)

    p_i = np.array([x[y == c].mean() / 2.0 for c in populated])
    h_i = np.array([(x[y == c] == 1).mean() for c in populated])
    p_bar = float((n_i * p_i).sum() / n_total)
    h_bar = float((n_i * h_i).sum() / n_total)
    s2 = float((n_i * (p_i - p_bar) ** 2).sum() / ((r - 1) * n_bar))

    p_var = p_bar * (1.0 - p_bar)
    ratio = (r - 1) / r
    a = (n_bar / n_c) * (
        s2 - (p_var - ratio * s2 - 0.25 * h_bar) / (n_bar - 1.0)
    )
    b = (n_bar / (n_bar - 1.0)) * (
        p_var - ratio * s2 - ((2.0 * n_bar - 1.0) / (4.0 * n_bar)) * h_bar
    )
    c = 0.5 * h_bar

    denom = a + b + c
    if denom <= 1e-8:
        return 0.0
    return float(np.clip(a / denom, 0.0, 1.0))
