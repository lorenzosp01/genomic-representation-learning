"""Tests for the shared train-fitted genomic marker preprocessing layer."""

import numpy as np
import pytest

from genomic.preprocessing import GenomicPreprocessor, load_bim


def _snps(columns, n):
    """Build a (n, len(columns)) float32 matrix from equal-length columns."""
    X = np.column_stack(columns).astype(np.float32)
    assert X.shape[0] == n
    return X


# ---------------------------------------------------------------------------
# marker missingness
# ---------------------------------------------------------------------------
def test_marker_missingness_strict_less_than():
    # SNP0: 2/20 NaN -> rate 0.10 -> removed (0.10 < 0.10 is False)
    # SNP1: 1/20 NaN -> rate 0.05 -> kept
    s0 = np.array([0, 1] * 10, dtype=np.float32)
    s0[0] = np.nan
    s0[1] = np.nan
    s1 = np.array([0, 1] * 10, dtype=np.float32)
    s1[0] = np.nan
    X = _snps([s0, s1], 20)
    pre = GenomicPreprocessor(maf_threshold=0.0, ld_window=1)
    pre.fit(X)
    assert pre.marker_missingness_mask_.tolist() == [False, True]


def test_marker_missingness_learned_from_train_only():
    rng = np.random.RandomState(0)
    s0 = np.where(rng.rand(30) < 0.5, 0.0, 1.0).astype(np.float32)
    s0[:3] = np.nan  # 3/30 = 0.10 -> removed
    s1 = np.where(rng.rand(30) < 0.5, 0.0, 2.0).astype(np.float32)
    X = _snps([s0, s1], 30)
    pre = GenomicPreprocessor(maf_threshold=0.0, ld_window=1)
    pre.fit(X)
    mask_before = pre.marker_missingness_mask_.copy()
    pre.transform(np.where(rng.rand(5, 2) < 0.5, 0.0, 1.0).astype(np.float32))
    assert np.array_equal(pre.marker_missingness_mask_, mask_before)


# ---------------------------------------------------------------------------
# MAF (observed-only, before imputation, >= boundary)
# ---------------------------------------------------------------------------
def test_maf_observed_only_and_boundary():
    # SNP0: 100 observed, 99x0 + 1x2 -> p = 2/200 = 0.01 -> maf=0.01 -> kept (>=)
    # SNP1: 100 observed, all 0 -> maf=0 -> removed
    s0 = np.zeros(100, dtype=np.float32)
    s0[0] = 2.0
    s1 = np.zeros(100, dtype=np.float32)
    X = _snps([s0, s1], 100)
    pre = GenomicPreprocessor(maf_threshold=0.01, ld_window=1)
    pre.fit(X)
    assert pre.maf_mask_.tolist() == [True, False]
    assert pre.maf_values_[0] == pytest.approx(0.01)


def test_maf_ignores_missing_values():
    s = np.array([0.0] * 50 + [2.0] * 50 + [np.nan] * 10, dtype=np.float32)
    X = _snps([s], 110)
    pre = GenomicPreprocessor(maf_threshold=0.01, ld_window=1)
    pre.fit(X)
    assert pre.maf_values_[0] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# mode imputation
# ---------------------------------------------------------------------------
def test_mode_imputation_lowest_tiebreak():
    s01 = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1], dtype=np.float32)  # 6x0 6x1 -> 0
    s12 = np.array([1, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 2], dtype=np.float32)  # 6x1 6x2 -> 1
    s02 = np.array([0, 0, 2, 2, 0, 2, 0, 2, 0, 2, 0, 2], dtype=np.float32)  # 6x0 6x2 -> 0
    s2 = np.array([2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1], dtype=np.float32)  # 9x2 3x1 -> 2
    X = _snps([s01, s12, s02, s2], 12)
    pre = GenomicPreprocessor(maf_threshold=0.0, ld_window=1)
    pre.fit(X)
    assert pre.imputation_values_[0] == 0.0
    assert pre.imputation_values_[1] == 1.0
    assert pre.imputation_values_[2] == 0.0
    assert pre.imputation_values_[3] == 2.0


def test_transform_uses_train_learned_mode():
    s = np.array([1.0] * 19 + [np.nan], dtype=np.float32)  # 19x1 -> mode 1
    s2 = np.array([0.0, 2.0] * 10, dtype=np.float32)
    X = _snps([s, s2], 20)
    pre = GenomicPreprocessor(maf_threshold=0.0, ld_window=1)
    pre.fit(X)
    val = np.array([[np.nan, 0.0]], dtype=np.float32)
    out = pre.transform(val)
    assert out[0, 0] == 1.0  # NaN imputed with train-learned mode 1


def test_mode_imputation_values_are_012():
    rng = np.random.RandomState(1)
    s = np.where(rng.rand(50) < 0.5, 0.0, 2.0).astype(np.float32)
    s[0] = np.nan
    X = _snps([s, s.copy()], 50)
    pre = GenomicPreprocessor(maf_threshold=0.0, ld_window=1)
    pre.fit(X)
    vals = pre.imputation_values_[~np.isnan(pre.imputation_values_)]
    assert set(np.unique(vals).tolist()) <= {0.0, 1.0, 2.0}


# ---------------------------------------------------------------------------
# transform semantics
# ---------------------------------------------------------------------------
def test_transform_preserves_sample_count_and_is_012_no_nan():
    rng = np.random.RandomState(2)
    s = np.where(rng.rand(40) < 0.5, 0.0, 1.0).astype(np.float32)
    s[0] = np.nan
    s3 = np.where(rng.rand(40) < 0.5, 1.0, 2.0).astype(np.float32)
    X = _snps([s, s.copy(), s3], 40)
    pre = GenomicPreprocessor(maf_threshold=0.01, ld_r2_threshold=0.99)
    pre.fit(X)
    val = np.where(np.random.RandomState(3).rand(7, 3) < 0.5, 0.0, 1.0).astype(np.float32)
    out = pre.transform(val)
    assert out.shape[0] == 7
    assert not np.isnan(out).any()
    assert set(np.unique(out).tolist()) <= {0.0, 1.0, 2.0}


def test_validation_receives_same_columns_as_train():
    rng = np.random.RandomState(4)
    s0 = np.where(rng.rand(30) < 0.5, 0.0, 1.0).astype(np.float32)
    s1 = s0.copy()  # correlated with s0 -> LD removes s1
    s2 = np.where(rng.rand(30) < 0.5, 0.0, 2.0).astype(np.float32)
    X = _snps([s0, s1, s2], 30)
    pre = GenomicPreprocessor(maf_threshold=0.01)
    pre.fit(X)
    retained = pre.retained_snp_indices_.copy()
    val = np.where(np.random.RandomState(5).rand(5, 3) < 0.5, 0.0, 1.0).astype(np.float32)
    out = pre.transform(val)
    assert out.shape[1] == len(retained)


def test_validation_cannot_change_fitted_state():
    rng = np.random.RandomState(6)
    s = np.where(rng.rand(50) < 0.5, 0.0, 2.0).astype(np.float32)
    s[0] = np.nan
    X = _snps([s, s.copy()], 50)
    pre = GenomicPreprocessor(maf_threshold=0.01, ld_r2_threshold=0.99)
    pre.fit(X)
    snap = {
        "mask": pre.marker_missingness_mask_.copy(),
        "maf": pre.maf_values_.copy(),
        "imp": pre.imputation_values_.copy(),
        "ret": pre.retained_snp_indices_.copy(),
    }
    pre.transform(np.full((3, 2), 2.0, dtype=np.float32))
    assert np.array_equal(pre.marker_missingness_mask_, snap["mask"])
    assert np.array_equal(pre.maf_values_, snap["maf"], equal_nan=True)
    assert np.array_equal(pre.imputation_values_, snap["imp"], equal_nan=True)
    assert np.array_equal(pre.retained_snp_indices_, snap["ret"])


def test_deterministic_repeat_fit():
    rng = np.random.RandomState(7)
    s = np.where(rng.rand(40) < 0.5, 0.0, 1.0).astype(np.float32)
    s2 = np.where(rng.rand(40) < 0.5, 0.0, 2.0).astype(np.float32)
    X = _snps([s, s2], 40)
    a = GenomicPreprocessor(maf_threshold=0.01).fit(X)
    b = GenomicPreprocessor(maf_threshold=0.01).fit(X)
    assert np.array_equal(a.retained_snp_indices_, b.retained_snp_indices_)
    assert np.array_equal(a.imputation_values_, b.imputation_values_, equal_nan=True)


def test_different_folds_may_differ():
    rng = np.random.RandomState(8)
    s0 = np.where(rng.rand(20) < 0.5, 0.0, 1.0).astype(np.float32)
    indep1 = np.where(rng.rand(20) < 0.5, 0.0, 2.0).astype(np.float32)

    X1 = _snps([s0, s0.copy(), indep1], 20)  # SNP1 correlates with SNP0 -> removed
    X2 = _snps([s0, indep1, s0.copy()], 20)  # SNP2 correlates with SNP0 -> removed

    p1 = GenomicPreprocessor(maf_threshold=0.01).fit(X1)
    p2 = GenomicPreprocessor(maf_threshold=0.01).fit(X2)
    assert not np.array_equal(p1.retained_snp_indices_, p2.retained_snp_indices_)


# ---------------------------------------------------------------------------
# index mapping and metadata
# ---------------------------------------------------------------------------
def test_raw_index_mapping_survives():
    rng = np.random.RandomState(9)
    cols = [
        np.where(rng.rand(30) < 0.5, 0.0, float((i % 2) + 1)).astype(np.float32)
        for i in range(5)
    ]
    cols[2][0] = np.nan
    X = _snps(cols, 30)
    pre = GenomicPreprocessor(maf_threshold=0.0, ld_window=1).fit(X)
    idx = pre.retained_snp_indices_
    assert np.all(np.diff(idx) > 0)
    assert set(idx.tolist()) <= set(range(5))


def test_transformed_column_maps_to_snp_id(tmp_path):
    bim = tmp_path / "t.bim"
    with open(bim, "w") as f:
        for i in range(5):
            f.write(f"0\tsnp{i}\t0\t0\tA\tG\n")
    meta = load_bim(str(bim))

    rng = np.random.RandomState(10)
    cols = [np.where(rng.rand(30) < 0.5, 0.0, 1.0).astype(np.float32) for _ in range(5)]
    X = _snps(cols, 30)
    pre = GenomicPreprocessor(maf_threshold=0.0, ld_window=1).fit(X, snp_metadata=meta)

    ret = pre.retained_snp_indices_
    ids = pre.retained_snp_metadata_["snp_id"]
    assert ids.tolist() == [f"snp{i}" for i in ret.tolist()]
    for k in range(len(ret)):
        assert pre.retained_snp_metadata_["snp_id"][k] == f"snp{ret[k]}"


def test_metadata_row_count_mismatch_raises():
    meta = {"snp_id": np.array(["a", "b", "c"])}  # 3 rows, but X has 5 cols
    rng = np.random.RandomState(11)
    cols = [np.where(rng.rand(20) < 0.5, 0.0, 1.0).astype(np.float32) for _ in range(5)]
    X = _snps(cols, 20)
    with pytest.raises(ValueError, match="rows, expected"):
        GenomicPreprocessor(maf_threshold=0.0, ld_window=1).fit(X, snp_metadata=meta)


# ---------------------------------------------------------------------------
# error handling
# ---------------------------------------------------------------------------
def test_transform_feature_count_mismatch_raises():
    rng = np.random.RandomState(12)
    cols = [np.where(rng.rand(20) < 0.5, 0.0, 1.0).astype(np.float32) for _ in range(3)]
    X = _snps(cols, 20)
    pre = GenomicPreprocessor(maf_threshold=0.0, ld_window=1).fit(X)
    with pytest.raises(ValueError, match="features"):
        pre.transform(np.zeros((2, 4), dtype=np.float32))


def test_transform_before_fit_raises():
    pre = GenomicPreprocessor()
    with pytest.raises(RuntimeError, match="before fit"):
        pre.transform(np.zeros((2, 3), dtype=np.float32))


def test_invalid_genotype_values_raise():
    rng = np.random.RandomState(13)
    s = np.where(rng.rand(20) < 0.5, 0.0, 1.0).astype(np.float32)
    X = _snps([s, s.copy()], 20)
    X[0, 0] = 5.0  # invalid
    with pytest.raises(ValueError, match="outside"):
        GenomicPreprocessor(maf_threshold=0.0, ld_window=1).fit(X)


def test_no_snp_survives_marker_missingness_raises():
    X = np.full((10, 3), np.nan, dtype=np.float32)
    with pytest.raises(ValueError, match="marker-missingness"):
        GenomicPreprocessor().fit(X)


def test_no_snp_survives_maf_raises():
    X = np.zeros((10, 2), dtype=np.float32)  # all maf 0
    with pytest.raises(ValueError, match="MAF"):
        GenomicPreprocessor(maf_threshold=0.01).fit(X)


def test_invalid_thresholds_raise():
    with pytest.raises(ValueError):
        GenomicPreprocessor(marker_missingness_threshold=0.0)
    with pytest.raises(ValueError):
        GenomicPreprocessor(maf_threshold=0.6)
    with pytest.raises(ValueError):
        GenomicPreprocessor(ld_window=0)
