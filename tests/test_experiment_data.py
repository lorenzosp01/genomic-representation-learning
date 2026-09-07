"""Tests for the shared experiment-data assembly layer.

Synthetic tests use a ``SplitIndex`` built from a small synthetic ``.fam`` and a
hand-crafted raw genotype matrix (no PLINK binary files required). A separate
real-data smoke test exercises the end-to-end PLINK load path when the data is
present.
"""

import os

import numpy as np
import pytest

from genomic.labels import CanonicalLabelMapper
from genomic.experiment_data import GenomicExperimentData, load_genotype_matrix
from genomic.preprocessing import GenomicPreprocessor
from genomic.splitting import build_split


# ---------------------------------------------------------------------------
# synthetic fixtures / helpers
# ---------------------------------------------------------------------------
def write_fam(path, breeds):
    lines = []
    for breed, iids in breeds.items():
        for iid in iids:
            lines.append(f"{breed}\t{iid}\t0\t0\t0\t-9\n")
    with open(path, "w") as f:
        f.writelines(lines)


def make_breeds(*specs):
    out = {}
    for b, n in specs:
        out[b] = [f"{b}_{i}" for i in range(n)]
    return out


def build_synthetic_split(tmp_path, k_folds=3):
    fam = tmp_path / "s.fam"
    write_fam(fam, make_breeds(("ALP", 6), ("BOR", 6), ("CEN", 6)))
    return build_split(
        str(fam),
        dataset_id="synth",
        outer_split_seed=42,
        fold_seed=42,
        k_folds=k_folds,
        cohort_min_breed_size=None,
    )


def synthetic_matrix(n_samples, n_snps, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randint(0, 3, size=(n_samples, n_snps)).astype(np.float32)
    # A few NaNs in early columns (rates stay well under 0.10).
    X[0, 0] = np.nan
    X[1, 1] = np.nan
    X[2, 2] = np.nan
    return X


# Keep nothing filtered so SNP-identity expectations are trivial in the
# plumbing tests (LD disabled, MAF floor 0).
NO_FILTER = {"ld_window": 1, "maf_threshold": 0.0}


# ---------------------------------------------------------------------------
# dosage representation
# ---------------------------------------------------------------------------
def test_fold_data_dosage_is_012_and_no_nan(tmp_path):
    split = build_synthetic_split(tmp_path)
    X = synthetic_matrix(split.meta["n_eligible"], 8)
    data = GenomicExperimentData(split, X, preprocessor_kwargs=NO_FILTER)

    for fd in data.iter_folds():
        for mat in (fd.X_train, fd.X_val):
            assert not np.isnan(mat).any()
            assert set(np.unique(mat).tolist()) <= {0.0, 1.0, 2.0}
        assert fd.X_train.shape[1] == fd.X_val.shape[1] == fd.n_features
        assert fd.n_features == len(fd.retained_snp_indices)


# ---------------------------------------------------------------------------
# validation must not influence preprocessing
# ---------------------------------------------------------------------------
def test_validation_does_not_influence_preprocessor(tmp_path):
    split = build_synthetic_split(tmp_path)
    X = synthetic_matrix(split.meta["n_eligible"], 8)
    data = GenomicExperimentData(split, X, preprocessor_kwargs=NO_FILTER)

    for fd in data.iter_folds():
        ref = GenomicPreprocessor(**NO_FILTER)
        ref.fit(X[fd.train_source_index])
        assert np.array_equal(fd.retained_snp_indices, ref.retained_snp_indices_)
        # transforming val must not change the fitted state
        before = fd.retained_snp_indices.copy()
        _ = fd.X_val
        assert np.array_equal(fd.retained_snp_indices, before)


def test_folds_have_independent_preprocessors(tmp_path):
    split = build_synthetic_split(tmp_path)
    X = synthetic_matrix(split.meta["n_eligible"], 8)
    data = GenomicExperimentData(split, X, preprocessor_kwargs=NO_FILTER)
    fds = list(data.iter_folds())
    assert len(fds) == split.n_folds
    preps = [fd.preprocessor for fd in fds]
    assert len({id(p) for p in preps}) == split.n_folds


# ---------------------------------------------------------------------------
# source identity chain
# ---------------------------------------------------------------------------
def test_source_identity_chain(tmp_path):
    split = build_synthetic_split(tmp_path)
    X = synthetic_matrix(split.meta["n_eligible"], 8)
    data = GenomicExperimentData(split, X, preprocessor_kwargs=NO_FILTER)

    for fd in data.iter_folds():
        assert np.array_equal(fd.train_source_index, split.fold_train_indices(fd.fold))
        assert np.array_equal(fd.val_source_index, split.fold_val_indices(fd.fold))

        # source_index -> FID/IID -> sample_id -> breed
        for pos in fd.train_positions:
            si = split.source_index[pos]
            assert split.sample_id[pos] == f"{split.fid[pos]}:{split.iid[pos]}"
            assert split.fid[pos] == split.breed[pos]
            assert si >= 0


def test_fold_excludes_test_and_covers_development(tmp_path):
    split = build_synthetic_split(tmp_path)
    X = synthetic_matrix(split.meta["n_eligible"], 8)
    data = GenomicExperimentData(split, X, preprocessor_kwargs=NO_FILTER)

    dev_pos = set(np.flatnonzero(split.outer_split == "development").tolist())
    test_pos = set(np.flatnonzero(split.outer_split == "test").tolist())

    for fd in data.iter_folds():
        covered = set(fd.train_positions.tolist()) | set(fd.val_positions.tolist())
        assert covered == dev_pos
        assert covered.isdisjoint(test_pos)
        assert set(fd.train_positions.tolist()).isdisjoint(set(fd.val_positions.tolist()))


# ---------------------------------------------------------------------------
# SNP identity via retained metadata
# ---------------------------------------------------------------------------
def test_snp_metadata_identity(tmp_path):
    split = build_synthetic_split(tmp_path)
    n_snp = 6
    X = synthetic_matrix(split.meta["n_eligible"], n_snp)
    meta = {
        "snp_id": np.array([f"snp{i}" for i in range(n_snp)], dtype=object),
        "chromosome": np.array(["1"] * n_snp, dtype=object),
    }
    data = GenomicExperimentData(
        split, X, snp_metadata=meta, preprocessor_kwargs=NO_FILTER
    )
    for fd in data.iter_folds():
        ids = fd.retained_snp_metadata["snp_id"]
        assert ids.tolist() == [f"snp{i}" for i in fd.retained_snp_indices.tolist()]


# ---------------------------------------------------------------------------
# canonical labels
# ---------------------------------------------------------------------------
def test_labels_are_canonical_and_dense(tmp_path):
    split = build_synthetic_split(tmp_path)
    X = synthetic_matrix(split.meta["n_eligible"], 8)
    data = GenomicExperimentData(split, X, preprocessor_kwargs=NO_FILTER)

    assert data.class_names == ["ALP", "BOR", "CEN"]  # sorted alphabetical
    assert data.n_classes == 3
    y = data.y_all()
    assert set(np.unique(y).tolist()) == {0, 1, 2}

    mapper = CanonicalLabelMapper.from_breeds(split.breed)
    assert np.array_equal(y, mapper.encode(split.breed))

    for fd in data.iter_folds():
        assert np.array_equal(fd.y_train, y[fd.train_positions])
        assert np.array_equal(fd.y_val, y[fd.val_positions])


# ---------------------------------------------------------------------------
# constructor validation
# ---------------------------------------------------------------------------
def test_source_index_out_of_range_raises(tmp_path):
    split = build_synthetic_split(tmp_path)
    X = synthetic_matrix(split.meta["n_eligible"] - 1, 8)
    with pytest.raises(ValueError, match="outside"):
        GenomicExperimentData(split, X, preprocessor_kwargs=NO_FILTER)


def test_fold_out_of_range_raises(tmp_path):
    split = build_synthetic_split(tmp_path)
    X = synthetic_matrix(split.meta["n_eligible"], 8)
    data = GenomicExperimentData(split, X, preprocessor_kwargs=NO_FILTER)
    with pytest.raises(ValueError, match="out of range"):
        data.fold_data(split.n_folds)


# ---------------------------------------------------------------------------
# real-data smoke test (end-to-end PLINK load path)
# ---------------------------------------------------------------------------
FAM = "data/ADAPTmap_genotypeTOP_20160222_full.fam"
BED = "data/ADAPTmap_genotypeTOP_20160222_full.bed"
DS = "ADAPTmap_genotypeTOP_20160222_full"


@pytest.mark.skipif(not (os.path.exists(FAM) and os.path.exists(BED)), reason="data missing")
def test_load_genotype_matrix_real_orientation():
    X = load_genotype_matrix(BED, fam_path=FAM)
    assert X.shape[0] == 4653
    assert X.dtype == np.float32
    assert np.isnan(X).any()  # raw matrix has missing genotypes


@pytest.mark.skipif(not (os.path.exists(FAM) and os.path.exists(BED)), reason="data missing")
def test_real_from_plink_smoke():
    from genomic.splitting import load_split

    split = load_split(DS, 42, 42, FAM, out_dir="splits", cohort_label="indmiss0p10")
    data = GenomicExperimentData.from_plink(
        split, BED, preprocessor_kwargs={"ld_window": 1}
    )
    assert data.n_samples == split.meta["n_eligible"] == 2863
    assert data.n_classes == 34
    assert data.n_snps_raw == 53347

    fd = data.fold_data(0)
    assert fd.X_train.shape[1] == fd.X_val.shape[1] == fd.n_features
    assert set(np.unique(fd.y_train).tolist()).issubset(set(range(34)))
    assert not np.isnan(fd.X_train).any()
    assert set(np.unique(fd.X_train).tolist()) <= {0.0, 1.0, 2.0}
