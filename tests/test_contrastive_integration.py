"""Integration tests for the contrastive pipeline over the shared protocol-v2
experiment-data layer.

Synthetic tests use a ``GenomicExperimentData`` built from a small synthetic
``.fam`` + raw genotype matrix (no PLINK binary files). Real-data tests verify
the frozen protocol-v2 fold sizes / 34-class mapping when the data is present.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from genomic.experiment_data import GenomicExperimentData
from genomic.labels import CanonicalLabelMapper
from genomic.splitting import build_split, load_split

from contrastive_learning import data as contrastive_data
from contrastive_learning.augmentation import GeneticAugmentation

REPO = Path(__file__).resolve().parents[1]


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


def synthetic_experiment_data(tmp_path, k_folds=3, n_snps=8, seed=0):
    fam = tmp_path / "s.fam"
    write_fam(fam, make_breeds(("ALP", 6), ("BOR", 6), ("CEN", 6)))
    split = build_split(
        str(fam), dataset_id="synth", outer_split_seed=42, fold_seed=42,
        k_folds=k_folds, cohort_min_breed_size=None,
    )
    rng = np.random.RandomState(seed)
    X = rng.randint(0, 3, size=(split.meta["n_eligible"], n_snps)).astype(np.float32)
    X[0, 0] = np.nan
    X[1, 1] = np.nan
    X[2, 2] = np.nan
    return GenomicExperimentData(
        split, X, preprocessor_kwargs={"ld_window": 1, "maf_threshold": 0.0}
    )


# ---------------------------------------------------------------------------
# dosage -> tensor -> one-hot representation
# ---------------------------------------------------------------------------
def test_fold_tensors_preserve_dosage_and_no_nan(tmp_path):
    ed = synthetic_experiment_data(tmp_path)
    for k in range(ed.split.n_folds):
        fd = ed.fold_data(k)
        Xtr, ytr, Xval, yval = contrastive_data.fold_tensors(fd)
        for t in (Xtr, Xval):
            assert t.dtype == torch.float32
            assert not torch.isnan(t).any()
            vals = torch.unique(t).tolist()
            assert set(vals) <= {0.0, 1.0, 2.0}
        assert ytr.dtype == torch.long
        assert yval.dtype == torch.long


def test_contrastive_tensors_align_with_folddata_source_indices(tmp_path):
    ed = synthetic_experiment_data(tmp_path)
    for k in range(ed.split.n_folds):
        fd = ed.fold_data(k)
        Xtr, ytr, Xval, yval = contrastive_data.fold_tensors(fd)
        assert Xtr.shape[0] == len(fd.train_source_index)
        assert Xval.shape[0] == len(fd.val_source_index)
        assert torch.equal(Xtr, torch.FloatTensor(fd.X_train))
        assert torch.equal(Xval, torch.FloatTensor(fd.X_val))
        assert torch.equal(ytr, torch.LongTensor(fd.y_train))
        assert torch.equal(yval, torch.LongTensor(fd.y_val))


def test_one_hot_exactly_one_active_channel_no_mask():
    X = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.float32)
    oh = GeneticAugmentation.one_hot_encode(torch.FloatTensor(X))
    assert oh.shape == (2, 3, 4)
    # exactly one active channel per SNP (no fractional all-zero rows)
    assert torch.all(oh.sum(dim=-1) == 1.0)
    # the 4th channel (augmentation mask) is unused before masking
    assert torch.all(oh[:, :, 3] == 0)


def test_validation_membership_unchanged_by_dataloader(tmp_path):
    ed = synthetic_experiment_data(tmp_path)
    fd = ed.fold_data(0)
    _, _, Xval, yval = contrastive_data.fold_tensors(fd)
    assert Xval.shape[0] == len(fd.val_source_index)
    assert torch.equal(Xval, torch.FloatTensor(fd.X_val))


def test_preprocessing_state_unchanged_by_dataloader(tmp_path):
    ed = synthetic_experiment_data(tmp_path)
    fd = ed.fold_data(0)
    before = fd.retained_snp_indices.copy()
    tr_dl, val_dl = contrastive_data.build_fold_dataloaders(
        fd, 16, num_workers=0, pin_memory=False
    )
    _ = list(tr_dl), list(val_dl)
    assert np.array_equal(fd.retained_snp_indices, before)


def test_repeated_fold_execution_deterministic(tmp_path):
    ed = synthetic_experiment_data(tmp_path)
    f1 = ed.fold_data(0)
    f2 = ed.fold_data(0)
    assert np.array_equal(f1.retained_snp_indices, f2.retained_snp_indices)
    assert np.array_equal(f1.X_train, f2.X_train)
    assert np.array_equal(f1.X_val, f2.X_val)
    assert np.array_equal(f1.y_train, f2.y_train)
    assert np.array_equal(f1.y_val, f2.y_val)


def test_model_receives_fold_specific_n_markers(tmp_path):
    ed = synthetic_experiment_data(tmp_path)
    seen = []
    for k in range(ed.split.n_folds):
        fd = ed.fold_data(k)
        seen.append(fd.X_train.shape[1])
        assert fd.X_train.shape[1] == len(fd.retained_snp_indices)
    assert len(seen) == ed.split.n_folds


# ---------------------------------------------------------------------------
# old paths must not be active
# ---------------------------------------------------------------------------
def test_genomic_data_module_removed_from_package():
    import contrastive_learning

    assert not hasattr(contrastive_learning, "GenomicDataModule")


def _package_src(name):
    return (REPO / "contrastive_learning" / name).read_text()


def test_runner_has_no_stratifiedkfold_or_split():
    src = _package_src("experiment.py")
    assert "StratifiedKFold" not in src
    assert "train_test_split" not in src


def test_data_layer_has_no_old_imputation_or_split():
    src = _package_src("data.py")
    assert "SimpleImputer" not in src
    assert "train_test_split" not in src
    assert "StratifiedKFold" not in src


# ---------------------------------------------------------------------------
# real-data protocol-v2 integration
# ---------------------------------------------------------------------------
FAM = str(REPO / "data" / "ADAPTmap_genotypeTOP_20160222_full.fam")
BED = str(REPO / "data" / "ADAPTmap_genotypeTOP_20160222_full.bed")
OUT = str(REPO / "splits")
DS = "ADAPTmap_genotypeTOP_20160222_full"


@pytest.fixture(scope="module")
def real_split():
    return load_split(DS, 42, 42, FAM, out_dir=OUT, cohort_label="indmiss0p10")


def _real_experiment_data(split):
    return GenomicExperimentData.from_plink(
        split, BED, preprocessor_kwargs={"ld_window": 1}
    )


@pytest.mark.skipif(not (os.path.exists(FAM) and os.path.exists(BED)), reason="data missing")
def test_real_fold_counts_and_no_test_leak(real_split):
    ed = _real_experiment_data(real_split)
    expected = [(1624, 813), (1625, 812), (1625, 812)]
    test_idx = set(real_split.test_indices().tolist())
    for k, (n_tr, n_va) in enumerate(expected):
        fd = ed.fold_data(k)
        assert len(fd.X_train) == n_tr
        assert len(fd.X_val) == n_va
        train_src = set(fd.train_source_index.tolist())
        val_src = set(fd.val_source_index.tolist())
        assert train_src.isdisjoint(test_idx)
        assert val_src.isdisjoint(test_idx)


@pytest.mark.skipif(not (os.path.exists(FAM) and os.path.exists(BED)), reason="data missing")
def test_real_num_classes_and_label_mapping(real_split):
    ed = _real_experiment_data(real_split)
    assert ed.n_classes == 34
    mapper = CanonicalLabelMapper.from_breeds(real_split.breed)
    assert ed.class_names == mapper.classes  # sorted alphabetical 0..33
    assert np.array_equal(ed.y_all(), mapper.encode(real_split.breed))
    for k in range(ed.split.n_folds):
        fd = ed.fold_data(k)
        assert set(np.unique(fd.y_train).tolist()).issubset(set(range(34)))
        assert set(np.unique(fd.y_val).tolist()).issubset(set(range(34)))


@pytest.mark.skipif(not (os.path.exists(FAM) and os.path.exists(BED)), reason="data missing")
def test_real_n_markers_and_clean_dosage(real_split):
    ed = _real_experiment_data(real_split)
    for k in range(ed.split.n_folds):
        fd = ed.fold_data(k)
        assert fd.X_train.shape[1] == fd.X_val.shape[1] == len(fd.retained_snp_indices)
        for mat in (fd.X_train, fd.X_val):
            assert not np.isnan(mat).any()
            assert set(np.unique(mat).tolist()) <= {0.0, 1.0, 2.0}
