"""Integration tests for the VAE/VMGP pipeline over the shared protocol-v2
experiment-data layer.

Synthetic tests use a ``GenomicExperimentData`` built from a small synthetic
``.fam`` + raw genotype matrix (no PLINK binary files). Real-data tests verify
the frozen protocol-v2 fold sizes / 34-class mapping / dosage cleanliness when
the data is present.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from genomic.experiment_data import GenomicExperimentData
from genomic.labels import CanonicalLabelMapper
from genomic.splitting import build_split, load_split

from vae.data_module import build_vae_fold, build_fold_dataloaders
from vae.network import VMGP_Network

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


def build_fold(ed, k):
    return build_vae_fold(
        ed.fold_data(k), class_names_breed=ed.class_names
    )


# ---------------------------------------------------------------------------
# fold source identity / dosage / num_snps (synthetic)
# ---------------------------------------------------------------------------
def test_vae_fold_source_indices_equal_folddata(tmp_path):
    ed = synthetic_experiment_data(tmp_path)
    for k in range(ed.split.n_folds):
        vfd = build_fold(ed, k)
        fd = ed.fold_data(k)
        assert np.array_equal(vfd.train_source_index, fd.train_source_index)
        assert np.array_equal(vfd.val_source_index, fd.val_source_index)
        assert np.array_equal(vfd.train_source_index, ed.split.fold_train_indices(k))
        assert np.array_equal(vfd.val_source_index, ed.split.fold_val_indices(k))


def test_vae_input_values_are_012_no_nan(tmp_path):
    ed = synthetic_experiment_data(tmp_path)
    for k in range(ed.split.n_folds):
        vfd = build_fold(ed, k)
        for mat in (vfd.X_train, vfd.X_val):
            assert not np.isnan(mat).any()
            assert set(np.unique(mat).tolist()) <= {0.0, 1.0, 2.0}


def test_fold_specific_num_snps(tmp_path):
    ed = synthetic_experiment_data(tmp_path)
    for k in range(ed.split.n_folds):
        vfd = build_fold(ed, k)
        assert vfd.num_snps == vfd.X_train.shape[1]
        assert vfd.num_snps == len(vfd.retained_snp_indices)


def test_model_constructor_accepts_fold_specific_num_snps(tmp_path):
    ed = synthetic_experiment_data(tmp_path)
    for k in range(ed.split.n_folds):
        vfd = build_fold(ed, k)
        model = VMGP_Network(
            num_snps=vfd.num_snps, num_classes_breed=vfd.num_classes_breed,
            use_breed=True,
        )
        x = torch.FloatTensor(vfd.X_train[:4])
        out = model(x)
        assert out["x_recon"].shape == (4, vfd.num_snps)
        assert out["logits_breed"].shape == (4, vfd.num_classes_breed)


def test_repeated_fold_loading_deterministic(tmp_path):
    ed = synthetic_experiment_data(tmp_path)
    f1 = build_fold(ed, 0)
    f2 = build_fold(ed, 0)
    assert np.array_equal(f1.X_train, f2.X_train)
    assert np.array_equal(f1.X_val, f2.X_val)
    assert np.array_equal(f1.y_breed_train, f2.y_breed_train)
    assert np.array_equal(f1.retained_snp_indices, f2.retained_snp_indices)


def test_preprocessing_state_unchanged_by_dataloader(tmp_path):
    ed = synthetic_experiment_data(tmp_path)
    vfd = build_fold(ed, 0)
    before = vfd.retained_snp_indices.copy()
    tr_dl, val_dl = build_fold_dataloaders(vfd, 8, num_workers=0, pin_memory=False)
    _ = list(tr_dl), list(val_dl)
    assert np.array_equal(vfd.retained_snp_indices, before)


# ---------------------------------------------------------------------------
# auxiliary targets (synthetic) — never drop samples, row-aligned
# ---------------------------------------------------------------------------
def test_auxiliary_targets_align_and_never_drop(tmp_path):
    csv = tmp_path / "meta.csv"
    with open(csv, "w") as f:
        f.write("Breed,Country,Continent,sub_area,Caseina\n")
        f.write("ALP,SWITZERLAND,Europe,Europe,\n")
        f.write("BOR,FRANCE,Europe,Europe,Forte\n")
        # CEN intentionally absent -> unmatched

    ed = synthetic_experiment_data(tmp_path)
    vfd = build_vae_fold(
        ed.fold_data(0),
        class_names_breed=ed.class_names,
        metadata_path=str(csv),
        use_continent=True,
        use_caseina=True,
        use_attitudine=True,
    )
    assert len(vfd.y_continent_train) == len(vfd.X_train)
    assert len(vfd.y_continent_val) == len(vfd.X_val)
    assert len(vfd.y_caseina_train) == len(vfd.X_train)
    assert len(vfd.y_attitudine_train) == len(vfd.X_train)
    # unmatched breeds -> -1 (continent/caseina) or ALTRO (attitudine), never dropped
    assert (-1 in vfd.y_continent_train) or (-1 in vfd.y_continent_val)


def test_auxiliary_labels_row_aligned_with_source_indices(tmp_path):
    csv = tmp_path / "meta.csv"
    with open(csv, "w") as f:
        f.write("Breed,Country,Continent,sub_area,Caseina\n")
        f.write("ALP,SWITZERLAND,Europe,Europe,\n")

    ed = synthetic_experiment_data(tmp_path)
    fd = ed.fold_data(0)
    vfd = build_vae_fold(
        fd, class_names_breed=ed.class_names, metadata_path=str(csv),
        use_continent=True,
    )
    # breed labels and auxiliary labels share the same row order (same length)
    assert len(vfd.y_breed_train) == len(vfd.y_continent_train) == len(fd.train_source_index)
    assert len(vfd.y_breed_val) == len(vfd.y_continent_val) == len(fd.val_source_index)


# ---------------------------------------------------------------------------
# old paths must not be active
# ---------------------------------------------------------------------------
def _pkg_src(name):
    return (REPO / "vae" / name).read_text()


def test_data_module_has_no_simple_imputer():
    src = _pkg_src("data_module.py")
    assert "SimpleImputer" not in src
    assert "train_test_split" not in src
    assert "StratifiedKFold" not in src


def test_experiment_has_no_split_or_stratifiedkfold():
    src = _pkg_src("experiment.py")
    assert "train_test_split" not in src
    assert "StratifiedKFold" not in src
    assert "min_samples_per_class" not in src


def test_genomic_data_module_removed_from_vae():
    import vae

    assert not hasattr(vae, "GenomicDataModule")


# ---------------------------------------------------------------------------
# cross-model consistency: VAE consumes the same fold data as contrastive
# ---------------------------------------------------------------------------
def test_vae_consumes_same_fold_data_as_shared_layer(tmp_path):
    ed = synthetic_experiment_data(tmp_path)
    for k in range(ed.split.n_folds):
        fd = ed.fold_data(k)
        vfd = build_vae_fold(fd, class_names_breed=ed.class_names)
        assert np.array_equal(vfd.X_train, fd.X_train)
        assert np.array_equal(vfd.X_val, fd.X_val)
        assert np.array_equal(vfd.y_breed_train, fd.y_train)
        assert np.array_equal(vfd.y_breed_val, fd.y_val)
        assert np.array_equal(vfd.retained_snp_indices, fd.retained_snp_indices)


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


def _real_ed(split):
    return GenomicExperimentData.from_plink(split, BED, preprocessor_kwargs={"ld_window": 1})


@pytest.mark.skipif(not (os.path.exists(FAM) and os.path.exists(BED)), reason="data missing")
def test_real_fold_sizes_and_no_test_leak(real_split):
    ed = _real_ed(real_split)
    expected = [(1624, 813), (1625, 812), (1625, 812)]
    test_idx = set(real_split.test_indices().tolist())
    for k, (n_tr, n_va) in enumerate(expected):
        vfd = build_vae_fold(ed.fold_data(k), class_names_breed=ed.class_names)
        assert len(vfd.X_train) == n_tr
        assert len(vfd.X_val) == n_va
        assert set(vfd.train_source_index.tolist()).isdisjoint(test_idx)
        assert set(vfd.val_source_index.tolist()).isdisjoint(test_idx)


@pytest.mark.skipif(not (os.path.exists(FAM) and os.path.exists(BED)), reason="data missing")
def test_real_breed_mapping_and_34_classes(real_split):
    ed = _real_ed(real_split)
    mapper = CanonicalLabelMapper.from_breeds(real_split.breed)
    assert ed.n_classes == 34
    vfd = build_vae_fold(ed.fold_data(0), class_names_breed=ed.class_names)
    assert vfd.num_classes_breed == 34
    assert vfd.class_names_breed == mapper.classes
    assert set(np.unique(vfd.y_breed_train).tolist()).issubset(set(range(34)))


@pytest.mark.skipif(not (os.path.exists(FAM) and os.path.exists(BED)), reason="data missing")
def test_real_breed_labels_equal_shared_mapper(real_split):
    ed = _real_ed(real_split)
    mapper = CanonicalLabelMapper.from_breeds(real_split.breed)
    for k in range(ed.split.n_folds):
        fd = ed.fold_data(k)
        vfd = build_vae_fold(fd, class_names_breed=ed.class_names)
        assert np.array_equal(vfd.y_breed_train, mapper.encode(fd.train_breed))
        assert np.array_equal(vfd.y_breed_val, mapper.encode(fd.val_breed))


@pytest.mark.skipif(not (os.path.exists(FAM) and os.path.exists(BED)), reason="data missing")
def test_real_dosage_clean_and_num_snps(real_split):
    ed = _real_ed(real_split)
    for k in range(ed.split.n_folds):
        vfd = build_vae_fold(ed.fold_data(k), class_names_breed=ed.class_names)
        assert vfd.num_snps == vfd.X_train.shape[1] == len(vfd.retained_snp_indices)
        for mat in (vfd.X_train, vfd.X_val):
            assert not np.isnan(mat).any()
            assert set(np.unique(mat).tolist()) <= {0.0, 1.0, 2.0}
