"""Test suite for genomic.splitting (leakage-safe dataset splitting).

These tests use synthetic .fam manifests only; no genotype data is required.
"""

import numpy as np
import pytest

from genomic.splitting import (
    SplitIndex,
    _manifest_sha256,
    build_split,
    load_split,
    save_split,
)


def write_fam(path, breeds):
    """Write a synthetic .fam file.

    breeds: dict mapping breed -> list of IIDs.
    """
    lines = []
    for breed, iids in breeds.items():
        for iid in iids:
            lines.append(f"{breed}\t{iid}\t0\t0\t0\t-9\n")
    with open(path, "w") as f:
        f.writelines(lines)


def make_breeds(*specs):
    """Build a breeds dict from (breed, count) pairs with unique IIDs."""
    out = {}
    for breed, n in specs:
        out[breed] = [f"{breed}_{i}" for i in range(n)]
    return out


@pytest.fixture
def fam(tmp_path):
    path = tmp_path / "synth.fam"
    write_fam(path, make_breeds(("A", 30), ("B", 20), ("C", 10)))
    return path


def build(fam_path, **kw):
    defaults = dict(
        dataset_id="synth",
        outer_split_seed=42,
        fold_seed=42,
        dev_frac=0.85,
        test_frac=0.15,
        k_folds=3,
        cohort_min_breed_size=None,
    )
    defaults.update(kw)
    return build_split(str(fam_path), **defaults)


# ---------------------------------------------------------------------------
# Disjointness / containment
# ---------------------------------------------------------------------------
def test_dev_test_disjointness(fam):
    s = build(fam)
    dev = set(s.development_indices().tolist())
    test = set(s.test_indices().tolist())
    assert dev.isdisjoint(test)
    assert dev | test == set(s.source_index.tolist())


def test_fold_test_disjointness(fam):
    s = build(fam)
    test = set(s.test_indices().tolist())
    for k in range(s.n_folds):
        assert set(s.fold_val_indices(k).tolist()).isdisjoint(test)
        assert set(s.fold_train_indices(k).tolist()).isdisjoint(test)


def test_fold_containment_exhaustiveness(fam):
    s = build(fam)
    dev = set(s.development_indices().tolist())
    for k in range(s.n_folds):
        val = set(s.fold_val_indices(k).tolist())
        trn = set(s.fold_train_indices(k).tolist())
        assert val.isdisjoint(trn)
        assert val | trn == dev


def test_fold_assignments_pairwise_disjoint(fam):
    s = build(fam)
    vals = [set(s.fold_val_indices(k).tolist()) for k in range(s.n_folds)]
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            assert vals[i].isdisjoint(vals[j])


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
def test_deterministic_outer_split(fam):
    a = build(fam)
    b = build(fam)
    assert np.array_equal(a.test_indices(), b.test_indices())
    assert np.array_equal(a.development_indices(), b.development_indices())


def test_deterministic_fold_assignment(fam):
    a = build(fam)
    b = build(fam)
    assert np.array_equal(a.fold, b.fold)


def test_different_outer_seed_different_assignment(fam):
    a = build(fam, outer_split_seed=1)
    b = build(fam, outer_split_seed=2)
    assert not np.array_equal(a.test_indices(), b.test_indices())


def test_different_fold_seed_different_assignment(fam):
    a = build(fam, fold_seed=1)
    b = build(fam, fold_seed=2)
    assert not np.array_equal(a.fold, b.fold)


# ---------------------------------------------------------------------------
# Stratification
# ---------------------------------------------------------------------------
def test_stratification_by_breed(fam):
    s = build(fam)
    for breed in ("A", "B", "C"):
        dev_breed = s.breed[s.outer_split == "development"]
        test_breed = s.breed[s.outer_split == "test"]
        assert (test_breed == breed).sum() >= 1
        assert (dev_breed == breed).sum() >= s.n_folds
        # every breed present in every validation fold
        for k in range(s.n_folds):
            mask = (s.outer_split == "development") & (s.fold == k)
            assert (s.breed[mask] == breed).sum() >= 1


def test_stratification_approximate_proportions(fam):
    s = build(fam)
    n_test = len(s.test_indices())
    n_dev = len(s.development_indices())
    n = n_test + n_dev
    assert abs(n_test / n - 0.15) < 0.05
    assert abs(n_dev / n - 0.85) < 0.05


# ---------------------------------------------------------------------------
# Source order preservation
# ---------------------------------------------------------------------------
def test_source_order_preserved(fam):
    s = build(fam)
    assert np.array_equal(s.source_index, np.arange(len(s.source_index)))
    assert np.all(np.diff(s.source_index) > 0)


def test_source_index_maps_back_to_original_rows(fam):
    s = build(fam)
    fam = np.loadtxt(fam, dtype=str)
    full_fid = fam[:, 0]
    full_iid = fam[:, 1]
    for j in range(len(s.source_index)):
        assert s.fid[j] == full_fid[s.source_index[j]]
        assert s.iid[j] == full_iid[s.source_index[j]]
        assert s.sample_id[j] == f"{full_fid[s.source_index[j]]}:{full_iid[s.source_index[j]]}"


def test_source_order_preserved_with_filtering(tmp_path):
    fam = tmp_path / "filter.fam"
    write_fam(fam, make_breeds(("A", 30), ("B", 2)))
    s = build(fam, cohort_min_breed_size=3)
    # B is filtered out (cohort eligibility), A retains original order.
    assert set(s.breed.tolist()) == {"A"}
    assert np.all(np.diff(s.source_index) > 0)


# ---------------------------------------------------------------------------
# Unique sample identifiers
# ---------------------------------------------------------------------------
def test_duplicate_sample_id_rejected(tmp_path):
    fam = tmp_path / "dup.fam"
    with open(fam, "w") as f:
        f.write("A\tX1\t0\t0\t0\t-9\n")
        f.write("A\tX1\t0\t0\t0\t-9\n")
    with pytest.raises(ValueError, match="Duplicate sample_id"):
        build(fam)


# ---------------------------------------------------------------------------
# Sufficiency validation
# ---------------------------------------------------------------------------
def test_insufficient_breed_rejected(tmp_path):
    fam = tmp_path / "tiny.fam"
    write_fam(fam, make_breeds(("A", 30), ("B", 2)))
    with pytest.raises(ValueError, match="Insufficient samples per breed") as e:
        build(fam, k_folds=3, cohort_min_breed_size=None)
    assert "B" in str(e.value)


def test_cohort_min_breed_size_is_cohort_eligibility_only(tmp_path):
    fam = tmp_path / "elig.fam"
    write_fam(fam, make_breeds(("A", 30), ("B", 2)))
    s = build(fam, cohort_min_breed_size=3)
    assert set(s.breed.tolist()) == {"A"}
    assert s.meta["n_eligible"] == 30


# ---------------------------------------------------------------------------
# Manifest fingerprint integrity
# ---------------------------------------------------------------------------
def test_fingerprint_detects_row_change(tmp_path):
    fam = tmp_path / "f1.fam"
    write_fam(fam, make_breeds(("A", 5), ("B", 5)))
    s1 = build(fam)

    with open(fam, "w") as f:
        for line in [
            "A\tA_0\t0\t0\t0\t-9\n",
            "A\tA_1\t0\t0\t0\t-9\n",
            "A\tA_2\t0\t0\t0\t-9\n",
            "A\tA_3\t0\t0\t0\t-9\n",
            "A\tA_4\t0\t0\t0\t-9\n",
            "B\tB_0\t0\t0\t0\t-9\n",
            "B\tB_1\t0\t0\t0\t-9\n",
            "B\tB_2\t0\t0\t0\t-9\n",
            "B\tB_3\t0\t0\t0\t-9\n",
            "B\tB_CHANGED\t0\t0\t0\t-9\n",  # changed IID
        ]:
            f.write(line)

    s2 = build(fam)
    assert s1.meta["source_manifest_sha256"] != s2.meta["source_manifest_sha256"]


def test_fingerprint_detects_reordering(tmp_path):
    fam = tmp_path / "f2.fam"
    write_fam(fam, make_breeds(("A", 5), ("B", 5)))
    s1 = build(fam)

    fam2 = tmp_path / "f2_reordered.fam"
    lines = open(fam).readlines()
    lines[0], lines[1] = lines[1], lines[0]
    with open(fam2, "w") as f:
        f.writelines(lines)
    s2 = build(fam2)
    assert s1.meta["source_manifest_sha256"] != s2.meta["source_manifest_sha256"]


def test_load_split_rejects_stale_fam(tmp_path):
    fam = tmp_path / "f3.fam"
    write_fam(fam, make_breeds(("A", 5), ("B", 5)))
    s = build(fam, dataset_id="ds")
    out_dir = tmp_path / "splits"
    save_split(s, out_dir=str(out_dir))

    # Mutate the .fam after the split was persisted.
    with open(fam, "w") as f:
        for line in [
            "A\tA_0\t0\t0\t0\t-9\n",
            "A\tA_1\t0\t0\t0\t-9\n",
            "A\tA_2\t0\t0\t0\t-9\n",
            "A\tA_3\t0\t0\t0\t-9\n",
            "A\tA_4\t0\t0\t0\t-9\n",
            "B\tB_0\t0\t0\t0\t-9\n",
            "B\tB_1\t0\t0\t0\t-9\n",
            "B\tB_2\t0\t0\t0\t-9\n",
            "B\tB_3\t0\t0\t0\t-9\n",
            "C\tC_0\t0\t0\t0\t-9\n",  # breed changed
        ]:
            f.write(line)

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_split("ds", 42, 42, str(fam), out_dir=str(out_dir))


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------
def test_save_load_roundtrip(fam, tmp_path):
    s = build(fam, dataset_id="ds", outer_split_seed=7, fold_seed=11)
    out_dir = tmp_path / "splits"
    csv_path, meta_path = save_split(s, out_dir=str(out_dir))

    assert csv_path.endswith("ds_outer7_fold11.csv")
    assert meta_path.endswith("ds_outer7_fold11.meta.json")

    loaded = load_split("ds", 7, 11, str(fam), out_dir=str(out_dir))
    assert isinstance(loaded, SplitIndex)
    assert np.array_equal(loaded.source_index, s.source_index)
    assert np.array_equal(loaded.fid, s.fid)
    assert np.array_equal(loaded.iid, s.iid)
    assert np.array_equal(loaded.sample_id, s.sample_id)
    assert np.array_equal(loaded.breed, s.breed)
    assert np.array_equal(loaded.outer_split, s.outer_split)
    assert np.array_equal(loaded.fold, s.fold)
    assert loaded.meta == s.meta


def test_meta_schema(fam):
    s = build(fam, dataset_id="ds")
    m = s.meta
    for key in [
        "protocol_version",
        "dataset_id",
        "outer_split_seed",
        "fold_seed",
        "dev_frac",
        "test_frac",
        "k_folds",
        "cohort_min_breed_size",
        "n_total",
        "n_eligible",
        "n_development",
        "n_test",
        "source_manifest_sha256",
    ]:
        assert key in m
    assert m["n_total"] == m["n_eligible"] == 60
    assert m["n_development"] + m["n_test"] == m["n_eligible"]


# ---------------------------------------------------------------------------
# RQ2 cohort derivation (in-memory breed filter over the persisted split)
# ---------------------------------------------------------------------------
def test_train_counts_per_breed(fam):
    s = build(fam)
    counts = s.train_counts_per_breed()
    assert set(counts) == {"A", "B", "C"}
    for breed, folds in counts.items():
        assert len(folds) == s.n_folds
        assert all(c >= 0 for c in folds)
        # training counts must be consistent with development minus one fold
        dev_n = int((s.outer_split == "development")[s.breed == breed].sum())
        assert sum(folds) == dev_n * (s.n_folds - 1)


def test_rq2_breeds_is_subset_and_matches_criterion(fam):
    s = build(fam)
    counts = s.train_counts_per_breed()
    for threshold in (1, 5, 12, 20):
        expected = sorted(b for b in counts if min(counts[b]) >= threshold)
        assert s.rq2_breeds(min_train_per_fold=threshold) == expected
        assert set(s.rq2_breeds(min_train_per_fold=threshold)).issubset(set(counts))


def test_rq2_breeds_never_reassigns_folds(fam):
    s = build(fam)
    full_fold = s.fold.copy()
    r = s.rq2_breeds(min_train_per_fold=5)
    # deriving the RQ2 cohort must not mutate the split or fold assignment
    assert np.array_equal(s.fold, full_fold)
    assert set(r).issubset(set(s.breed.tolist()))
