"""Tests for pre-split sample-level cohort eligibility (individual QC) and the
protocol-v2 split generation.
"""

import os

import numpy as np
import pytest

from genomic.cohort import compute_individual_missingness, individual_qc_mask
from genomic.splitting import PROTOCOL_VERSION, build_split, load_split

FAM = "data/ADAPTmap_genotypeTOP_20160222_full.fam"
BED = "data/ADAPTmap_genotypeTOP_20160222_full.bed"
OUT = "splits"
DS = "ADAPTmap_genotypeTOP_20160222_full"

EXPECTED_RQ2 = [
    "ABR", "ALP", "ANG", "BOE", "BRK", "BUR", "CRE", "LNR",
    "NBN", "OSS", "RAN", "SAA", "SEA", "SID", "WAD",
]


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


# ---------------------------------------------------------------------------
# individual_qc_mask (pure rule)
# ---------------------------------------------------------------------------
def test_individual_qc_mask_uses_strict_less_than():
    m = np.array([0.05, 0.10, 0.15])
    mask = individual_qc_mask(m, 0.10)
    assert mask.tolist() == [True, False, False]


def test_individual_qc_mask_is_per_sample_independent():
    m = np.array([0.05, 0.5])
    assert individual_qc_mask(m, 0.10).tolist() == [True, False]
    # adding a third sample must not change the first two decisions
    m3 = np.array([0.05, 0.5, 0.99])
    assert individual_qc_mask(m3, 0.10)[:2].tolist() == [True, False]


# ---------------------------------------------------------------------------
# build_split with an individual-QC mask (synthetic)
# ---------------------------------------------------------------------------
def test_build_split_applies_individual_qc_before_breed(tmp_path):
    fam = tmp_path / "s.fam"
    # COMMON: 40 (remove 5 -> 35, kept). DROPOUT: 31 (remove 2 -> 29, excluded).
    write_fam(fam, make_breeds(("COMMON", 40), ("DROPOUT", 31)))
    fam_arr = np.loadtxt(fam, dtype=str)

    mask = np.ones(len(fam_arr), dtype=bool)
    mask[np.where(fam_arr[:, 0] == "COMMON")[0][:5]] = False
    mask[np.where(fam_arr[:, 0] == "DROPOUT")[0][:2]] = False

    split = build_split(
        str(fam), dataset_id="ds", cohort_min_breed_size=30,
        individual_missingness_mask=mask,
        individual_missingness_threshold=0.10,
    )
    assert set(split.breed.tolist()) == {"COMMON"}


def test_build_split_breed_threshold_counts_qc_eligible_only(tmp_path):
    fam = tmp_path / "s.fam"
    # COMMON: 40 samples, remove 5 -> 35 eligible (>=30, kept).
    write_fam(fam, make_breeds(("COMMON", 40), ("OTHER", 40)))
    fam_arr = np.loadtxt(fam, dtype=str)

    mask = np.ones(len(fam_arr), dtype=bool)
    common_idx = np.where(fam_arr[:, 0] == "COMMON")[0]
    mask[common_idx[:5]] = False

    split = build_split(
        str(fam), dataset_id="ds", cohort_min_breed_size=30,
        individual_missingness_mask=mask,
        individual_missingness_threshold=0.10,
    )
    breeds = set(split.breed.tolist())
    assert breeds == {"COMMON", "OTHER"}
    # COMMON has exactly 35 eligible samples
    assert int((split.breed == "COMMON").sum()) == 35


def test_no_previously_ineligible_breed_becomes_eligible(tmp_path):
    fam = tmp_path / "s.fam"
    write_fam(fam, make_breeds(("COMMON", 40), ("RARE", 25)))
    fam_arr = np.loadtxt(fam, dtype=str)
    mask = np.ones(len(fam_arr), dtype=bool)  # remove nothing
    split = build_split(
        str(fam), dataset_id="ds", cohort_min_breed_size=30,
        individual_missingness_mask=mask,
        individual_missingness_threshold=0.10,
    )
    assert "RARE" not in set(split.breed.tolist())


def test_build_split_preserves_source_order_and_index(tmp_path):
    fam = tmp_path / "s.fam"
    write_fam(fam, make_breeds(("A", 30), ("B", 30)))
    fam_arr = np.loadtxt(fam, dtype=str)
    mask = np.ones(len(fam_arr), dtype=bool)
    mask[0] = False  # remove first A sample
    split = build_split(
        str(fam), dataset_id="ds", cohort_min_breed_size=30,
        individual_missingness_mask=mask,
        individual_missingness_threshold=0.10,
    )
    assert np.all(np.diff(split.source_index) > 0)  # strictly increasing
    for j in range(len(split.source_index)):
        assert split.fid[j] == fam_arr[split.source_index[j], 0]
        assert split.iid[j] == fam_arr[split.source_index[j], 1]


def test_mask_length_mismatch_raises(tmp_path):
    fam = tmp_path / "s.fam"
    write_fam(fam, make_breeds(("A", 30), ("B", 30)))
    with pytest.raises(ValueError, match="one entry per raw sample"):
        build_split(
            str(fam), dataset_id="ds", cohort_min_breed_size=30,
            individual_missingness_mask=np.ones(10, dtype=bool),
            individual_missingness_threshold=0.10,
        )


def test_cohort_fingerprint_changes_with_eligible_set(tmp_path):
    fam = tmp_path / "s.fam"
    write_fam(fam, make_breeds(("A", 30), ("B", 30)))
    fam_arr = np.loadtxt(fam, dtype=str)

    mask1 = np.ones(len(fam_arr), dtype=bool)
    mask2 = np.ones(len(fam_arr), dtype=bool)
    mask2[0] = False

    s1 = build_split(str(fam), dataset_id="ds", cohort_min_breed_size=30,
                     individual_missingness_mask=mask1, individual_missingness_threshold=0.10)
    s2 = build_split(str(fam), dataset_id="ds", cohort_min_breed_size=30,
                     individual_missingness_mask=mask2, individual_missingness_threshold=0.10)

    assert s1.meta["cohort_manifest_sha256"] != s2.meta["cohort_manifest_sha256"]
    assert s1.meta["source_manifest_sha256"] == s2.meta["source_manifest_sha256"]


# ---------------------------------------------------------------------------
# Real-data protocol-v2 integration
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not (os.path.exists(FAM) and os.path.exists(BED)), reason="data missing")
def test_real_cohort_split_and_rq2():
    missingness = compute_individual_missingness(BED, FAM)
    assert len(missingness) == 4653
    mask = individual_qc_mask(missingness, 0.10)

    split = build_split(
        FAM, dataset_id=DS,
        cohort_min_breed_size=30,
        individual_missingness_mask=mask,
        individual_missingness_threshold=0.10,
        individual_missingness_rule="<",
    )

    m = split.meta
    assert m["protocol_version"] == PROTOCOL_VERSION == 2
    assert m["raw_sample_count"] == 4653
    assert m["final_primary_cohort_count"] == 2863
    assert m["eligible_after_individual_qc_count"] == int(mask.sum())
    assert m["n_development"] == 2437
    assert m["n_test"] == 426
    assert len(set(split.breed.tolist())) == 34

    # disjoint + exhaustive
    dev = set(split.development_indices().tolist())
    test = set(split.test_indices().tolist())
    assert dev.isdisjoint(test)
    assert dev | test == set(split.source_index.tolist())

    # fold sizes
    for k, (exp_tr, exp_va) in enumerate([(1624, 813), (1625, 812), (1625, 812)]):
        assert len(split.fold_train_indices(k)) == exp_tr
        assert len(split.fold_val_indices(k)) == exp_va

    # RQ2 derivation
    rq2 = split.rq2_breeds(min_train_per_fold=30)
    assert set(rq2) == set(EXPECTED_RQ2)
    assert len(rq2) == 15
    total = sum(int((split.breed == b).sum()) for b in rq2)
    assert total == 2126
    counts = split.train_counts_per_breed()
    for b in rq2:
        for N in [5, 10, 15, 20, 25, 30]:
            assert min(counts[b]) >= N


@pytest.mark.skipif(not os.path.exists(OUT), reason="splits dir missing")
def test_protocol_v1_artifacts_untouched():
    import json
    v1_meta = os.path.join(OUT, "ADAPTmap_genotypeTOP_20160222_full_outer42_fold42.meta.json")
    v1_csv = os.path.join(OUT, "ADAPTmap_genotypeTOP_20160222_full_outer42_fold42.csv")
    assert os.path.exists(v1_meta) and os.path.exists(v1_csv)
    meta = json.load(open(v1_meta))
    assert meta["protocol_version"] == 1
    assert meta["n_eligible"] == 2976
