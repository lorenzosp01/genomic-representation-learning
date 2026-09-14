"""Tests for the RQ2 sample-efficiency infrastructure.

No real RQ2 training is executed: the VMGP runner is exercised with a fake
``run_experiment`` / fake Trainer on synthetic development data, and real-data
tests are limited to split/cohort metadata + development-only raw data (never
the locked-test genotypes).
"""

import dataclasses
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from genomic.classification_metrics import compute_classification_metrics
from genomic.experiment_data import DevelopmentRawData, GenomicExperimentData, locked_test_overlap
from genomic.preprocessing import GenomicPreprocessor
from genomic.rq2 import (
    EPSILON_N,
    RQ2_BREEDS,
    RQ2_FOLDS,
    RQ2_MAX_N,
    RQ2_N_GRID,
    RQ2_SEEDS,
    aggregate_rq2,
    assert_rq2_breeds,
    breed_availability_table,
    build_rq2_sample,
    load_json,
    rq2_fold_partitions,
    sample_nested_rq2,
    validate_run_record,
)
from genomic.splitting import build_split, load_split

import vae.experiment as vae_experiment
from vae.rq2_experiment import (
    RQ2_LATENT_DIM,
    RQ2_MAX_EPOCHS,
    build_rq2_vae_fold_data,
    load_completed_runs,
    rq2_vae_config,
    run_rq2_experiment,
    run_rq2_single,
)

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# synthetic fixtures / helpers
# ---------------------------------------------------------------------------
def make_split(tmp_path, n_per_breed=60):
    fam = tmp_path / "s.fam"
    lines = []
    for b in RQ2_BREEDS:
        for i in range(n_per_breed):
            lines.append(f"{b}\t{b}_{i}\t0\t0\t0\t-9\n")
    with open(fam, "w") as f:
        f.writelines(lines)
    return build_split(str(fam), dataset_id="s", k_folds=3, cohort_min_breed_size=None)


def make_dev_raw(tmp_path, n_per_breed=60, n_snps=8, seed=0):
    split = make_split(tmp_path, n_per_breed)
    rng = np.random.RandomState(seed)
    X = rng.randint(0, 3, size=(split.meta["n_eligible"], n_snps)).astype(np.float32)
    ed = GenomicExperimentData(split, X)
    return ed.development_raw_data(), ed


class _NoopPlt:
    def __getattr__(self, name):
        return lambda *a, **k: None


class _FakeTrainer:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.fit_loaders = None
        _FakeTrainer.instances.append(self)

    def fit(self, model, *loaders):
        self.fit_loaders = loaders

    def save_checkpoint(self, path):
        self.saved = path


class _CbTrainer:
    def __init__(self):
        self.sanity_checking = False
        self.current_epoch = 1
        self.max_epochs = RQ2_MAX_EPOCHS
        self.early_stopping_callback = None


def make_fake_run_experiment(calls):
    """Fake run_experiment: records kwargs, fills the history callback, returns metrics."""
    def fake(**kwargs):
        calls.append(kwargs)
        cb = kwargs.get("history_callback")
        if cb is not None:
            cb.on_train_epoch_start(_CbTrainer(), None)
            cb.on_train_end(_CbTrainer(), None)
        n = len(RQ2_BREEDS)
        return {
            "Macro_F1 (mean)": 0.90,
            "Balanced_Accuracy (mean)": 0.91,
            "Accuracy (mean)": 0.92,
            "Per_Class_Recall (mean)": np.full(n, 0.9),
            "Confusion_Matrix": np.eye(n, dtype=np.int64),
            "Num SNPs": 8,
            "Best_Epochs": [10],
        }
    return fake


def make_record(fold, seed, n, macro_f1=0.9, bal=0.9, acc=0.9):
    n_classes = len(RQ2_BREEDS)
    return {
        "model": "vae", "latent_dim": RQ2_LATENT_DIM,
        "fold": fold, "replicate_seed": seed, "N": n,
        "macro_f1": macro_f1, "balanced_accuracy": bal, "accuracy": acc,
        "per_class_recall": [0.9] * n_classes,
        "confusion_matrix": np.eye(n_classes, dtype=np.int64).tolist(),
        "class_names": list(RQ2_BREEDS), "n_classes": n_classes,
        "retained_snp_count": 8, "best_epoch": 10, "completed_epochs": 20,
        "early_stopping_triggered": True, "max_epochs": RQ2_MAX_EPOCHS,
        "runtime_s": 1.0, "n_train": 15 * n, "n_val": 10,
        "train_source_index": list(range(15 * n)), "val_source_index": [0],
        "preprocessing": {"marker_missingness_threshold": 0.1, "maf_threshold": 0.01,
                          "ld_window": 50, "ld_r2_threshold": 0.2,
                          "n_features_in": 8, "n_retained_snps": 8},
        "checkpoint_kept": False,
    }


# ---------------------------------------------------------------------------
# constants / cohort
# ---------------------------------------------------------------------------
def test_rq2_constants():
    assert list(RQ2_N_GRID) == [5, 10, 15, 20, 25, 30]
    assert list(RQ2_SEEDS) == [42, 43, 44, 45, 46]
    assert list(RQ2_FOLDS) == [0, 1, 2]
    assert EPSILON_N == 0.02
    assert RQ2_MAX_N == 30


def test_rq2_breeds_exact_synthetic(tmp_path):
    split = make_split(tmp_path)
    breeds = split.rq2_breeds(30)
    assert_rq2_breeds(breeds)
    assert breeds == RQ2_BREEDS


def test_breed_availability_min_ge_30_synthetic(tmp_path):
    split = make_split(tmp_path)
    table = breed_availability_table(split)
    assert [r["breed"] for r in table] == RQ2_BREEDS
    assert all(r["min_train_count"] >= 30 for r in table)


FAM = str(REPO / "data" / "ADAPTmap_genotypeTOP_20160222_full.fam")
BED = str(REPO / "data" / "ADAPTmap_genotypeTOP_20160222_full.bed")
BIM = str(REPO / "data" / "ADAPTmap_genotypeTOP_20160222_full.bim")
OUT = str(REPO / "splits")
DS = "ADAPTmap_genotypeTOP_20160222_full"
DATA_PRESENT = Path(FAM).exists() and Path(BED).exists()


@pytest.mark.skipif(not DATA_PRESENT, reason="data missing")
def test_rq2_breeds_and_availability_real():
    split = load_split(DS, 42, 42, FAM, out_dir=OUT, cohort_label="indmiss0p10")
    assert_rq2_breeds(split.rq2_breeds(30))
    table = breed_availability_table(split)
    assert all(r["min_train_count"] >= 30 for r in table)


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------
def test_nested_subsets_and_exactly_n_per_breed(tmp_path):
    dev_raw, _ = make_dev_raw(tmp_path)
    sampled, _ = sample_nested_rq2(dev_raw, fold=0, seed=42)

    for n in RQ2_N_GRID:
        for b in RQ2_BREEDS:
            assert int(np.sum(dev_raw.dev_breed[sampled[n]] == b)) == n

    ns = list(RQ2_N_GRID)
    for a, b in zip(ns[:-1], ns[1:]):
        assert set(sampled[a].tolist()) <= set(sampled[b].tolist())


def test_sampling_deterministic_and_fold_seed_specific(tmp_path):
    dev_raw, _ = make_dev_raw(tmp_path)
    s1, _ = sample_nested_rq2(dev_raw, fold=0, seed=42)
    s2, _ = sample_nested_rq2(dev_raw, fold=0, seed=42)
    for n in RQ2_N_GRID:
        assert np.array_equal(s1[n], s2[n])

    s3, _ = sample_nested_rq2(dev_raw, fold=0, seed=43)
    s4, _ = sample_nested_rq2(dev_raw, fold=1, seed=42)
    assert not np.array_equal(s1[30], s3[30])
    assert not np.array_equal(s1[30], s4[30])


def test_validation_restricted_to_15_breeds_and_invariant(tmp_path):
    dev_raw, _ = make_dev_raw(tmp_path)
    for fold in RQ2_FOLDS:
        _, val_pos = rq2_fold_partitions(dev_raw, fold)
        assert set(dev_raw.dev_breed[val_pos].tolist()) <= set(RQ2_BREEDS)
        # invariant under repeated calls / seeds / N
        _, val_pos2 = rq2_fold_partitions(dev_raw, fold)
        assert np.array_equal(val_pos, val_pos2)


def test_zero_locked_test_overlap(tmp_path):
    dev_raw, ed = make_dev_raw(tmp_path)
    for fold in RQ2_FOLDS:
        sampled, _ = sample_nested_rq2(dev_raw, fold=fold, seed=42)
        _, val_pos = rq2_fold_partitions(dev_raw, fold)
        assert locked_test_overlap(dev_raw.dev_source_index[sampled[30]], ed.split) == 0
        assert locked_test_overlap(dev_raw.dev_source_index[val_pos], ed.split) == 0


# ---------------------------------------------------------------------------
# development-only raw data (locked test structurally absent)
# ---------------------------------------------------------------------------
def test_development_raw_data_excludes_locked_test(tmp_path):
    dev_raw, ed = make_dev_raw(tmp_path)
    test_idx = set(ed.split.test_indices().tolist())
    assert set(dev_raw.dev_source_index.tolist()).isdisjoint(test_idx)
    assert dev_raw.n_samples == len(ed.split.development_indices())


def test_development_raw_data_has_no_test_field(tmp_path):
    dev_raw, _ = make_dev_raw(tmp_path)
    names = [f.name for f in dataclasses.fields(DevelopmentRawData)]
    assert not any("test" in name.lower() for name in names)
    assert not hasattr(dev_raw, "X_test")
    assert not hasattr(dev_raw, "test_source_index")


def test_development_raw_data_is_a_copy(tmp_path):
    dev_raw, ed = make_dev_raw(tmp_path)
    dev_pos = np.flatnonzero(ed.split.outer_split == "development")
    before = ed._X[dev_pos].copy()
    dev_raw.X_dev[:] = -99.0
    # mutating the development-only object must not modify the source matrix
    assert np.array_equal(ed._X[dev_pos], before)


@pytest.mark.skipif(not DATA_PRESENT, reason="data missing")
def test_real_development_raw_data_sizes():
    split = load_split(DS, 42, 42, FAM, out_dir=OUT, cohort_label="indmiss0p10")
    ed = GenomicExperimentData.from_plink(split, BED, bim_path=BIM)
    dev_raw = ed.development_raw_data()
    assert dev_raw.n_samples == 2437
    assert dev_raw.n_classes == 34
    assert len(split.test_indices()) == 426
    assert locked_test_overlap(dev_raw.dev_source_index, split) == 0


# ---------------------------------------------------------------------------
# fresh preprocessing on S_N only
# ---------------------------------------------------------------------------
def test_preprocessing_fit_receives_only_s_n(monkeypatch, tmp_path):
    dev_raw, _ = make_dev_raw(tmp_path)
    mapper = _mapper15()
    calls = []
    orig_fit = GenomicPreprocessor.fit

    def spy(self, X, snp_metadata=None):
        calls.append(int(np.asarray(X).shape[0]))
        return orig_fit(self, X, snp_metadata=snp_metadata)

    monkeypatch.setattr(GenomicPreprocessor, "fit", spy)
    sampled, _ = sample_nested_rq2(dev_raw, fold=0, seed=42)
    _, val_pos = rq2_fold_partitions(dev_raw, 0)
    for n in RQ2_N_GRID:
        sample = build_rq2_sample(dev_raw, fold=0, seed=42, n=n,
                                  train_positions=sampled[n], val_positions=val_pos, mapper=mapper)
        assert sample.n_train == len(RQ2_BREEDS) * n
    assert calls == [len(RQ2_BREEDS) * n for n in RQ2_N_GRID]


def test_fresh_preprocessor_per_run(tmp_path):
    dev_raw, _ = make_dev_raw(tmp_path)
    mapper = _mapper15()
    sampled, _ = sample_nested_rq2(dev_raw, fold=0, seed=42)
    _, val_pos = rq2_fold_partitions(dev_raw, 0)
    a = build_rq2_sample(dev_raw, fold=0, seed=42, n=5, train_positions=sampled[5],
                         val_positions=val_pos, mapper=mapper)
    b = build_rq2_sample(dev_raw, fold=0, seed=42, n=5, train_positions=sampled[5],
                         val_positions=val_pos, mapper=mapper)
    assert a.preprocessor is not b.preprocessor


def test_test_only_perturbation_cannot_affect_preprocessing(tmp_path):
    dev_raw1, ed = make_dev_raw(tmp_path)
    mapper = _mapper15()
    sampled, _ = sample_nested_rq2(dev_raw1, fold=0, seed=42)
    _, val_pos = rq2_fold_partitions(dev_raw1, 0)
    base = build_rq2_sample(dev_raw1, fold=0, seed=42, n=5, train_positions=sampled[5],
                            val_positions=val_pos, mapper=mapper)

    # Perturb only locked-test genotype rows in the underlying matrix.
    test_pos = np.flatnonzero(ed.split.outer_split == "test")
    ed._X[test_pos] = np.nan

    dev_raw2 = ed.development_raw_data()
    after = build_rq2_sample(dev_raw2, fold=0, seed=42, n=5, train_positions=sampled[5],
                             val_positions=val_pos, mapper=mapper)
    assert np.array_equal(base.preprocessor.retained_snp_indices_,
                          after.preprocessor.retained_snp_indices_)
    assert np.array_equal(base.X_train, after.X_train)


def _mapper15():
    from genomic.labels import CanonicalLabelMapper
    return CanonicalLabelMapper.from_breeds(RQ2_BREEDS)


# ---------------------------------------------------------------------------
# RQ2 VMGP runner (fake run_experiment; no training)
# ---------------------------------------------------------------------------
def test_rq2_config_frozen():
    cfg = rq2_vae_config(accelerator="cpu")
    assert cfg["latent_dim"] == 96
    assert cfg["alpha"] == 0.5
    assert cfg["lr"] == 1e-4
    assert cfg["batch_size"] == 64
    assert RQ2_MAX_EPOCHS == 200


def test_run_rq2_single_passes_rq2_semantics(tmp_path):
    dev_raw, _ = make_dev_raw(tmp_path)
    mapper = _mapper15()
    calls = []
    fake = make_fake_run_experiment(calls)
    sampled, _ = sample_nested_rq2(dev_raw, fold=0, seed=42)
    _, val_pos = rq2_fold_partitions(dev_raw, 0)

    rec = run_rq2_single(
        dev_raw, mapper, fold=0, seed=42, n=5,
        train_positions=sampled[5], val_positions=val_pos,
        out_dir=str(tmp_path / "out"), ckpt_dir=str(tmp_path / "ckpt"),
        run_experiment_fn=fake, accelerator="cpu",
    )
    kw = calls[0]
    assert kw["drop_last"] is False                 # RQ2-specific deviation
    assert kw["config"]["latent_dim"] == RQ2_LATENT_DIM == 96
    assert kw["max_epochs"] == RQ2_MAX_EPOCHS == 200
    assert kw["classifier_config"] == "breed_only"
    assert kw["balanced"] is False and kw["cap_samples"] is False
    assert len(kw["folds"]) == 1
    assert rec["n_train"] == 15 * 5
    assert rec["class_names"] == RQ2_BREEDS


def test_run_rq2_single_persists_and_cleans_checkpoint(tmp_path):
    dev_raw, _ = make_dev_raw(tmp_path)
    mapper = _mapper15()
    fake = make_fake_run_experiment([])
    sampled, _ = sample_nested_rq2(dev_raw, fold=0, seed=42)
    _, val_pos = rq2_fold_partitions(dev_raw, 0)
    ckpt_dir = tmp_path / "ckpt"
    stale = ckpt_dir / "rq2_fold0_seed42_N5"
    stale.mkdir(parents=True)

    rec = run_rq2_single(
        dev_raw, mapper, fold=0, seed=42, n=5,
        train_positions=sampled[5], val_positions=val_pos,
        out_dir=str(tmp_path / "out"), ckpt_dir=str(ckpt_dir),
        keep_checkpoints=False, run_experiment_fn=fake, accelerator="cpu",
    )
    out = tmp_path / "out"
    assert os.path.exists(out / "runs" / "fold0_seed42_N5.json")
    assert os.path.exists(out / "confusion_matrices" / "fold0_seed42_N5.json")
    assert os.path.exists(out / "per_class_recall" / "fold0_seed42_N5.json")
    assert not stale.exists()                       # temporary checkpoint deleted
    assert not rec["checkpoint_kept"]
    validate_run_record(rec)


def test_run_rq2_single_saves_pca_plot(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    dev_raw, _ = make_dev_raw(tmp_path)
    mapper = _mapper15()
    fake = make_fake_run_experiment([])
    sampled, _ = sample_nested_rq2(dev_raw, fold=0, seed=42)
    _, val_pos = rq2_fold_partitions(dev_raw, 0)

    # Simulate the PNG that run_experiment's PCA block writes before plt.show().
    src = tmp_path / "plot_vae_rq2_fold0_seed42_N5_kfold.png"
    src.write_bytes(b"png")

    rec = run_rq2_single(
        dev_raw, mapper, fold=0, seed=42, n=5,
        train_positions=sampled[5], val_positions=val_pos,
        out_dir=str(tmp_path / "out"), ckpt_dir=str(tmp_path / "ckpt"),
        run_experiment_fn=fake, accelerator="cpu",
    )
    dst = tmp_path / "out" / "pca" / "fold0_seed42_N5.png"
    assert dst.exists()                       # PCA preserved in the RQ2 tree
    assert not src.exists()                   # moved, not left in CWD
    assert rec["pca_plot"].endswith("pca/fold0_seed42_N5.png")


def test_suppress_plt_show_is_non_blocking():
    from vae.rq2_experiment import _suppress_plt_show

    original = plt.show
    with _suppress_plt_show():
        assert plt.show is not original
        plt.show()                            # must return immediately
    assert plt.show is original


def test_15_class_metrics_and_cm_shape():
    m = compute_classification_metrics(np.arange(15), np.arange(15), labels=np.arange(15))
    assert m["confusion_matrix"].shape == (15, 15)
    assert m["per_class_recall"].shape == (15,)
    assert m["labels"].tolist() == list(range(15))


# ---------------------------------------------------------------------------
# frozen run_experiment behavior (drop_last param + class-names fix)
# ---------------------------------------------------------------------------
def _patch_eval(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(vae_experiment, "compute_local_structure", lambda e, l: 1.0)
    monkeypatch.setattr(vae_experiment, "compute_generalization", lambda a, b, c, d: 1.0)
    monkeypatch.setattr(vae_experiment, "compute_neighbor_overlap",
                        lambda e, g, **k: {"NO_k3": 1.0, "NO_k10": 1.0, "NO_k30": 1.0})
    monkeypatch.setattr(vae_experiment, "silhouette_score", lambda e, l: 1.0)
    monkeypatch.setattr(vae_experiment, "davies_bouldin_score", lambda e, l: 0.5)
    monkeypatch.setattr(vae_experiment, "mean_squared_error", lambda a, b: 0.1)
    monkeypatch.setattr(vae_experiment, "cohen_kappa_score", lambda a, b: 0.9)
    monkeypatch.setattr(vae_experiment, "plt", _NoopPlt())


def _build_rq2_vfd(dev_raw, mapper, n=5):
    sampled, _ = sample_nested_rq2(dev_raw, fold=0, seed=42)
    _, val_pos = rq2_fold_partitions(dev_raw, 0)
    sample = build_rq2_sample(dev_raw, fold=0, seed=42, n=n,
                              train_positions=sampled[n], val_positions=val_pos, mapper=mapper)
    return build_rq2_vae_fold_data(sample)


def test_run_experiment_drop_last_param_and_class_names_fix(monkeypatch, tmp_path):
    dev_raw, _ = make_dev_raw(tmp_path)
    vfd = _build_rq2_vfd(dev_raw, _mapper15(), n=5)
    _patch_eval(monkeypatch, tmp_path)
    _FakeTrainer.instances = []
    monkeypatch.setattr("pytorch_lightning.Trainer", _FakeTrainer)

    res = vae_experiment.run_experiment(
        name="t", experiment_data=None, config=rq2_vae_config(accelerator="cpu"),
        max_epochs=2, classifier_config="breed_only", balanced=False,
        cap_samples=False, folds=[vfd], drop_last=False,
    )
    loader = _FakeTrainer.instances[-1].fit_loaders[0]
    assert loader.drop_last is False

    # class names come from the provided fold, not from experiment_data (None)
    assert res["Class Names Breed"] == RQ2_BREEDS
    assert res["Num Classes Breed"] == 15
    cm_csv = Path(tmp_path) / "results" / "confusion_matrices" / "cm_t.csv"
    assert cm_csv.exists()
    assert cm_csv.read_text().count("\n") - 1 == 15   # 15 rows


def test_run_experiment_default_drop_last_is_true(monkeypatch, tmp_path):
    dev_raw, _ = make_dev_raw(tmp_path)
    vfd = _build_rq2_vfd(dev_raw, _mapper15(), n=5)
    _patch_eval(monkeypatch, tmp_path)
    _FakeTrainer.instances = []
    monkeypatch.setattr("pytorch_lightning.Trainer", _FakeTrainer)

    vae_experiment.run_experiment(
        name="t2", experiment_data=None, config=rq2_vae_config(accelerator="cpu"),
        max_epochs=2, classifier_config="breed_only", balanced=False,
        cap_samples=False, folds=[vfd],  # drop_last default
    )
    loader = _FakeTrainer.instances[-1].fit_loaders[0]
    assert loader.drop_last is True


# ---------------------------------------------------------------------------
# aggregation + N_min
# ---------------------------------------------------------------------------
def _full_grid_records(macro_by_n):
    recs = []
    for n in RQ2_N_GRID:
        for fold in RQ2_FOLDS:
            for seed in RQ2_SEEDS:
                recs.append(make_record(fold, seed, n, macro_f1=macro_by_n[n]))
    return recs


def test_aggregate_requires_exactly_15_observations():
    recs = _full_grid_records({n: 0.9 for n in RQ2_N_GRID})
    recs = [r for r in recs if not (r["N"] == 5 and r["fold"] == 0 and r["replicate_seed"] == 42)]
    with pytest.raises(ValueError, match="expected exactly 15 observations"):
        aggregate_rq2(recs)


def test_aggregation_mean_std_and_n_min():
    macro = {5: 0.925, 10: 0.950, 15: 0.975, 20: 0.980, 25: 0.982, 30: 0.985}
    recs = _full_grid_records(macro)
    out = aggregate_rq2(recs)
    assert out["epsilon_N"] == 0.02
    assert out["S_ref"] == pytest.approx(0.985)
    threshold = 0.98 * 0.985
    assert out["threshold"] == pytest.approx(threshold)
    expected = min(n for n in RQ2_N_GRID if macro[n] >= threshold)
    assert out["N_min"] == expected                       # 15
    assert out["N_min"] in list(RQ2_N_GRID)
    for row in out["aggregates"]:
        assert row["n_observations"] == 15
        assert row["macro_f1_std"] == pytest.approx(0.0)


def test_invalid_n30_reference_raises():
    recs = _full_grid_records({n: 0.9 for n in RQ2_N_GRID})
    recs = [r for r in recs if r["N"] != 30]
    with pytest.raises(ValueError, match="expected exactly 15 observations"):
        aggregate_rq2(recs)

    recs = _full_grid_records({n: 0.9 for n in RQ2_N_GRID})
    for r in recs:
        if r["N"] == 30:
            r["macro_f1"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        aggregate_rq2(recs)


# ---------------------------------------------------------------------------
# resume / crash safety
# ---------------------------------------------------------------------------
def _run_small_grid(dev_raw, out_dir, *, fake, resume=True, force=False):
    return run_rq2_experiment(
        dev_raw, out_dir=str(out_dir), ckpt_dir=str(out_dir / "ckpt"),
        folds=[0], seeds=[42], ns=[5],
        resume=resume, force=force, run_experiment_fn=fake, accelerator="cpu",
    )


def test_resume_skips_valid_completed_runs(tmp_path):
    dev_raw, _ = make_dev_raw(tmp_path)
    out = tmp_path / "out"

    calls1 = []
    _run_small_grid(dev_raw, out, fake=make_fake_run_experiment(calls1))
    assert len(calls1) == 1

    calls2 = []
    _run_small_grid(dev_raw, out, fake=make_fake_run_experiment(calls2))
    assert len(calls2) == 0                       # skipped valid completed run

    calls3 = []
    _run_small_grid(dev_raw, out, fake=make_fake_run_experiment(calls3), force=True)
    assert len(calls3) == 1                       # forced re-run


def test_corrupt_or_partial_artifacts_are_not_accepted(tmp_path):
    out = tmp_path / "out"
    runs = out / "runs"
    runs.mkdir(parents=True)

    # invalid JSON
    (runs / "foldX.json").write_text("{not json")
    with pytest.raises(json.JSONDecodeError):
        load_completed_runs(str(out))
    (runs / "foldX.json").unlink()

    # valid record but missing convenience artifacts (partial write)
    rec = make_record(0, 42, 5)
    (runs / "fold0_seed42_N5.json").write_text(json.dumps(rec))
    with pytest.raises(FileNotFoundError, match="incomplete run artifact"):
        load_completed_runs(str(out))

    # invalid record (bad CM shape) -> rejected
    (out / "confusion_matrices").mkdir(exist_ok=True)
    (out / "per_class_recall").mkdir(exist_ok=True)
    rec_bad = make_record(0, 42, 5)
    rec_bad["confusion_matrix"] = [[1, 2], [3, 4]]
    (runs / "fold0_seed42_N5.json").write_text(json.dumps(rec_bad))
    with pytest.raises(ValueError, match="confusion_matrix shape"):
        load_completed_runs(str(out))
