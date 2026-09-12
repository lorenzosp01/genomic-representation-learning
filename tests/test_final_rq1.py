"""Tests for the PRIMARY RQ1 final-development training + locked-test path.

No real final training or real locked-test access happens in these tests:
the training functions are exercised with synthetic data and a fake Trainer.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import RandomSampler

from genomic.classification_metrics import compute_classification_metrics
from genomic.experiment_data import GenomicExperimentData, locked_test_overlap
from genomic.preprocessing import GenomicPreprocessor
from genomic.splitting import build_split, load_split

import vae.final as vae_final
import contrastive_learning.final as ctr_final

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def write_fam(path, breeds):
    lines = []
    for b, iids in breeds.items():
        for iid in iids:
            lines.append(f"{b}\t{iid}\t0\t0\t0\t-9\n")
    with open(path, "w") as f:
        f.writelines(lines)


def synthetic_ed(tmp_path, n_snps=8, seed=0):
    fam = tmp_path / "s.fam"
    write_fam(fam, {
        "ALP": [f"ALP_{i}" for i in range(6)],
        "BOR": [f"BOR_{i}" for i in range(6)],
        "CEN": [f"CEN_{i}" for i in range(6)],
    })
    split = build_split(str(fam), dataset_id="s", k_folds=3, cohort_min_breed_size=None)
    rng = np.random.RandomState(seed)
    X = rng.randint(0, 3, size=(split.meta["n_eligible"], n_snps)).astype(np.float32)
    return GenomicExperimentData(
        split, X, preprocessor_kwargs={"ld_window": 1, "maf_threshold": 0.0}
    )


def vae_config():
    return {"batch_size": 64, "alpha": 0.5, "lr": 1e-4, "weight_breed": 1.0,
            "weight_continent": 0.0, "weight_caseina": 0.0,
            "accelerator": "cpu", "devices": 1}


def ctr_config():
    return {"embedding_dim": 3, "flip_max": 0.99, "mask_max": 0.99,
            "learning_rate": 0.001, "lr_decay_factor": 0.99, "lr_decay_interval": 10,
            "accelerator": "cpu", "devices": 1}


class _FakeTrainer:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.fit_loaders = None
        self.saved = None
        _FakeTrainer.instances.append(self)

    def fit(self, model, *loaders):
        self.fit_loaders = loaders

    def save_checkpoint(self, path):
        self.saved = path


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "run_rq1_final", REPO / "scripts" / "run_rq1_final.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 1-2. sizes / frozen source indices (real data)
# ---------------------------------------------------------------------------
FAM = str(REPO / "data" / "ADAPTmap_genotypeTOP_20160222_full.fam")
BED = str(REPO / "data" / "ADAPTmap_genotypeTOP_20160222_full.bed")
BIM = str(REPO / "data" / "ADAPTmap_genotypeTOP_20160222_full.bim")
OUT = str(REPO / "splits")
DS = "ADAPTmap_genotypeTOP_20160222_full"


@pytest.mark.skipif(not (Path(FAM).exists() and Path(BED).exists()), reason="data missing")
def test_real_final_dev_test_sizes_and_indices():
    split = load_split(DS, 42, 42, FAM, out_dir=OUT, cohort_label="indmiss0p10")

    # Frozen split metadata (no genotype access): sizes, classes, disjointness.
    assert len(split.development_indices()) == 2437
    assert len(split.test_indices()) == 426
    assert len(set(split.breed.tolist())) == 34
    assert set(split.development_indices().tolist()).isdisjoint(set(split.test_indices().tolist()))

    # Development-only preparation: preprocessor fit uses dev rows only.
    ed = GenomicExperimentData.from_plink(split, BED, bim_path=BIM)
    dev = ed.build_final_development_data()
    assert dev.n_samples == 2437
    assert dev.n_classes == 34
    assert np.array_equal(dev.dev_source_index, split.development_indices())
    assert locked_test_overlap(dev.dev_source_index, split) == 0
    # NOTE: `transform_locked_test` is intentionally NOT called on real data here;
    # the locked-test transform path is exercised on synthetic data only.


# ---------------------------------------------------------------------------
# 3. preprocessing fit receives development rows only
# ---------------------------------------------------------------------------
def test_fit_receives_development_rows_only(monkeypatch, tmp_path):
    ed = synthetic_ed(tmp_path)
    captured = {}
    orig_fit = GenomicPreprocessor.fit

    def spy_fit(self, X, snp_metadata=None):
        captured["n"] = int(np.asarray(X).shape[0])
        captured["X"] = np.array(X)
        return orig_fit(self, X, snp_metadata=snp_metadata)

    monkeypatch.setattr(GenomicPreprocessor, "fit", spy_fit)
    dev = ed.build_final_development_data()

    dev_pos = np.flatnonzero(ed.split.outer_split == "development")
    assert captured["n"] == len(dev_pos) == dev.n_samples
    assert np.array_equal(captured["X"], ed._X[dev_pos])


# ---------------------------------------------------------------------------
# 4. locked-test-only missingness cannot affect the fit
# ---------------------------------------------------------------------------
def test_locked_test_missingness_cannot_affect_fit(tmp_path):
    fam = tmp_path / "s.fam"
    write_fam(fam, {
        "ALP": [f"ALP_{i}" for i in range(6)],
        "BOR": [f"BOR_{i}" for i in range(6)],
        "CEN": [f"CEN_{i}" for i in range(6)],
    })
    split = build_split(str(fam), dataset_id="s", k_folds=3, cohort_min_breed_size=None)
    rng = np.random.RandomState(0)
    X = rng.randint(0, 3, size=(split.meta["n_eligible"], 4)).astype(np.float32)

    test_mask = split.outer_split == "test"
    dev_idx = np.flatnonzero(~test_mask)

    # SNP 0: missingness exists ONLY in locked test -> must be retained.
    X[test_mask, 0] = np.nan
    # SNP 1: >10% missingness in development -> must be removed.
    X[dev_idx[:2], 1] = np.nan

    ed = GenomicExperimentData(split, X)  # default frozen thresholds
    dev = ed.build_final_development_data()
    retained = set(dev.retained_snp_indices.tolist())
    assert 0 in retained       # test-only missingness did not influence the fit
    assert 1 not in retained   # development missingness did


# ---------------------------------------------------------------------------
# 5-6. dev construction does not touch test; locked-test transform is explicit
# ---------------------------------------------------------------------------
def test_build_dev_transforms_dev_only(monkeypatch, tmp_path):
    ed = synthetic_ed(tmp_path)
    shapes = []
    orig = GenomicPreprocessor.transform

    def spy(self, X):
        shapes.append(int(np.asarray(X).shape[0]))
        return orig(self, X)

    monkeypatch.setattr(GenomicPreprocessor, "transform", spy)
    dev = ed.build_final_development_data()

    assert shapes == [dev.n_samples]           # one transform, dev rows only
    assert not hasattr(dev, "X_test")          # no locked-test data produced


def test_locked_test_transform_is_explicit_separate_call(tmp_path):
    ed = synthetic_ed(tmp_path)
    dev = ed.build_final_development_data()
    test = ed.transform_locked_test(dev.preprocessor)

    n_test = int((ed.split.outer_split == "test").sum())
    assert test.n_samples == n_test
    assert test.preprocessor is dev.preprocessor
    assert set(test.test_source_index.tolist()).isdisjoint(set(dev.dev_source_index.tolist()))


# ---------------------------------------------------------------------------
# 7-9. final trainers: no validation, no callbacks, exact epochs, loader semantics
# ---------------------------------------------------------------------------
def test_frozen_final_constants():
    assert vae_final.FINAL_LATENT_DIM == 96
    assert vae_final.FINAL_EPOCHS == 168
    assert vae_final.FINAL_PRECISION == "16-mixed"
    assert ctr_final.FINAL_EMBEDDING_DIM == 3
    assert ctr_final.FINAL_EPOCHS == 1216


def test_vae_final_trainer_no_val_no_callbacks_168(monkeypatch, tmp_path):
    ed = synthetic_ed(tmp_path)
    dev = ed.build_final_development_data()
    _FakeTrainer.instances = []
    monkeypatch.setattr("pytorch_lightning.Trainer", _FakeTrainer)

    ckpt = str(tmp_path / "vae.ckpt")
    out = vae_final.train_vae_final(dev, vae_config(), epochs=168, checkpoint_path=ckpt)

    t = _FakeTrainer.instances[-1]
    assert t.kwargs["callbacks"] == []
    assert t.kwargs["enable_checkpointing"] is False
    assert t.kwargs["max_epochs"] == 168 == t.kwargs["min_epochs"]
    assert t.kwargs["num_sanity_val_steps"] == 0
    assert t.kwargs["precision"] == "16-mixed"
    assert len(t.fit_loaders) == 1  # train loader only, no validation loader
    assert out["epochs"] == 168
    assert t.saved == ckpt


def test_contrastive_final_trainer_no_val_no_callbacks_1216(monkeypatch, tmp_path):
    ed = synthetic_ed(tmp_path)
    dev = ed.build_final_development_data()
    _FakeTrainer.instances = []
    monkeypatch.setattr("pytorch_lightning.Trainer", _FakeTrainer)

    ckpt = str(tmp_path / "ctr.ckpt")
    out = ctr_final.train_contrastive_final(dev, ctr_config(), epochs=1216, checkpoint_path=ckpt)

    t = _FakeTrainer.instances[-1]
    assert t.kwargs["callbacks"] == []
    assert t.kwargs["enable_checkpointing"] is False
    assert t.kwargs["max_epochs"] == 1216 == t.kwargs["min_epochs"]
    assert t.kwargs["num_sanity_val_steps"] == 0
    assert t.kwargs["precision"] == "32-true"
    assert len(t.fit_loaders) == 1
    assert out["epochs"] == 1216
    assert t.saved == ckpt


def test_final_dataloader_behavior_preserved(monkeypatch, tmp_path):
    ed = synthetic_ed(tmp_path)
    dev = ed.build_final_development_data()
    _FakeTrainer.instances = []
    monkeypatch.setattr("pytorch_lightning.Trainer", _FakeTrainer)

    vae_final.train_vae_final(dev, vae_config(), epochs=1, checkpoint_path=str(tmp_path / "v.ckpt"))
    lv = _FakeTrainer.instances[-1].fit_loaders[0]
    assert lv.batch_size == 64
    assert lv.drop_last is True                 # VAE CV training semantics
    assert isinstance(lv.sampler, RandomSampler)
    assert lv.num_workers == 4 and lv.pin_memory is True

    ctr_final.train_contrastive_final(dev, ctr_config(), epochs=1, checkpoint_path=str(tmp_path / "c.ckpt"))
    lc = _FakeTrainer.instances[-1].fit_loaders[0]
    assert lc.batch_size == min(dev.n_samples, 512)   # contrastive CV semantics
    assert lc.drop_last is False
    assert isinstance(lc.sampler, RandomSampler)
    assert lc.num_workers == 4 and lc.pin_memory is True


# ---------------------------------------------------------------------------
# 10. contrastive KNN fits development embeddings/labels only
# ---------------------------------------------------------------------------
def test_contrastive_knn_fits_dev_only(monkeypatch, tmp_path):
    ed = synthetic_ed(tmp_path)
    dev = ed.build_final_development_data()
    test = ed.transform_locked_test(dev.preprocessor)

    captured = {}
    monkeypatch.setattr(
        ctr_final, "extract_embeddings",
        lambda model, X, batch_size=512: np.zeros((len(X), 3), dtype=np.float32),
    )

    def fake_knn(Z_train, y_train, Z_val, y_val, *, k=3, labels=None):
        captured.update(n_train=len(Z_train), n_val=len(Z_val), k=k,
                        labels=np.asarray(labels))
        return (
            {"accuracy": 1.0, "macro_f1": 1.0, "balanced_accuracy": 1.0,
             "per_class_recall": np.ones(len(labels)), "labels": np.asarray(labels),
             "confusion_matrix": np.eye(len(labels), dtype=np.int64)},
            np.zeros(len(y_val), dtype=np.int64),
        )

    monkeypatch.setattr(ctr_final, "evaluate_knn_classification", fake_knn)

    trained = {"model": object(), "embedding_dim": 3, "epochs": 1216,
               "checkpoint_path": "x", "runtime_s": 1.0}
    res = ctr_final.evaluate_contrastive_final(trained, dev, test, output_dir=str(tmp_path))

    assert captured["n_train"] == dev.n_samples
    assert captured["n_val"] == test.n_samples
    assert captured["k"] == 3
    assert captured["labels"].tolist() == list(range(dev.n_classes))
    assert res["n_train_embeddings"] == dev.n_samples
    assert res["n_test"] == test.n_samples


# ---------------------------------------------------------------------------
# 11. canonical 34-class order / 34x34 confusion matrix
# ---------------------------------------------------------------------------
def test_metric_canonical_order_34():
    y_true = np.array([0, 1, 2, 33])
    y_pred = np.array([0, 1, 33, 33])
    m = compute_classification_metrics(y_true, y_pred, labels=np.arange(34))
    assert m["confusion_matrix"].shape == (34, 34)
    assert m["per_class_recall"].shape == (34,)
    assert m["labels"].tolist() == list(range(34))


# ---------------------------------------------------------------------------
# 12. shared preprocessor built once; locked-test transform after all training
# ---------------------------------------------------------------------------
def test_shared_preprocessor_built_once_and_eval_after_training(monkeypatch, tmp_path):
    mod = _load_script()
    ed = synthetic_ed(tmp_path)
    events = []
    devs = []

    orig_build = GenomicExperimentData.build_final_development_data

    def counting_build(self, **kw):
        events.append("build_dev")
        return orig_build(self, **kw)

    monkeypatch.setattr(GenomicExperimentData, "build_final_development_data", counting_build)

    def fake_train_vae(dev, config, **kw):
        events.append("train_vae")
        devs.append(dev)
        return {"epochs": 168, "latent_dim": 96, "runtime_s": 0.0}

    def fake_train_ctr(dev, config, **kw):
        events.append("train_ctr")
        devs.append(dev)
        return {"epochs": 1216, "embedding_dim": 3, "runtime_s": 0.0}

    monkeypatch.setattr(mod, "train_vae_final", fake_train_vae)
    monkeypatch.setattr(mod, "train_contrastive_final", fake_train_ctr)
    monkeypatch.setattr(mod, "evaluate_vae_final",
                        lambda trained, test, **kw: events.append("eval_vae") or {"a": 1})
    monkeypatch.setattr(mod, "evaluate_contrastive_final",
                        lambda trained, dev, test, **kw: events.append("eval_ctr") or {"a": 1})

    orig_transform = GenomicExperimentData.transform_locked_test

    def transform_spy(self, pre):
        events.append("transform_test")
        return orig_transform(self, pre)

    monkeypatch.setattr(GenomicExperimentData, "transform_locked_test", transform_spy)

    mod.run_final_with_data("both", ed, out_dir=str(tmp_path / "out"), ckpt_dir=str(tmp_path / "ckpt"))

    assert events.count("build_dev") == 1
    assert devs[0] is devs[1]  # same shared development data/preprocessor
    assert events.index("train_vae") < events.index("transform_test")
    assert events.index("train_ctr") < events.index("transform_test")
    assert events.index("transform_test") < events.index("eval_vae")
    assert events.index("transform_test") < events.index("eval_ctr")


# ---------------------------------------------------------------------------
# 13. final artifact paths cannot overwrite CV artifacts
# ---------------------------------------------------------------------------
def test_final_artifact_paths_distinct_from_cv():
    for p in (vae_final.DEFAULT_CHECKPOINT, ctr_final.DEFAULT_CHECKPOINT):
        assert "rq1_final" in p
        assert "LatDim" not in p and "rq1_fold" not in p and "pilot" not in p
    assert vae_final.DEFAULT_CHECKPOINT != ctr_final.DEFAULT_CHECKPOINT
    assert "rq1_final" in vae_final.DEFAULT_OUTPUT_DIR
    assert "rq1_final" in ctr_final.DEFAULT_OUTPUT_DIR
