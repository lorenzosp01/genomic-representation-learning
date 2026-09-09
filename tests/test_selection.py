"""Tests for the RQ1 selection policy, checkpoint best-epoch reporting, and
primary-RQ1 training-distribution fairness."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np
import pytest
import torch

from genomic.selection import best_epoch_from_checkpoint_path, select_best_configuration
from genomic.splitting import build_split
from genomic.experiment_data import GenomicExperimentData

import vae.experiment as vae_exp
from vae.experiment import _balance_training


# ---------------------------------------------------------------------------
# selection policy
# ---------------------------------------------------------------------------
def _res(macro_f1, bal_acc=0.0, acc=0.0, **extra):
    r = {"Macro_F1 (mean)": macro_f1, "Balanced_Accuracy (mean)": bal_acc,
         "Accuracy (mean)": acc}
    r.update(extra)
    return r


def test_select_by_macro_f1():
    results = [_res(0.5, 0.9, 0.9, name="a"), _res(0.7, 0.1, 0.1, name="b")]
    i, r = select_best_configuration(results)
    assert i == 1 and r["name"] == "b"


def test_tiebreak_balanced_accuracy_then_accuracy():
    results = [
        _res(0.7, 0.5, 0.9, name="a"),
        _res(0.7, 0.8, 0.1, name="b"),
        _res(0.7, 0.8, 0.9, name="c"),
    ]
    i, r = select_best_configuration(results)
    assert i == 2 and r["name"] == "c"


def test_deterministic_order_resolves_full_ties():
    results = [_res(0.7, 0.8, 0.9, name="a"), _res(0.7, 0.8, 0.9, name="b")]
    i, _ = select_best_configuration(results)
    assert i == 0  # first in list wins


def test_selection_ignores_legacy_fields():
    # A config with higher GE/Kappa/self-consistency but lower Macro-F1 must lose.
    results = [
        _res(0.6, 0.6, 0.6, name="low_mf",
             **{"Best Fold GE (legacy)": 0.99, "Kappa_Breed (mean)": 0.99,
                "Self_Consistency_knn_acc_k3 (mean)": 0.99}),
        _res(0.8, 0.5, 0.5, name="high_mf",
             **{"Best Fold GE (legacy)": 0.1, "Kappa_Breed (mean)": 0.1,
                "Self_Consistency_knn_acc_k3 (mean)": 0.1}),
    ]
    i, r = select_best_configuration(results)
    assert i == 1 and r["name"] == "high_mf"


def test_selection_empty_raises():
    with pytest.raises(ValueError):
        select_best_configuration([])


# ---------------------------------------------------------------------------
# best epoch / median
# ---------------------------------------------------------------------------
def test_best_epoch_from_checkpoint_path_is_one_based():
    assert best_epoch_from_checkpoint_path("checkpoints/x/fold_1/epoch=0000-val_loss=1.0.ckpt") == 1
    assert best_epoch_from_checkpoint_path("checkpoints/x/fold_1/epoch=0007-val_loss=0.5.ckpt") == 8


def test_best_epoch_from_checkpoint_path_invalid():
    with pytest.raises(ValueError):
        best_epoch_from_checkpoint_path("no_epoch_here.ckpt")


# ---------------------------------------------------------------------------
# primary-RQ1 training distribution (no cap / no SMOTE)
# ---------------------------------------------------------------------------
def test_balance_training_disabled_returns_inputs_unchanged():
    X = np.zeros((10, 4), dtype=np.float32)
    yb = np.arange(10)
    yc = ycas = yatt = np.full(10, -1)
    out = _balance_training(X, yb, yc, ycas, yatt, cap_samples=False, balanced=False, target_samples=3)
    assert out[0] is X
    assert out[1] is yb
    assert out[2] is yc


# ---------------------------------------------------------------------------
# VAE: primary RQ1 train membership equals FoldData (no capping)
# ---------------------------------------------------------------------------
def _write_fam(path, breeds):
    lines = []
    for b, iids in breeds.items():
        for iid in iids:
            lines.append(f"{b}\t{iid}\t0\t0\t0\t-9\n")
    with open(path, "w") as f:
        f.writelines(lines)


def _synthetic_ed(tmp_path):
    fam = tmp_path / "s.fam"
    _write_fam(fam, {
        "ALP": [f"ALP_{i}" for i in range(6)],
        "BOR": [f"BOR_{i}" for i in range(6)],
        "CEN": [f"CEN_{i}" for i in range(6)],
    })
    split = build_split(str(fam), dataset_id="s", k_folds=3, cohort_min_breed_size=None)
    rng = np.random.RandomState(0)
    X = rng.randint(0, 3, size=(split.meta["n_eligible"], 8)).astype(np.float32)
    return GenomicExperimentData(split, X, preprocessor_kwargs={"ld_window": 1, "maf_threshold": 0.0})


def test_vae_primary_rq1_fold_membership_equals_folddata(tmp_path):
    ed = _synthetic_ed(tmp_path)
    folds = vae_exp._build_folds(ed, {}, True, False, False, False, None)
    for vfd in folds:
        fd = ed.fold_data(vfd.fold)
        assert np.array_equal(vfd.train_source_index, fd.train_source_index)
        # With cap/balanced disabled, the training set is the full fold training set.
        assert len(vfd.X_train) == len(fd.X_train)


# ---------------------------------------------------------------------------
# VAE: best-val_loss checkpoint creation + restoration
# ---------------------------------------------------------------------------
class _FakeLightning:
    def __init__(self, **kwargs):
        self.device = torch.device("cpu")
        self.is_loaded = False

    @classmethod
    def load_from_checkpoint(cls, path):
        m = cls.__new__(cls)
        m.device = torch.device("cpu")
        m.is_loaded = True
        return m

    def __call__(self, x):
        return {
            "mu": x.new_zeros(x.shape[0], 3),
            "x_recon": x,
            "logits_breed": x.new_zeros(x.shape[0], 3),
        }

    def eval(self):
        return self

    def freeze(self):
        return self


def test_vae_restores_best_val_loss_checkpoint(monkeypatch, tmp_path):
    ed = _synthetic_ed(tmp_path)
    loaded_paths = []
    fitted = []

    def fake_init(self, **kwargs):
        fitted.append(kwargs)
        self.device = torch.device("cpu")
        self.is_loaded = False

    def fake_load(cls, path):
        loaded_paths.append(path)
        m = cls.__new__(cls)
        m.device = torch.device("cpu")
        m.is_loaded = True
        return m

    monkeypatch.setattr(vae_exp, "VMGP_LightningSystem", _FakeLightning)
    monkeypatch.setattr(_FakeLightning, "__init__", fake_init)
    monkeypatch.setattr(_FakeLightning, "load_from_checkpoint", classmethod(fake_load))

    class FakeCheckpoint:
        best_model_path = "checkpoints/t/fold_1/epoch=0004-val_loss=0.5.ckpt"

        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, model, tr_dl, val_dl):
            self.model = model

    monkeypatch.setattr(vae_exp, "ModelCheckpoint", FakeCheckpoint)
    monkeypatch.setattr("pytorch_lightning.Trainer", FakeTrainer)
    monkeypatch.setattr(vae_exp, "compute_local_structure", lambda e, l: 1.0)
    monkeypatch.setattr(vae_exp, "compute_generalization", lambda et, lt, ev, lv: 1.0)
    monkeypatch.setattr(
        vae_exp, "compute_neighbor_overlap",
        lambda e, g, **kw: {"NO_k3": 1.0, "NO_k10": 1.0, "NO_k30": 1.0},
    )
    monkeypatch.setattr(vae_exp, "silhouette_score", lambda e, l: 1.0)
    monkeypatch.setattr(vae_exp, "davies_bouldin_score", lambda e, l: 0.5)
    monkeypatch.setattr(vae_exp, "mean_squared_error", lambda a, b: 0.1)
    monkeypatch.setattr(vae_exp, "cohen_kappa_score", lambda a, b: 0.9)

    config = {
        "batch_size": 4, "alpha": 0.5, "lr": 1e-4, "latent_dim": 8,
        "weight_breed": 1.0, "weight_continent": 0.0, "weight_caseina": 0.0,
        "accelerator": "cpu", "devices": 1,
    }

    res = vae_exp.run_experiment("t", ed, config, max_epochs=2, classifier_config="breed_only")

    # ModelCheckpoint created once per fold with the frozen val_loss monitor.
    assert len(fitted) == 3  # one model constructed per fold
    # Best-val_loss checkpoint is restored before evaluation (once per fold).
    assert loaded_paths == ["checkpoints/t/fold_1/epoch=0004-val_loss=0.5.ckpt"] * 3
    assert res["Best_Epochs"] == [5, 5, 5]  # epoch=0004 -> 1-based 5
    assert res["Median_Best_Epoch"] == 5.0

