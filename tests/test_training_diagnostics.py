"""Tests for the convergence-pilot instrumentation (history callback,
best-epoch hardening, and ceiling classification)."""

import torch
import numpy as np
import pytest

from genomic.selection import best_epoch_from_checkpoint_path
from genomic.training_diagnostics import (
    ConvergenceHistoryCallback,
    classify_ceiling,
    summarize_convergence,
    tail_slope,
)
from genomic.splitting import build_split
from genomic.experiment_data import GenomicExperimentData

import vae.experiment as vae_exp
import contrastive_learning.experiment as contrastive_exp


# ---------------------------------------------------------------------------
# fake trainer plumbing for callback tests
# ---------------------------------------------------------------------------
class _FakeOpt:
    def __init__(self, lr):
        self.param_groups = [{"lr": lr}]


class _FakeES:
    def __init__(self, stopped_epoch):
        self.stopped_epoch = stopped_epoch


class _FakeTrainer:
    def __init__(self, current_epoch, max_epochs, *, train_loss=None, val_loss=None,
                 lr=0.1, stopped_epoch=0):
        self.current_epoch = current_epoch
        self.max_epochs = max_epochs
        self.sanity_checking = False
        self.callback_metrics = {}
        self.logged_metrics = {}
        if train_loss is not None:
            self.callback_metrics["train_loss"] = torch.tensor(train_loss)
        if val_loss is not None:
            self.callback_metrics["val_loss"] = torch.tensor(val_loss)
        self.optimizers = [_FakeOpt(lr)]
        self.early_stopping_callback = _FakeES(stopped_epoch)


# ---------------------------------------------------------------------------
# history callback
# ---------------------------------------------------------------------------
def test_callback_records_epoch_metrics():
    cb = ConvergenceHistoryCallback()
    tr = _FakeTrainer(0, 10, train_loss=1.0, val_loss=0.9, lr=0.1)
    cb.on_train_epoch_start(tr, None)
    cb.on_train_epoch_end(tr, None)
    cb.on_validation_epoch_end(tr, None)

    assert cb.history[0]["epoch"] == 0
    assert cb.history[0]["train_loss"] == pytest.approx(1.0)
    assert cb.history[0]["val_loss"] == pytest.approx(0.9)
    assert cb.history[0]["learning_rate"] == pytest.approx(0.1)


def test_callback_skips_sanity_checking():
    cb = ConvergenceHistoryCallback()
    tr = _FakeTrainer(0, 10, train_loss=1.0, val_loss=0.9, lr=0.1)
    tr.sanity_checking = True
    cb.on_train_epoch_start(tr, None)
    cb.on_train_epoch_end(tr, None)
    cb.on_validation_epoch_end(tr, None)
    assert cb.history == []


def test_callback_reports_early_stopping_triggered():
    cb = ConvergenceHistoryCallback()
    tr = _FakeTrainer(6, 100, stopped_epoch=6, lr=0.1)  # ES fired at 0-based epoch 6
    for e in range(7):  # 7 completed epochs
        tr.current_epoch = e
        cb.on_train_epoch_start(tr, None)
    cb.on_train_end(tr, None)
    assert cb.early_stopping_triggered is True
    assert cb.stopped_epoch == 7  # history records 7 completed epochs


def test_callback_reports_no_early_stopping_when_ceiling_reached():
    cb = ConvergenceHistoryCallback()
    tr = _FakeTrainer(99, 100, stopped_epoch=0, lr=0.1)  # natural completion
    for e in range(100):  # 100 completed epochs
        tr.current_epoch = e
        cb.on_train_epoch_start(tr, None)
    cb.on_train_end(tr, None)
    assert cb.early_stopping_triggered is False
    assert cb.stopped_epoch == 100  # history records 100 completed epochs


def test_callback_uses_es_state_not_current_epoch():
    # Training stopped early for a non-EarlyStopping reason: 6 epochs completed
    # but the EarlyStopping callback never fired (stopped_epoch == 0). Must NOT
    # be reported as EarlyStopping-triggered, and stopped_epoch must reflect the
    # actual completed-epoch count, not max_epochs.
    cb = ConvergenceHistoryCallback()
    tr = _FakeTrainer(5, 100, stopped_epoch=0, lr=0.1)
    for e in range(6):  # 6 completed epochs
        tr.current_epoch = e
        cb.on_train_epoch_start(tr, None)
    cb.on_train_end(tr, None)
    assert cb.early_stopping_triggered is False
    assert cb.stopped_epoch == 6


# ---------------------------------------------------------------------------
# hardened best_epoch (checkpoint metadata)
# ---------------------------------------------------------------------------
def test_best_epoch_from_checkpoint_metadata(tmp_path):
    p = tmp_path / "epoch=0000-val_loss=1.0.ckpt"  # filename misleadingly says 0
    torch.save({"epoch": 99, "state_dict": {}}, str(p))
    assert best_epoch_from_checkpoint_path(str(p)) == 100  # metadata 99 + 1

    p2 = tmp_path / "epoch=0050-val_loss=1.0.ckpt"
    torch.save({"epoch": 0}, str(p2))
    assert best_epoch_from_checkpoint_path(str(p2)) == 1  # metadata 0 + 1


def test_best_epoch_fallback_to_filename():
    # Non-existent path -> filename fallback.
    assert best_epoch_from_checkpoint_path("nonexistent/epoch=0007-val_loss=1.0.ckpt") == 8


# ---------------------------------------------------------------------------
# summary / ceiling diagnostics
# ---------------------------------------------------------------------------
def test_summarize_epochs_since_best_and_ratio():
    history = [
        {"epoch": 0, "train_loss": 1.0, "val_loss": 0.9, "learning_rate": 0.1},
        {"epoch": 1, "train_loss": 0.9, "val_loss": 0.8, "learning_rate": 0.1},
        {"epoch": 2, "train_loss": 0.8, "val_loss": 0.7, "learning_rate": 0.1},
    ]
    s = summarize_convergence(history, best_epoch=2, stopped_epoch=None,
                              max_epochs=10, early_stopping_triggered=False)
    assert s["best_epoch_ratio"] == pytest.approx(0.2)
    assert s["epochs_since_best"] == 8
    assert s["lr_at_best_epoch"] == pytest.approx(0.1)
    assert s["tail_slope"] < 0


def test_tail_slope_deterministic_and_signed():
    vals = [0.9, 0.8, 0.7, 0.6, 0.5]
    assert tail_slope(vals) == pytest.approx(tail_slope(vals))
    assert tail_slope(vals) < 0.0
    assert tail_slope([0.5] * 5) == pytest.approx(0.0, abs=1e-9)
    assert tail_slope([]) is None


def test_classify_ceiling_distinguishes_states():
    assert classify_ceiling(True, 0.9, -0.1) == "CONVERGED"
    assert classify_ceiling(False, 0.9, -0.1) == "CEILING_SUSPICIOUS"
    assert classify_ceiling(False, 0.5, -0.1) == "UNCERTAIN"
    assert classify_ceiling(False, 0.9, 0.0) == "UNCERTAIN"
    assert classify_ceiling(False, None, None) == "UNCERTAIN"


# ---------------------------------------------------------------------------
# pilot entry points (no training; run_experiment monkeypatched)
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


def test_vae_pilot_uses_pilot_ceiling_and_rq1(monkeypatch, tmp_path):
    ed = _synthetic_ed(tmp_path)
    captured = {}

    def fake_run_experiment(**kwargs):
        captured.update(kwargs)
        return {"Best_Epochs": [5]}

    monkeypatch.setattr(vae_exp, "run_experiment", fake_run_experiment)

    config = {"latent_dim": 32, "alpha": 0.5, "lr": 1e-4, "batch_size": 4,
              "weight_breed": 1.0, "weight_continent": 0.0, "weight_caseina": 0.0,
              "accelerator": "cpu", "devices": 1}
    config_copy = dict(config)

    vae_exp.run_vae_pilot(ed, config, latent_dim=96, fold=0, max_epochs=100,
                          output_dir=str(tmp_path / "pilot"))

    assert captured["max_epochs"] == 100
    assert captured["balanced"] is False
    assert captured["cap_samples"] is False
    assert captured["classifier_config"] == "breed_only"
    assert captured["history_callback"] is not None
    assert len(captured["folds"]) == 1
    assert captured["folds"][0].fold == 0
    # pilot fold membership == normal FoldData membership (no test rows)
    fd = ed.fold_data(0)
    assert np.array_equal(captured["folds"][0].train_source_index, fd.train_source_index)
    # production config not mutated
    assert config == config_copy
    assert config["latent_dim"] == 32


def test_contrastive_pilot_uses_pilot_ceiling(monkeypatch, tmp_path):
    ed = _synthetic_ed(tmp_path)
    captured = {}

    def fake_run_experiment(**kwargs):
        captured.update(kwargs)
        return {"Best_Epochs": [7]}

    monkeypatch.setattr(contrastive_exp, "run_experiment", fake_run_experiment)

    config = {"embedding_dim": 3, "flip_max": 0.99, "mask_max": 0.99,
              "learning_rate": 0.001, "lr_decay_factor": 0.99, "lr_decay_interval": 10,
              "accelerator": "cpu", "devices": 1}
    config_copy = dict(config)

    contrastive_exp.run_contrastive_pilot(ed, config, fold=0, max_epochs=1500,
                                          output_dir=str(tmp_path / "pilot"))

    assert captured["max_epochs"] == 1500
    assert captured["folds"] == [0]
    assert captured["history_callback"] is not None
    assert config == config_copy
