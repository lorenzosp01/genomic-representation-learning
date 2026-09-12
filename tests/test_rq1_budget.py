"""Tests for the frozen PRIMARY RQ1 training budgets."""

import inspect
from pathlib import Path

import pytest

import vae.experiment as vae_exp
import contrastive_learning.experiment as contrastive_exp
from contrastive_learning import model as contrastive_model
from contrastive_learning import loss as contrastive_loss
from vae import lightning_module as vae_lm

REPO = Path(__file__).resolve().parents[1]


def _src(path):
    return (REPO / path).read_text()


# ---------------------------------------------------------------------------
# VAE primary RQ1 budget
# ---------------------------------------------------------------------------
def test_vae_run_experiment_default_max_epochs_is_200():
    sig = inspect.signature(vae_exp.run_experiment)
    assert sig.parameters["max_epochs"].default == 200


def test_vae_no_competing_hardcoded_defaults():
    src = _src("vae/experiment.py")
    # legacy production default removed
    assert "max_epochs: int = 50" not in src
    # grid/single fall back to the frozen 200 (no hardcoded competing value)
    assert "config.get('max_epochs', 200)" in src


def test_vae_patience_remains_15():
    src = _src("vae/experiment.py")
    assert "patience=15" in src


def test_vae_lr_is_config_driven(monkeypatch):
    calls = []

    def fake_run_experiment(**kwargs):
        calls.append(kwargs)
        return {"Best_Epochs": [1]}

    monkeypatch.setattr(vae_exp, "run_experiment", fake_run_experiment)
    monkeypatch.setattr(vae_exp, "_build_folds", lambda *a, **k: [object()])
    config = {"latent_dim": 96, "batch_size": 64, "alpha": 0.5, "lr": 1e-4,
              "weight_breed": 1.0, "weight_continent": 0.0, "weight_caseina": 0.0,
              "accelerator": "cpu", "devices": 1}
    vae_exp.run_single(object(), config, 96)
    # lr is passed through from config (not a hardcoded budget value)
    assert calls[0]["config"]["lr"] == 1e-4


# ---------------------------------------------------------------------------
# contrastive production budget
# ---------------------------------------------------------------------------
def test_contrastive_default_max_epochs_is_5000():
    sig = inspect.signature(contrastive_exp.run_experiment)
    assert sig.parameters["max_epochs"].default == 5000


def test_contrastive_patience_remains_200():
    assert "patience=200" in _src("contrastive_learning/experiment.py")


def test_contrastive_steplr_unchanged():
    sig = inspect.signature(contrastive_model.ContrastiveGeneticModel.__init__)
    assert sig.parameters["learning_rate"].default == 0.001
    assert sig.parameters["lr_decay_interval"].default == 10
    assert sig.parameters["lr_decay_factor"].default == 0.99
    src = inspect.getsource(contrastive_model.ContrastiveGeneticModel.configure_optimizers)
    assert "StepLR" in src


# ---------------------------------------------------------------------------
# pilot ceilings remain caller-supplied diagnostics (no production mutation)
# ---------------------------------------------------------------------------
def test_pilot_ceilings_do_not_change_production_defaults():
    assert inspect.signature(vae_exp.run_vae_pilot).parameters["max_epochs"].default == 100
    assert inspect.signature(contrastive_exp.run_contrastive_pilot).parameters["max_epochs"].default == 1500
    # production defaults are independent of the pilot defaults
    assert inspect.signature(vae_exp.run_experiment).parameters["max_epochs"].default == 200
    assert inspect.signature(contrastive_exp.run_experiment).parameters["max_epochs"].default == 5000


# ---------------------------------------------------------------------------
# scientific model/loss unchanged (guards)
# ---------------------------------------------------------------------------
def test_vmcp_loss_weights_unchanged():
    src = _src("vae/lightning_module.py")
    assert "0.0001 * kl_loss" in src       # frozen KL coefficient
    assert "self.alpha * loss_rec" in src  # frozen alpha weighting


def test_contrastive_loss_class_unchanged():
    assert "class CentroidNPairLoss" in _src("contrastive_learning/loss.py")
