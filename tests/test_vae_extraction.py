"""Behaviour-equivalence tests for the VAE helper extraction.

These verify that the extracted helpers preserve the original behaviour
without running full model training.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import matplotlib

matplotlib.use("Agg", force=True)

from vae import experiment as vae_experiment
from vae.attribution import BreedWrapper, aggregate_shap_to_global
from vae.plotting import plot_grid_heatmaps, plot_trends
from vae.results_io import load_cached_results, save_results_incrementally


# ---------------------------------------------------------------------------
# results_io
# ---------------------------------------------------------------------------
def test_results_io_roundtrip(tmp_path):
    path = tmp_path / "cache.json"
    results = [
        {"a": np.int64(3), "b": 1.5, "c": [np.float32(1.0), 2.0], "d": "x"},
        {"e": np.int32(7)},
    ]
    save_results_incrementally(results, path=str(path))
    loaded = load_cached_results(str(path))
    assert loaded == [
        {"a": 3, "b": 1.5, "c": [1.0, 2.0], "d": "x"},
        {"e": 7},
    ]


def test_load_cached_results_missing(tmp_path):
    assert load_cached_results(str(tmp_path / "missing.json")) == []


# ---------------------------------------------------------------------------
# attribution.aggregate_shap_to_global
# ---------------------------------------------------------------------------
def test_aggregate_shap_to_global_ndarray():
    arr = np.array([[1.0, -2.0, 3.0], [0.5, 1.0, -1.0]])
    out = aggregate_shap_to_global(arr, n_features=3)
    assert np.allclose(out, np.abs(arr).mean(axis=0))


def test_aggregate_shap_to_global_list():
    c0 = np.array([[1.0, -2.0, 3.0], [0.5, 1.0, -1.0]])
    c1 = np.array([[0.0, 1.0, 1.0], [2.0, -1.0, 0.0]])
    out = aggregate_shap_to_global([c0, c1], n_features=3)
    expected = np.abs(np.stack([c0, c1], axis=0)).mean(axis=(0, 1))
    assert np.allclose(out, expected)


def test_aggregate_shap_to_global_raises_when_no_feature_axis():
    arr = np.zeros((4, 4))
    with pytest.raises(ValueError):
        aggregate_shap_to_global(arr, n_features=3)


def test_breed_wrapper_matches_active_definition():
    import torch

    logits = torch.tensor([[0.1, 0.9, -0.3], [0.7, 0.2, 0.1]])

    class FakeSys:
        def __call__(self, x):
            return {"mu": x, "logits_breed": logits, "x_recon": x}

    x = torch.zeros(2, 5)
    wrapper = BreedWrapper(FakeSys())
    out = wrapper(x)
    # The active definition returned outputs['logits_breed'] verbatim.
    assert out is logits
    assert torch.equal(out, logits)


# ---------------------------------------------------------------------------
# plotting smoke tests (consume the same DataFrames as the original runs)
# ---------------------------------------------------------------------------
def test_plot_grid_heatmaps_smoke(tmp_path):
    df = pd.DataFrame({
        "Min_Samples": [30, 30, 40, 40],
        "Latent_Dim": [32, 64, 32, 64],
        "Acc_Breed (mean)": [0.91, 0.92, 0.93, 0.94],
    })
    out = tmp_path / "heat.png"
    plot_grid_heatmaps(df, {"Acc_Breed (mean)": "Breed Accuracy (raw)"}, save_path=str(out))
    assert out.exists()


def test_plot_trends_smoke(tmp_path):
    df = pd.DataFrame({
        "Min_Samples": [30, 40],
        "Latent_Dim": [32, 32],
        "Acc_Breed (mean)": [0.91, 0.93],
        "Kappa_Breed (mean)": [0.5, 0.6],
        "Generalization (GE) (mean)": [0.8, 0.82],
        "Silhouette (mean)": [0.7, 0.71],
    })
    out = tmp_path / "trends.png"
    plot_trends(df, [30, 40], [32], save_path=str(out))
    assert out.exists()


# ---------------------------------------------------------------------------
# run_grid / run_single invocation semantics (monkeypatched, no training)
# ---------------------------------------------------------------------------
class FakeRunExperiment:
    def __init__(self, calls):
        self.calls = calls

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return {"name": kwargs["name"]}


class FakeExperimentData:
    class_names = [f"B{i}" for i in range(34)]

    def __init__(self, n_folds=3):
        self.split = SimpleNamespace(n_folds=n_folds)


def make_config():
    return {
        "metadata_path": "x.csv",
        "batch_size": 64,
        "target_samples": 30,
        "alpha": 0.5,
        "lr": 0.0001,
        "latent_dim": 32,
        "weight_breed": 1.0,
        "weight_continent": 0.0,
        "weight_caseina": 0.0,
        "accelerator": "cpu",
        "devices": 1,
    }


def test_run_grid_invocation_semantics(monkeypatch):
    calls = []
    fake_folds = [object(), object(), object()]
    monkeypatch.setattr(vae_experiment, "_build_folds", lambda *a, **k: fake_folds)
    monkeypatch.setattr(vae_experiment, "run_experiment", FakeRunExperiment(calls))
    monkeypatch.setattr("vae.results_io.save_results_incrementally", lambda rl, path: None)

    config = make_config()
    grid_results, all_results = vae_experiment.run_grid(FakeExperimentData(), config, [32, 64])

    assert len(grid_results) == 2
    assert len(all_results) == 2
    assert {c["name"] for c in calls} == {"LatDim_32", "LatDim_64"}
    for call in calls:
        assert call["balanced"] is False
        assert call["cap_samples"] is False
        assert call["classifier_config"] == "breed_only"
        assert call["max_epochs"] == 200  # frozen PRIMARY RQ1 default
        assert call["folds"] is fake_folds
        assert call["config"]["latent_dim"] in (32, 64)
        assert call["name"] == f"LatDim_{call['config']['latent_dim']}"


def test_run_single_invocation_semantics(monkeypatch):
    calls = []
    fake_folds = [object(), object(), object()]
    monkeypatch.setattr(vae_experiment, "_build_folds", lambda *a, **k: fake_folds)
    monkeypatch.setattr(vae_experiment, "run_experiment", FakeRunExperiment(calls))

    config = make_config()
    single_result, exp_name = vae_experiment.run_single(FakeExperimentData(), config, 96)

    assert exp_name == "LatDim_96"
    assert single_result["Latent_Dim"] == 96
    assert len(calls) == 1

    call = calls[0]
    assert call["balanced"] is False
    assert call["cap_samples"] is False
    assert call["classifier_config"] == "breed_only"
    assert call["config"]["latent_dim"] == 96
    assert call["save_checkpoint_path"] == "checkpoints/LatDim_96.ckpt"
