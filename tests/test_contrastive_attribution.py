"""Equivalence tests for the contrastive attribution extraction.

These verify that ``contrastive_learning/attribution.py`` reproduces the exact
notebook behaviour on small deterministic fixtures (no full training).
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

import contrastive_learning.attribution as att
from contrastive_learning.attribution import (
    EncoderToCentroid,
    EncoderWithProbeMLP,
    aggregate_attributions,
    compute_breed_centroids,
    compute_centroid_ig_batched,
    compute_centroid_ig_single,
    compute_ig_batched,
    panel_accuracy_curve,
    select_best_checkpoint,
    train_probe_mlp,
)
from contrastive_learning.augmentation import GeneticAugmentation


DEVICE = torch.device("cpu")


def _build_probe_model(M, C, seed):
    torch.manual_seed(seed)
    encoder = nn.Sequential(nn.Flatten(), nn.Linear(M * 4, 3))
    probe = nn.Linear(3, C)
    sc_mean = np.zeros(3, dtype=np.float32)
    sc_std = np.ones(3, dtype=np.float32)
    return EncoderWithProbeMLP(encoder, sc_mean, sc_std, probe).eval()


def _build_centroid_model(M, C, seed):
    torch.manual_seed(seed)
    encoder = nn.Sequential(nn.Flatten(), nn.Linear(M * 4, 3))
    centroids = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]][:C]).float()
    return EncoderToCentroid(encoder, centroids).eval()


# ---------------------------------------------------------------------------
# 1. EncoderWithProbeMLP
# ---------------------------------------------------------------------------
def test_encoder_with_probe_mlp():
    torch.manual_seed(0)
    encoder = nn.Linear(4, 3)
    probe = nn.Linear(3, 2)
    sc_mean = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    sc_std = np.array([2.0, 1.0, 4.0], dtype=np.float32)
    wrapper = EncoderWithProbeMLP(encoder, sc_mean, sc_std, probe).eval()

    x = torch.randn(2, 4)
    with torch.no_grad():
        out = wrapper(x)
        z = encoder(x)
        z_scaled = (z - torch.FloatTensor(sc_mean)) / (torch.FloatTensor(sc_std) + 1e-8)
        expected = probe(z_scaled)
    assert torch.allclose(out, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# 2. EncoderToCentroid
# ---------------------------------------------------------------------------
def test_encoder_to_centroid():
    torch.manual_seed(1)
    encoder = nn.Linear(4, 3)
    centroids = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    wrapper = EncoderToCentroid(encoder, centroids).eval()

    x = torch.randn(1, 4)
    target = 0
    with torch.no_grad():
        z = encoder(x)
        expected = (z * centroids[target].unsqueeze(0)).sum(dim=1)
        out = wrapper(x, target)
    assert torch.allclose(out, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# 3. compute_ig_batched (probe IG) vs reference
# ---------------------------------------------------------------------------
def _reference_ig_batched(model, X_snps, target_classes, device, baseline_val=0.0, n_steps=10):
    N, M = X_snps.shape
    ig = np.zeros((N, M), dtype=np.float32)
    for i in range(N):
        x = torch.FloatTensor(X_snps[i:i + 1]).to(device)
        with torch.no_grad():
            x_oh = GeneticAugmentation.one_hot_encode(x).float()
        baseline = torch.full_like(x_oh, baseline_val)
        diff = x_oh - baseline
        alphas = torch.linspace(0.0, 1.0, n_steps + 1, device=device)
        grads = []
        for a in alphas:
            inp = (baseline + a * diff).requires_grad_(True)
            out = model(inp)
            score = out[0, target_classes[i]]
            g = torch.autograd.grad(score, inp)[0]
            grads.append(g)
        gs = torch.stack(grads)
        avg = (gs[0] + gs[-1]) / 2.0 + gs[1:-1].sum(0)
        avg /= n_steps
        ig_map = diff * avg
        ig[i] = ig_map.abs().sum(dim=-1).cpu().numpy()
    return ig


def test_compute_ig_batched_matches_reference():
    M, C = 4, 2
    model = _build_probe_model(M, C, seed=2)
    X = np.random.RandomState(0).randint(0, 3, size=(2, M)).astype(np.float32)
    y = np.array([0, 1])
    tc = torch.LongTensor(y).to(DEVICE)

    impl = compute_ig_batched(model, X, tc, DEVICE, n_steps=10)
    ref = _reference_ig_batched(model, X, tc, DEVICE, n_steps=10)
    assert np.allclose(impl, ref, atol=1e-5)


# ---------------------------------------------------------------------------
# 4. compute_centroid_ig_batched vs reference
# ---------------------------------------------------------------------------
def _reference_centroid_ig(model, X_snps, target_classes, device, n_steps=10):
    N, M = X_snps.shape
    ig = np.zeros((N, M), dtype=np.float32)
    for i in range(N):
        x = torch.FloatTensor(X_snps[i:i + 1]).to(device)
        with torch.no_grad():
            x_oh = GeneticAugmentation.one_hot_encode(x).float()
        baseline = torch.zeros_like(x_oh)
        diff = x_oh - baseline
        alphas = torch.linspace(0.0, 1.0, n_steps + 1, device=device)
        grads = []
        for a in alphas:
            inp = (baseline + a * diff).requires_grad_(True)
            z = model.encoder(inp)
            c = model.centroids[target_classes[i]].unsqueeze(0)
            score = (z * c).sum()
            g = torch.autograd.grad(score, inp)[0]
            grads.append(g)
        gs = torch.stack(grads)
        avg = (gs[0] + gs[-1]) / 2.0 + gs[1:-1].sum(0)
        avg /= n_steps
        ig_map = diff * avg
        ig[i] = ig_map.abs().sum(dim=-1).cpu().numpy()
    return ig


def test_compute_centroid_ig_batched_matches_reference():
    M, C = 4, 2
    model = _build_centroid_model(M, C, seed=3)
    X = np.random.RandomState(1).randint(0, 3, size=(2, M)).astype(np.float32)
    y = np.array([1, 0])
    tc = torch.LongTensor(y).to(DEVICE)

    impl = compute_centroid_ig_batched(model, X, tc, DEVICE, n_steps=10)
    ref = _reference_centroid_ig(model, X, tc, DEVICE, n_steps=10)
    assert np.allclose(impl, ref, atol=1e-5)


# ---------------------------------------------------------------------------
# 5. compute_centroid_ig_single == batched (single sample) + convergence delta
# ---------------------------------------------------------------------------
def test_centroid_ig_single_matches_batched():
    M, C = 4, 2
    model = _build_centroid_model(M, C, seed=4)
    X = np.random.RandomState(2).randint(0, 3, size=(1, M)).astype(np.float32)
    y = np.array([1])

    ig_batched = compute_centroid_ig_batched(model, X, torch.LongTensor(y).to(DEVICE), DEVICE, n_steps=10)

    with torch.no_grad():
        x_oh = GeneticAugmentation.one_hot_encode(torch.FloatTensor(X).to(DEVICE)).float()
    ig_single, delta = compute_centroid_ig_single(model, x_oh, int(y[0]), DEVICE, n_steps=10)

    assert np.allclose(ig_single, ig_batched[0], atol=1e-5)
    assert delta < 0.05


# ---------------------------------------------------------------------------
# 6. compute_breed_centroids
# ---------------------------------------------------------------------------
def test_compute_breed_centroids():
    Z = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    y = np.array([0, 0, 1])
    centroids = compute_breed_centroids(Z, y, 2)
    expected = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    assert np.allclose(centroids, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# 7. aggregate_attributions (per-breed equal weight, NOT sample-weighted)
# ---------------------------------------------------------------------------
def test_aggregate_attributions_equal_weight():
    # breed 0: 3 samples, attribution [10, 0]; breed 1: 1 sample, [2, 0]
    ig = np.array([[10.0, 0.0], [10.0, 0.0], [10.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    y = np.array([0, 0, 0, 1])
    gmean, gstd = aggregate_attributions(ig, y, 2)
    # equal-weight: (10 + 2)/2 = 6; sample-weighted would be 8
    assert np.allclose(gmean, np.array([6.0, 0.0]), atol=1e-6)
    assert np.allclose(gstd, np.array([4.0, 0.0]), atol=1e-6)


# ---------------------------------------------------------------------------
# 8. panel_accuracy_curve
# ---------------------------------------------------------------------------
def test_panel_accuracy_curve():
    rng = np.random.RandomState(0)
    X = rng.rand(30, 8)
    y = np.array([0] * 10 + [1] * 10 + [2] * 10)
    ig_scores = rng.rand(8)

    df, acc_full = panel_accuracy_curve(ig_scores, X, y, percentiles=[1.0, 5.0])

    assert len(df) == 2
    assert set(acc_full) == {"KNN", "LR", "RF"}

    # top-k selection matches a direct recomputation
    for _, row in df.iterrows():
        thr = np.percentile(ig_scores, 100 - row["percentile"])
        assert row["n_snp"] == int((ig_scores >= thr).sum())

    for name in ["KNN", "LR", "RF"]:
        assert f"acc_{name}_ig" in df.columns
        assert f"acc_{name}_rand_mean" in df.columns
        assert f"acc_{name}_rand_std" in df.columns

    # acc_full matches a direct cross_val_score
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    Xs = StandardScaler().fit_transform(X)
    direct = cross_val_score(KNeighborsClassifier(n_neighbors=3), Xs, y, cv=skf,
                             scoring="accuracy").mean()
    assert abs(acc_full["KNN"] - direct) < 1e-9

    # determinism
    df2, acc_full2 = panel_accuracy_curve(ig_scores, X, y, percentiles=[1.0, 5.0])
    assert np.allclose(df.values, df2.values)
    assert acc_full == acc_full2


# ---------------------------------------------------------------------------
# 9. train_probe_mlp
# ---------------------------------------------------------------------------
def test_train_probe_mlp():
    rng = np.random.RandomState(0)
    best_Z = rng.randn(60, 3).astype(np.float32)
    best_y = np.array([0] * 30 + [1] * 30)

    torch.manual_seed(0)
    probe, scaler, acc = train_probe_mlp(best_Z, best_y, 2, DEVICE, epochs=5)

    assert isinstance(probe, nn.Sequential)
    assert probe[0].in_features == 3
    assert probe[0].out_features == 64
    assert probe[-1].out_features == 2
    assert 0.0 <= acc <= 1.0
    assert hasattr(scaler, "mean_") and hasattr(scaler, "scale_")

    torch.manual_seed(0)
    probe2, scaler2, acc2 = train_probe_mlp(best_Z, best_y, 2, DEVICE, epochs=5)
    assert acc == acc2
    assert np.allclose(scaler.mean_, scaler2.mean_)


# ---------------------------------------------------------------------------
# 10. select_best_checkpoint
# ---------------------------------------------------------------------------
def test_select_best_checkpoint(monkeypatch):
    checkpoints = [
        "checkpoints/Original/fold_1/epoch=0001-val_loss=1.0.ckpt",
        "checkpoints/Original/fold_2/epoch=0001-val_loss=1.0.ckpt",
        "checkpoints/Original/fold_3/epoch=0001-val_loss=1.0.ckpt",
    ]
    acc_by_ckpt = {checkpoints[0]: 0.9, checkpoints[1]: 0.95, checkpoints[2]: 0.8}

    monkeypatch.setattr(att.glob, "glob", lambda pattern: list(checkpoints))

    class FakeModel:
        def __init__(self, ckpt):
            self.ckpt = ckpt

        @classmethod
        def load_from_checkpoint(cls, ckpt):
            return cls(ckpt)

    monkeypatch.setattr(att, "ContrastiveGeneticModel", FakeModel)
    monkeypatch.setattr(
        att, "extract_embeddings",
        lambda model, X, batch_size=512: np.array([acc_by_ckpt[model.ckpt]]),
    )
    monkeypatch.setattr(
        att, "compute_metrics",
        lambda Z, y, k=3: {"knn_acc_k3": float(Z[0])},
    )

    class DummyDM:
        X_processed = np.zeros((1, 1))
        y_processed = np.zeros(1)

    best_ckpt, best_acc, best_fold_idx = att.select_best_checkpoint(
        DummyDM(), "checkpoints/Original/fold_*/*.ckpt"
    )

    assert best_ckpt == checkpoints[1]
    assert best_acc == 0.95
    assert best_fold_idx == 1
