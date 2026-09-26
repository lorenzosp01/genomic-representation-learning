"""Tests for the protocol-v2 zero-leakage marker-ranking pipeline.

Covered:

* ``rank_fst`` / ``rank_random_forest`` output contracts (shape, permutation,
  descending order, determinism);
* zero-leakage contract: the fitted statistic/model consumes exactly the
  supplied training partition (spied call arguments), never held-out data;
* ``_stratified_sample_indices`` determinism and class coverage;
* model-based ranking methods on tiny toy models (no checkpoint loading);
* dispatcher routing and invalid-method error handling.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

import genomic.ranking as ranking
from genomic.ranking import (
    MarkerRankingResult,
    compute_marker_ranking,
    rank_chi2_association,
    rank_contrastive_ig,
    rank_contrastive_ig_mean,
    rank_contrastive_ig_probe,
    rank_fst,
    rank_random_forest,
    rank_vmgp_shap,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
def _synthetic(n_samples=90, n_features=12, n_classes=3, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randint(0, 3, size=(n_samples, n_features)).astype(np.float32)
    y = np.repeat(np.arange(n_classes), n_samples // n_classes)
    return X, y


def _assert_ranking_contract(result: MarkerRankingResult, method, n_features):
    assert isinstance(result, MarkerRankingResult)
    assert result.method == method
    assert result.scores.shape == (n_features,)
    assert result.ranked_indices.shape == (n_features,)
    assert np.isfinite(result.scores).all()

    assert sorted(result.ranked_indices.tolist()) == list(range(n_features))
    ordered = result.scores[result.ranked_indices]
    assert np.all(np.diff(ordered) <= 0.0)


# ---------------------------------------------------------------------------
# rank_fst
# ---------------------------------------------------------------------------
def test_rank_fst_output_contract():
    X, y = _synthetic(n_features=17)
    result = rank_fst(X, y, 3)
    _assert_ranking_contract(result, "fst", 17)
    assert ((result.scores >= 0.0) & (result.scores <= 1.0)).all()


def test_rank_fst_matches_weir_cockerham_fst():
    from genomic.fst import weir_cockerham_fst

    X, y = _synthetic()
    result = rank_fst(X, y, 3)
    assert np.allclose(result.scores, weir_cockerham_fst(X, y, 3))


def test_rank_fst_stable_tie_break():
    # Constant marker scores must keep ascending feature-index order.
    X = np.zeros((6, 5), dtype=np.float32)
    y = np.array([0, 0, 0, 1, 1, 1])
    result = rank_fst(X, y, 2)
    assert np.array_equal(result.ranked_indices, np.arange(5))


def test_rank_fst_rejects_invalid_inputs():
    X, y = _synthetic()
    with pytest.raises(ValueError):
        rank_fst(X, y[:-1], 3)
    with pytest.raises(ValueError):
        rank_fst(X.ravel(), y, 3)

    X_bad = X.copy()
    X_bad[0, 0] = 1.5
    with pytest.raises(ValueError):
        rank_fst(X_bad, y, 3)


def test_rank_fst_consumes_only_training_arrays(monkeypatch):
    X, y = _synthetic()
    seen = {}

    def spy(X_arg, y_arg, n_classes):
        seen["X"] = np.array(X_arg, copy=True)
        seen["y"] = np.array(y_arg, copy=True)
        seen["n_classes"] = n_classes
        return np.zeros(X_arg.shape[1], dtype=np.float64)

    monkeypatch.setattr(ranking, "weir_cockerham_fst", spy)
    result = rank_fst(X, y, 3)

    assert np.array_equal(seen["X"], X)
    assert np.array_equal(seen["y"], y)
    assert seen["X"].shape[0] == X.shape[0]
    assert seen["n_classes"] == 3
    assert result.scores.shape == (X.shape[1],)


# ---------------------------------------------------------------------------
# rank_random_forest
# ---------------------------------------------------------------------------
def test_rank_random_forest_output_contract():
    X, y = _synthetic(n_features=17)
    result = rank_random_forest(X, y, n_estimators=10, random_state=0)
    _assert_ranking_contract(result, "random_forest", 17)
    assert (result.scores >= 0.0).all()


def test_rank_random_forest_matches_direct_fit():
    from sklearn.ensemble import RandomForestClassifier

    X, y = _synthetic()
    result = rank_random_forest(X, y, n_estimators=10, random_state=0)
    direct = RandomForestClassifier(n_estimators=10, random_state=0, n_jobs=-1)
    direct.fit(X, y)
    assert np.allclose(result.scores, direct.feature_importances_)


def test_rank_random_forest_is_deterministic():
    X, y = _synthetic()
    a = rank_random_forest(X, y, n_estimators=10, random_state=7)
    b = rank_random_forest(X, y, n_estimators=10, random_state=7)
    assert np.array_equal(a.scores, b.scores)
    assert np.array_equal(a.ranked_indices, b.ranked_indices)


def test_rank_random_forest_consumes_only_training_arrays(monkeypatch):
    X, y = _synthetic()
    captured = {}

    class SpyForest:
        def __init__(self, **kwargs):
            captured["init_kwargs"] = kwargs

        def fit(self, X_fit, y_fit):
            captured["X_fit"] = np.array(X_fit, copy=True)
            captured["y_fit"] = np.array(y_fit, copy=True)
            return self

        @property
        def feature_importances_(self):
            return np.ones(captured["X_fit"].shape[1], dtype=np.float64)

    monkeypatch.setattr(ranking, "RandomForestClassifier", SpyForest)
    result = rank_random_forest(X, y, n_estimators=7, random_state=3, n_jobs=1)

    assert np.array_equal(captured["X_fit"], X)
    assert np.array_equal(captured["y_fit"], y)
    assert captured["X_fit"].shape[0] == X.shape[0]
    assert captured["init_kwargs"] == {
        "n_estimators": 7, "random_state": 3, "n_jobs": 1,
    }
    assert result.scores.shape == (X.shape[1],)


# ---------------------------------------------------------------------------
# rank_chi2_association (extended baseline)
# ---------------------------------------------------------------------------
def test_rank_chi2_association_output_contract():
    X, y = _synthetic(n_features=23)
    result = rank_chi2_association(X, y)
    _assert_ranking_contract(result, "chi2_association", 23)
    assert (result.scores >= 0.0).all()


def test_rank_chi2_association_ranks_associated_marker_first():
    rng = np.random.RandomState(3)
    n_samples = 300
    y = np.repeat([0, 1], n_samples // 2)
    X = rng.randint(0, 3, size=(n_samples, 6)).astype(np.float32)
    X[:, 4] = np.where(y == 0, 0.0, 2.0)
    result = rank_chi2_association(X, y, n_classes=2)
    assert result.ranked_indices[0] == 4
    # perfectly associated 2x2 table: chi-square equals the sample count
    assert result.scores[4] == pytest.approx(float(n_samples), rel=1e-6)


def test_rank_chi2_association_constant_marker_scores_zero():
    X, y = _synthetic()
    X = X.copy()
    X[:, 0] = 1.0
    result = rank_chi2_association(X, y, n_classes=3)
    assert result.scores[0] == 0.0


def test_rank_chi2_association_is_deterministic():
    X, y = _synthetic()
    a = rank_chi2_association(X, y, n_classes=3)
    b = rank_chi2_association(X, y, n_classes=3)
    assert np.array_equal(a.scores, b.scores)
    assert np.array_equal(a.ranked_indices, b.ranked_indices)


def test_rank_chi2_association_rejects_invalid_dosages():
    X, y = _synthetic()
    X = X.copy()
    X[0, 0] = 3.0
    with pytest.raises(ValueError, match="dosages"):
        rank_chi2_association(X, y, n_classes=3)


def test_dispatcher_routes_chi2_association():
    X, y = _synthetic()
    result = compute_marker_ranking("chi2_association", X, y, n_classes=3)
    _assert_ranking_contract(result, "chi2_association", X.shape[1])


# ---------------------------------------------------------------------------
# stratified subsampling (SHAP background/eval)
# ---------------------------------------------------------------------------
def test_stratified_sample_indices_deterministic_and_covering():
    _, y = _synthetic(n_samples=90, n_classes=3)
    idx_a = ranking._stratified_sample_indices(y, 9, random_state=42)
    idx_b = ranking._stratified_sample_indices(y, 9, random_state=42)

    assert np.array_equal(idx_a, idx_b)
    assert idx_a.shape == (9,)
    assert len(np.unique(idx_a)) == 9
    for cls in np.unique(y):
        assert np.isin(idx_a, np.flatnonzero(y == cls)).any()


def test_stratified_sample_indices_returns_all_when_n_exceeds_pool():
    _, y = _synthetic(n_samples=30, n_classes=3)
    idx = ranking._stratified_sample_indices(y, 100, random_state=0)
    assert np.array_equal(idx, np.arange(30))


# ---------------------------------------------------------------------------
# rank_contrastive_ig (toy encoder, no checkpoints)
# ---------------------------------------------------------------------------
def test_rank_contrastive_ig_output_contract():
    from contrastive_learning.model import ContrastiveGeneticModel

    X, y = _synthetic(n_samples=30, n_features=8, n_classes=3, seed=1)
    torch.manual_seed(0)
    model = ContrastiveGeneticModel(n_markers=8, embedding_dim=3)

    result = rank_contrastive_ig(
        model, X, y, n_steps=4, batch_size=8, device="cpu"
    )
    _assert_ranking_contract(result, "contrastive_ig", 8)
    assert (result.scores >= 0.0).all()


def test_ig_batched_explicit_zero_baseline_matches_default():
    from contrastive_learning.attribution import (
        EncoderWithProbeMLP,
        compute_ig_batched,
    )
    from contrastive_learning.model import ContrastiveGeneticModel

    X, y = _synthetic(n_samples=12, n_features=6, n_classes=2, seed=5)
    torch.manual_seed(0)
    model = ContrastiveGeneticModel(n_markers=6, embedding_dim=3)
    probe = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
    wrapper = EncoderWithProbeMLP(
        model.encoder, np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32),
        probe,
    )
    target = torch.as_tensor(y, dtype=torch.long)
    device = torch.device("cpu")
    default = compute_ig_batched(wrapper, X, target, device, n_steps=2, batch_size=4)
    explicit = compute_ig_batched(
        wrapper, X, target, device, n_steps=2, batch_size=4,
        baseline=torch.zeros(6, 4),
    )
    assert np.allclose(default, explicit)


def test_rank_contrastive_ig_mean_output_contract():
    from contrastive_learning.model import ContrastiveGeneticModel

    X, y = _synthetic(n_samples=30, n_features=8, n_classes=3, seed=1)
    torch.manual_seed(0)
    model = ContrastiveGeneticModel(n_markers=8, embedding_dim=3)

    result = rank_contrastive_ig_mean(
        model, X, y, n_steps=3, batch_size=8, device="cpu"
    )
    _assert_ranking_contract(result, "contrastive_ig_mean", 8)
    assert (result.scores >= 0.0).all()


def test_rank_contrastive_ig_probe_output_contract():
    from contrastive_learning.model import ContrastiveGeneticModel

    X, y = _synthetic(n_samples=30, n_features=8, n_classes=3, seed=1)
    torch.manual_seed(0)
    model = ContrastiveGeneticModel(n_markers=8, embedding_dim=3)

    result = rank_contrastive_ig_probe(
        model, X, y, n_steps=3, batch_size=8, device="cpu", probe_epochs=3
    )
    _assert_ranking_contract(result, "contrastive_ig_probe", 8)
    assert (result.scores >= 0.0).all()


def test_dispatcher_routes_contrastive_ig_probe():
    from contrastive_learning.model import ContrastiveGeneticModel

    X, y = _synthetic(n_samples=30, n_features=8, n_classes=3, seed=1)
    torch.manual_seed(0)
    model = ContrastiveGeneticModel(n_markers=8, embedding_dim=3)
    result = compute_marker_ranking(
        "contrastive_ig_probe", X, y, model=model,
        n_steps=2, batch_size=8, device="cpu", probe_epochs=2,
    )
    _assert_ranking_contract(result, "contrastive_ig_probe", 8)


# ---------------------------------------------------------------------------
# rank_vmgp_shap (toy breed head, no checkpoints)
# ---------------------------------------------------------------------------
class _ToyVMGP(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 8), nn.ReLU(), nn.Linear(8, n_classes)
        )

    def forward(self, x):
        return {"logits_breed": self.net(x)}


def test_rank_vmgp_shap_output_contract():
    X, y = _synthetic(n_samples=30, n_features=10, n_classes=3, seed=2)

    torch.manual_seed(0)
    model = _ToyVMGP(10, 3)
    result = rank_vmgp_shap(
        model, X, y, n_background=6, n_eval=6, random_state=42
    )
    _assert_ranking_contract(result, "vmgp_shap", 10)


def test_rank_vmgp_shap_is_deterministic():
    X, y = _synthetic(n_samples=30, n_features=10, n_classes=3, seed=2)

    torch.manual_seed(0)
    model = _ToyVMGP(10, 3)
    a = rank_vmgp_shap(model, X, y, n_background=6, n_eval=6, random_state=42)
    b = rank_vmgp_shap(model, X, y, n_background=6, n_eval=6, random_state=42)
    assert np.allclose(a.scores, b.scores, atol=1e-7)
    assert np.array_equal(a.ranked_indices, b.ranked_indices)


# ---------------------------------------------------------------------------
# dispatcher
# ---------------------------------------------------------------------------
def test_dispatcher_routes_fst():
    X, y = _synthetic()
    result = compute_marker_ranking("fst", X, y, n_classes=3)
    _assert_ranking_contract(result, "fst", X.shape[1])


def test_dispatcher_routes_random_forest():
    X, y = _synthetic()
    result = compute_marker_ranking(
        "random_forest", X, y, n_estimators=10, random_state=0
    )
    _assert_ranking_contract(result, "random_forest", X.shape[1])


def test_dispatcher_routes_contrastive_ig():
    from contrastive_learning.model import ContrastiveGeneticModel

    X, y = _synthetic(n_samples=30, n_features=8, n_classes=3, seed=1)
    torch.manual_seed(0)
    model = ContrastiveGeneticModel(n_markers=8, embedding_dim=3)
    result = compute_marker_ranking(
        "contrastive_ig", X, y, model=model, n_steps=4, batch_size=8
    )
    _assert_ranking_contract(result, "contrastive_ig", 8)


def test_dispatcher_unknown_method_raises():
    X, y = _synthetic()
    with pytest.raises(ValueError, match="Unknown ranking method"):
        compute_marker_ranking("not_a_method", X, y)


def test_dispatcher_model_method_requires_model():
    X, y = _synthetic()
    with pytest.raises(TypeError, match="model"):
        compute_marker_ranking("vmgp_shap", X, y)
