"""Equivalence tests for the contrastive core extraction.

These verify that the classes/functions moved out of the contrastive notebooks
preserve the exact notebook behaviour, without running full model training.
"""

import numpy as np
import pytest
import torch

from contrastive_learning.augmentation import GeneticAugmentation
from contrastive_learning.encoder import GeneticEncoder
from contrastive_learning.loss import CentroidNPairLoss
from contrastive_learning.evaluation import compute_metrics, equal_earth_projection, extract_embeddings
from contrastive_learning import experiment as contrastive_experiment


# ---------------------------------------------------------------------------
# GeneticAugmentation
# ---------------------------------------------------------------------------
def test_one_hot_encode_exact_mapping():
    snps = torch.tensor([[0, 1, 2, -1]])
    oh = GeneticAugmentation.one_hot_encode(snps)
    expected = torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]).unsqueeze(0).float()
    assert torch.equal(oh, expected)


def test_augmentation_eval_is_pure_one_hot():
    aug = GeneticAugmentation()
    snps = torch.tensor([[0, 1, 2, -1]])
    assert torch.equal(aug(snps, training=False), GeneticAugmentation.one_hot_encode(snps))


def test_augmentation_deterministic_given_seed():
    aug = GeneticAugmentation()
    snps = torch.tensor([[0, 1, 2, 0], [1, 1, 2, 2]])
    torch.manual_seed(0)
    out1 = aug(snps, training=True)
    torch.manual_seed(0)
    out2 = aug(snps, training=True)
    assert torch.equal(out1, out2)
    assert out1.shape == (2, 4, 4)


# ---------------------------------------------------------------------------
# GeneticEncoder
# ---------------------------------------------------------------------------
def test_encoder_deterministic_and_unit_norm():
    enc = GeneticEncoder(n_markers=4, embedding_dim=3).eval()
    x = torch.rand(2, 4, 4)
    z1 = enc(x)
    z2 = enc(x)
    assert torch.equal(z1, z2)
    norms = z1.norm(p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


# ---------------------------------------------------------------------------
# CentroidNPairLoss
# ---------------------------------------------------------------------------
def test_centroid_loss_matches_reference():
    B, N_neg, D = 3, 2, 4
    torch.manual_seed(1)
    anchor = torch.randn(B, D)
    positive = torch.randn(B, D)
    negatives = torch.randn(B, N_neg, D)

    loss = CentroidNPairLoss()(anchor, positive, negatives)

    # Independent recomputation of Eq. (3)-(6).
    e2 = float(np.exp(-2))
    z = anchor.unsqueeze(1).expand(-1, N_neg, -1)
    zp = positive.unsqueeze(1).expand(-1, N_neg, -1)
    zn = negatives
    C = (z + 2.0 * zn + zp) / 4.0
    z_c, zp_c, zn_c = z - C, zp - C, zn - C
    mu = torch.max(torch.stack([(z_c**2).sum(-1), (zp_c**2).sum(-1), (zn_c**2).sum(-1)]), dim=0).values
    d = torch.sqrt(mu.unsqueeze(-1) + 1e-8)
    z_t, zp_t, zn_t = z_c / d, zp_c / d, zn_c / d
    term = torch.exp((z_t * zn_t).sum(-1) - (z_t * zp_t).sum(-1)) - e2
    expected = torch.log(1.0 + term.sum(dim=1).clamp(min=0.0)).mean()

    assert loss.ndim == 0
    assert torch.allclose(loss, expected)


def test_centroid_loss_deterministic():
    B, N_neg, D = 4, 3, 3
    torch.manual_seed(2)
    anchor = torch.randn(B, D)
    positive = torch.randn(B, D)
    negatives = torch.randn(B, N_neg, D)
    l1 = CentroidNPairLoss()(anchor, positive, negatives)
    l2 = CentroidNPairLoss()(anchor, positive, negatives)
    assert torch.equal(l1, l2)


# ---------------------------------------------------------------------------
# evaluation helpers
# ---------------------------------------------------------------------------
def test_equal_earth_projection_known_point():
    xyz = np.array([[1.0, 0.0, 0.0]])
    out = equal_earth_projection(xyz)
    assert out.shape == (1, 2)
    assert np.allclose(out, np.array([[0.0, 0.0]]), atol=1e-6)


def test_compute_metrics_separable_fixture():
    Z = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1],
                  [10.0, 10.0], [10.1, 10.0], [10.0, 10.1]])
    y = np.array([0, 0, 0, 1, 1, 1])
    m = compute_metrics(Z, y, k=3)
    assert m["knn_acc_k3"] == 1.0
    assert m["knn_f1_k3"] == 1.0
    assert set(m) == {"knn_acc_k3", "knn_f1_k3", "silhouette", "davies_bouldin"}


def test_extract_embeddings():
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, x, augment=False):
            return x.sum(dim=-1, keepdim=True) * self.w

    model = DummyModel().eval()
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    Z = extract_embeddings(model, X, batch_size=2)
    expected = np.array([[3.0], [7.0], [11.0]], dtype=np.float32)
    assert Z.shape == (3, 1)
    assert np.allclose(Z, expected)


# ---------------------------------------------------------------------------
# run_experiment invocation / config semantics (no training)
# ---------------------------------------------------------------------------
class FakeDataModule:
    def __init__(self):
        self.X_processed = np.random.RandomState(0).rand(9, 10).astype(np.float32)
        self.y_processed = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        self.num_snps = 10


class FakeModel:
    calls = []

    def __init__(self, **kwargs):
        FakeModel.calls.append(kwargs)

    @classmethod
    def load_from_checkpoint(cls, path):
        # Bypass __init__ so the checkpoint load is not recorded in cls.calls.
        return cls.__new__(cls)


class FakeTrainer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, model, tr_dl, val_dl):
        self.model = model


def test_run_experiment_model_config_semantics(monkeypatch):
    FakeModel.calls = []
    monkeypatch.setattr(contrastive_experiment, "ContrastiveGeneticModel", FakeModel)
    monkeypatch.setattr("pytorch_lightning.Trainer", FakeTrainer)
    monkeypatch.setattr(
        contrastive_experiment, "extract_embeddings",
        lambda model, X, batch_size=512: np.zeros((len(X), 3), dtype=np.float32),
    )
    monkeypatch.setattr(
        contrastive_experiment, "compute_metrics",
        lambda Z, y, k=3: {"knn_acc_k3": 1.0, "knn_f1_k3": 1.0, "silhouette": 0.5, "davies_bouldin": 1.0},
    )

    config = {
        "embedding_dim": 3,
        "flip_max": 0.99,
        "mask_max": 0.99,
        "learning_rate": 0.001,
        "lr_decay_factor": 0.99,
        "lr_decay_interval": 10,
        "accelerator": "cpu",
        "devices": 1,
    }
    dm = FakeDataModule()
    result = contrastive_experiment.run_experiment("t", dm, config, max_epochs=2, k_folds=3)

    assert len(FakeModel.calls) == 3  # one per fold
    for kw in FakeModel.calls:
        assert kw["n_markers"] == dm.num_snps
        assert kw["embedding_dim"] == config["embedding_dim"]
        assert kw["flip_max"] == config["flip_max"]
        assert kw["mask_max"] == config["mask_max"]
        assert kw["learning_rate"] == config["learning_rate"]
        assert kw["lr_decay_factor"] == config["lr_decay_factor"]
        assert kw["lr_decay_interval"] == config["lr_decay_interval"]

    assert result["Experiment"] == "t"
    assert result["knn_acc_k3 (mean)"] == 1.0
    assert result["_best_Z"] is not None
