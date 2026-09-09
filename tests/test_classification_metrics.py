"""Tests for the shared breed-classification metrics and the VAE/contrastive
classification evaluation paths."""

import numpy as np
import pytest
import torch

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.neighbors import KNeighborsClassifier

from genomic.classification_metrics import aggregate_folds, compute_classification_metrics
from contrastive_learning.evaluation import evaluate_knn_classification
from vae.network import VMGP_Network


# ---------------------------------------------------------------------------
# shared helper
# ---------------------------------------------------------------------------
def test_accuracy():
    y_true = [0, 1, 2, 2, 0]
    y_pred = [0, 1, 2, 1, 0]
    m = compute_classification_metrics(y_true, y_pred, labels=[0, 1, 2])
    assert m["accuracy"] == pytest.approx(accuracy_score(y_true, y_pred))


def test_macro_f1():
    y_true = [0, 1, 2, 2, 0]
    y_pred = [0, 1, 2, 1, 0]
    m = compute_classification_metrics(y_true, y_pred, labels=[0, 1, 2])
    assert m["macro_f1"] == pytest.approx(
        f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)
    )


def test_balanced_accuracy():
    y_true = [0, 1, 2, 2, 0]
    y_pred = [0, 1, 2, 1, 0]
    m = compute_classification_metrics(y_true, y_pred, labels=[0, 1, 2])
    assert m["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(y_true, y_pred))


def test_per_class_recall():
    y_true = [0, 1, 2, 2, 0]
    y_pred = [0, 1, 2, 1, 0]
    m = compute_classification_metrics(y_true, y_pred, labels=[0, 1, 2])
    expected = recall_score(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
    assert np.allclose(m["per_class_recall"], expected)


def test_confusion_matrix():
    y_true = [0, 1, 2, 2, 0]
    y_pred = [0, 1, 2, 1, 0]
    m = compute_classification_metrics(y_true, y_pred, labels=[0, 1, 2])
    assert np.array_equal(m["confusion_matrix"], confusion_matrix(y_true, y_pred, labels=[0, 1, 2]))


def test_canonical_ordering_and_full_vocab_dimensions():
    y_true = [2, 0, 2]
    y_pred = [2, 2, 0]
    m = compute_classification_metrics(y_true, y_pred, labels=[0, 1, 2])
    assert m["labels"].tolist() == [0, 1, 2]
    assert m["confusion_matrix"].shape == (3, 3)
    assert len(m["per_class_recall"]) == 3
    # class 1 is absent from both true and pred, but retains its row/column
    assert m["confusion_matrix"][1, :].sum() == 0
    assert m["confusion_matrix"][:, 1].sum() == 0


def test_zero_predicted_class_is_deterministic():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 0])  # class 1 never predicted
    m1 = compute_classification_metrics(y_true, y_pred, labels=[0, 1])
    m2 = compute_classification_metrics(y_true, y_pred, labels=[0, 1])
    assert m1["per_class_recall"][1] == 0.0
    assert m1["macro_f1"] == m2["macro_f1"]
    assert m1["balanced_accuracy"] == m2["balanced_accuracy"]
    assert np.array_equal(m1["confusion_matrix"], m2["confusion_matrix"])


def test_34_class_canonical_output_is_model_agnostic():
    y_true = np.array([0, 1, 2, 33, 0])
    y_pred = np.array([0, 1, 33, 33, 2])
    m = compute_classification_metrics(y_true, y_pred, labels=np.arange(34))
    assert m["confusion_matrix"].shape == (34, 34)
    assert m["per_class_recall"].shape == (34,)
    assert m["labels"].tolist() == list(range(34))


def test_aggregate_folds_mean_std_and_summed_cm():
    f1 = compute_classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], labels=[0, 1])
    f2 = compute_classification_metrics([0, 0, 1, 1], [0, 0, 1, 0], labels=[0, 1])
    agg = aggregate_folds([f1, f2])

    assert agg["accuracy_mean"] == pytest.approx((f1["accuracy"] + f2["accuracy"]) / 2)
    assert agg["macro_f1_mean"] == pytest.approx((f1["macro_f1"] + f2["macro_f1"]) / 2)
    assert agg["balanced_accuracy_mean"] == pytest.approx(
        (f1["balanced_accuracy"] + f2["balanced_accuracy"]) / 2
    )
    assert agg["per_class_recall_mean"].shape == (2,)
    assert agg["per_class_recall_std"].shape == (2,)
    assert np.array_equal(
        agg["confusion_matrix_sum"], f1["confusion_matrix"] + f2["confusion_matrix"]
    )


# ---------------------------------------------------------------------------
# VAE: metrics from breed-head logits
# ---------------------------------------------------------------------------
def test_vae_breed_head_metrics_match_sklearn():
    torch.manual_seed(0)
    model = VMGP_Network(num_snps=8, num_classes_breed=34, use_breed=True).eval()
    X = np.random.RandomState(1).randint(0, 3, size=(40, 8)).astype(np.float32)
    with torch.no_grad():
        logits = model(torch.FloatTensor(X))["logits_breed"]
    y_pred = logits.argmax(dim=1).numpy()
    y_true = np.array([i % 34 for i in range(40)])
    labels = np.arange(34)

    m = compute_classification_metrics(y_true, y_pred, labels=labels)

    assert m["macro_f1"] == pytest.approx(
        f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    )
    assert m["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(y_true, y_pred))
    assert m["accuracy"] == pytest.approx(accuracy_score(y_true, y_pred))


# ---------------------------------------------------------------------------
# contrastive: train -> validation KNN
# ---------------------------------------------------------------------------
def test_evaluate_knn_fits_train_only_predicts_val():
    rng = np.random.RandomState(0)
    Z_train = rng.randn(30, 3)
    y_train = np.array([0] * 15 + [1] * 15)
    Z_val = rng.randn(10, 3)
    y_val = np.array([0] * 5 + [1] * 5)
    labels = np.arange(2)

    metrics, y_pred = evaluate_knn_classification(
        Z_train, y_train, Z_val, y_val, k=3, labels=labels
    )

    ref = KNeighborsClassifier(n_neighbors=3).fit(Z_train, y_train)
    ref_pred = ref.predict(Z_val)

    # predictions are on validation samples only, using a train-only fit
    assert len(y_pred) == len(Z_val) == 10
    assert np.array_equal(y_pred, ref_pred)

    assert metrics["macro_f1"] == pytest.approx(
        f1_score(y_val, ref_pred, labels=labels, average="macro", zero_division=0)
    )
    assert metrics["balanced_accuracy"] == pytest.approx(
        balanced_accuracy_score(y_val, ref_pred)
    )
    assert metrics["accuracy"] == pytest.approx(accuracy_score(y_val, ref_pred))


def test_evaluate_knn_validation_never_in_fit_set():
    # A val sample that duplicates a train sample of a different class must be
    # classified by the train label (proving the fit set is train-only).
    Z_train = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]])
    y_train = np.array([0, 0, 1, 1])
    Z_val = np.array([[0.0, 0.0]])  # identical to a train sample of class 0
    y_val = np.array([1])  # but labeled differently

    _, y_pred = evaluate_knn_classification(
        Z_train, y_train, Z_val, y_val, k=1, labels=np.arange(2)
    )
    # If the val sample were in the fit set it would be its own nearest
    # neighbour (distance 0) and predict its own (wrong) label 1; with a
    # train-only fit it matches the identical train sample of class 0.
    assert y_pred.tolist() == [0]
