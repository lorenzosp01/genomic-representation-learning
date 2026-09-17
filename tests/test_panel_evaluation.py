"""Tests for the RQ3/RQ4 downstream panel evaluation and run_rq3 orchestration.

Covered:

* ``evaluate_panel`` output contract, exact column subsetting, frozen RF
  hyperparameters, reproducibility and input validation;
* canonical label set usage (34-class primary cohort);
* ``evaluate_full_panel`` equivalence to an explicit all-columns call;
* per-(method, K) aggregation (mean / sample std, full-panel excluded);
* ``minimum_panel_size`` / ``retention_threshold`` criterion;
* a tiny synthetic end-to-end smoke run of ``scripts/run_rq3.py`` with
  ``K=[10, 20]`` verifying artifacts, ranking manifests and cache reuse.
"""

import json
from pathlib import Path

import numpy as np
import pytest

import genomic.panel_evaluation as pe
from genomic.panel_evaluation import (
    aggregate_panel_records,
    evaluate_full_panel,
    evaluate_panel,
    minimum_panel_size,
    retention_threshold,
)

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
def _data(n_samples=90, n_features=12, n_classes=3, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randint(0, 3, size=(n_samples, n_features)).astype(np.float32)
    y = np.repeat(np.arange(n_classes), n_samples // n_classes)
    return X, y


# ---------------------------------------------------------------------------
# evaluate_panel
# ---------------------------------------------------------------------------
def test_evaluate_panel_returns_valid_floats():
    X_train, y_train = _data()
    X_val, y_val = _data(seed=1)
    out = evaluate_panel(
        X_train, y_train, X_val, y_val, np.array([0, 2, 5]), random_state=42
    )
    assert set(out) == {"macro_f1", "balanced_accuracy", "accuracy"}
    for value in out.values():
        assert isinstance(value, float)
        assert 0.0 <= value <= 1.0


def test_evaluate_panel_subsets_columns_and_uses_frozen_rf(monkeypatch):
    X_train, y_train = _data()
    X_val, y_val = _data(seed=1)
    idx = np.array([1, 4, 7])
    captured = {}

    class SpyRF:
        def __init__(self, **kwargs):
            captured["init_kwargs"] = kwargs

        def fit(self, X, y):
            captured["X_fit"] = np.array(X, copy=True)
            captured["y_fit"] = np.array(y, copy=True)
            return self

        def predict(self, X):
            captured["X_pred"] = np.array(X, copy=True)
            return np.zeros(X.shape[0], dtype=np.int64)

    monkeypatch.setattr(pe, "RandomForestClassifier", SpyRF)
    evaluate_panel(X_train, y_train, X_val, y_val, idx, random_state=42)

    assert np.array_equal(captured["X_fit"], X_train[:, idx])
    assert np.array_equal(captured["X_pred"], X_val[:, idx])
    assert np.array_equal(captured["y_fit"], y_train)
    assert captured["init_kwargs"] == {
        "n_estimators": 200, "random_state": 42, "n_jobs": -1,
    }


def test_evaluate_panel_ignores_unselected_columns():
    X_train, y_train = _data()
    X_val, y_val = _data(seed=1)
    idx = np.array([0, 1, 2])

    base = evaluate_panel(X_train, y_train, X_val, y_val, idx, random_state=0)

    X_train2 = X_train.copy()
    X_val2 = X_val.copy()
    X_train2[:, 6:] = np.where(X_train2[:, 6:] == 2, 0, 2).astype(np.float32)
    X_val2[:, 6:] = np.where(X_val2[:, 6:] == 2, 0, 2).astype(np.float32)

    perturbed = evaluate_panel(X_train2, y_train, X_val2, y_val, idx, random_state=0)
    assert base == perturbed


def test_evaluate_panel_reproducible_with_fixed_random_state():
    X_train, y_train = _data()
    X_val, y_val = _data(seed=1)
    idx = np.array([0, 3, 6, 9])
    a = evaluate_panel(X_train, y_train, X_val, y_val, idx, random_state=7)
    b = evaluate_panel(X_train, y_train, X_val, y_val, idx, random_state=7)
    assert a == b


def test_evaluate_panel_uses_canonical_label_set(monkeypatch):
    X_train, y_train = _data()
    X_val, y_val = _data(seed=1)
    captured = {}

    def spy_metrics(y_true, y_pred, *, labels):
        captured["labels"] = np.asarray(labels)
        return {"macro_f1": 0.5, "balanced_accuracy": 0.5, "accuracy": 0.5}

    monkeypatch.setattr(pe, "compute_classification_metrics", spy_metrics)
    evaluate_panel(
        X_train, y_train, X_val, y_val, np.array([0, 1]),
        random_state=42, n_classes=34,
    )
    assert np.array_equal(captured["labels"], np.arange(34))


def test_evaluate_panel_rejects_invalid_inputs():
    X_train, y_train = _data()
    X_val, y_val = _data(seed=1)
    p = X_train.shape[1]

    with pytest.raises(ValueError):
        evaluate_panel(X_train, y_train, X_val, y_val,
                       np.array([], dtype=np.int64))
    with pytest.raises(ValueError):
        evaluate_panel(X_train, y_train, X_val, y_val, np.array([0, 0, 1]))
    with pytest.raises(ValueError):
        evaluate_panel(X_train, y_train, X_val, y_val, np.array([0, p]))
    with pytest.raises(ValueError):
        evaluate_panel(X_train, y_train, X_val, y_val, np.array([-1, 0]))
    with pytest.raises(ValueError):
        evaluate_panel(X_train, y_train, X_val, y_val, np.array([0.5, 1.0]))
    with pytest.raises(ValueError):
        evaluate_panel(X_train, y_train[:-1], X_val, y_val, np.array([0]))
    with pytest.raises(ValueError):
        evaluate_panel(X_train[:, :5], y_train, X_val, y_val, np.array([0]))


# ---------------------------------------------------------------------------
# evaluate_full_panel
# ---------------------------------------------------------------------------
def test_evaluate_full_panel_matches_explicit_all_indices():
    X_train, y_train = _data()
    X_val, y_val = _data(seed=1)
    all_idx = np.arange(X_train.shape[1])
    full = evaluate_full_panel(X_train, y_train, X_val, y_val, random_state=42)
    explicit = evaluate_panel(
        X_train, y_train, X_val, y_val, all_idx, random_state=42
    )
    assert full == explicit
    assert set(full) == {"macro_f1", "balanced_accuracy", "accuracy"}


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
def test_aggregate_panel_records_mean_and_sample_std():
    records = [
        {"method": "fst", "K": 50, "macro_f1": 0.80,
         "balanced_accuracy": 0.70, "accuracy": 0.90},
        {"method": "fst", "K": 50, "macro_f1": 0.90,
         "balanced_accuracy": 0.90, "accuracy": 0.80},
        {"method": "fst", "K": 100, "macro_f1": 0.95,
         "balanced_accuracy": 0.95, "accuracy": 0.95},
        {"method": "full", "K": 1000, "macro_f1": 0.99,
         "balanced_accuracy": 0.99, "accuracy": 0.99},
    ]
    rows = aggregate_panel_records(records)

    assert [r["K"] for r in rows] == [50, 100]  # full excluded
    k50 = rows[0]
    assert k50["method"] == "fst"
    assert k50["n_observations"] == 2
    assert k50["macro_f1_mean"] == pytest.approx(0.85)
    expected_std = float(np.std([0.80, 0.90], ddof=1))
    assert k50["macro_f1_std"] == pytest.approx(expected_std)
    assert k50["balanced_accuracy_mean"] == pytest.approx(0.80)
    assert k50["accuracy_mean"] == pytest.approx(0.85)

    k100 = rows[1]
    assert k100["n_observations"] == 1
    assert k100["macro_f1_std"] == 0.0


# ---------------------------------------------------------------------------
# P_min criterion
# ---------------------------------------------------------------------------
def test_retention_threshold():
    assert retention_threshold(0.90, 0.02) == pytest.approx(0.882)
    assert retention_threshold(0.90, 0.05) == pytest.approx(0.855)
    with pytest.raises(ValueError):
        retention_threshold(float("nan"), 0.02)
    with pytest.raises(ValueError):
        retention_threshold(0.9, 1.0)


def test_minimum_panel_size_criterion():
    aggregates = [
        {"method": "fst", "K": 50, "macro_f1_mean": 0.90},
        {"method": "fst", "K": 100, "macro_f1_mean": 0.99},
        {"method": "fst", "K": 200, "macro_f1_mean": 1.00},
    ]
    s_full = 1.0

    assert minimum_panel_size(
        "fst", aggregates, s_full, epsilon=0.02, k_grid=[50, 100, 200]
    ) == 100
    assert minimum_panel_size(
        "fst", aggregates, s_full, epsilon=0.005, k_grid=[50, 100, 200]
    ) == 200
    assert minimum_panel_size(
        "fst", aggregates, s_full, epsilon=0.20, k_grid=[50]
    ) == 50
    assert minimum_panel_size(
        "random_forest", aggregates, s_full, epsilon=0.02, k_grid=[50, 100, 200]
    ) is None


# ---------------------------------------------------------------------------
# end-to-end smoke run (synthetic, K=[10, 20])
# ---------------------------------------------------------------------------
def test_run_smoke_executes_with_k_10_20(tmp_path):
    from scripts import run_rq3

    rq3 = tmp_path / "rq3"
    rq4 = tmp_path / "rq4"
    methods = ("fst", "random_forest", "vmgp_shap", "contrastive_ig")

    summary = run_rq3.run_smoke(
        rq3_out=str(rq3),
        rq4_out=str(rq4),
        ks=(10, 20),
        methods=methods,
        force_rankings=True,
    )

    for out_dir in (rq3, rq4):
        for name in ("protocol.json", "run_metrics.csv",
                     "aggregate_metrics.csv", "summary.json"):
            assert (out_dir / name).exists(), f"missing {out_dir / name}"

    for method in methods:
        for fold in range(3):
            for suffix in ("_scores.npy", "_meta.json", "_top.csv"):
                assert (rq3 / "rankings" / f"{method}_fold{fold}{suffix}").exists()
            assert (rq4 / "rankings" / f"{method}_fold{fold}_top.csv").exists()

    protocol = json.loads((rq3 / "protocol.json").read_text())
    assert protocol["smoke"] is True
    assert protocol["locked_test_accessed"] is False
    assert protocol["panel_sizes"] == [10, 20]
    assert protocol["experiment"] == "rq3_marker_efficiency"
    protocol_rq4 = json.loads((rq4 / "protocol.json").read_text())
    assert protocol_rq4["experiment"] == "rq4_marker_ranking"

    expected_methods = set(methods) | {"random"}
    assert set(summary["p_min"]) == expected_methods
    for per_eps in summary["p_min"].values():
        assert set(per_eps) == {"0.02", "0.05"}
    assert summary["s_full"]["per_fold"]
    assert summary["locked_test_accessed"] is False

    # run_metrics: per fold 1 full + 4 methods x 2 K + 5 random x 2 K = 19
    import csv

    with open(rq3 / "run_metrics.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3 * 19
    assert {r["method"] for r in rows} == expected_methods | {"full"}

    # top-SNP manifest carries raw .bim SNP IDs
    with open(rq3 / "rankings" / "fst_fold0_top.csv") as f:
        header = next(csv.reader(f))
    assert "snp_id" in header
    assert "raw_snp_index" in header

    # second run reuses the ranking cache (no model re-ranking needed)
    summary_cached = run_rq3.run_smoke(
        rq3_out=str(rq3),
        rq4_out=str(rq4),
        ks=(10, 20),
        methods=methods,
        force_rankings=False,
    )
    assert summary_cached["p_min"] == summary["p_min"]
