"""Marker-ranking / feature-importance pipeline (protocol-v2, zero-leakage).

RQ4 candidate SNP rankings computed strictly from the fold's TRAINING
partition:

* locus-wise ``F_ST`` (Weir & Cockerham 1984);
* Random Forest impurity importance;
* VMGP breed-head SHAP (``shap.GradientExplainer``, mean absolute attribution);
* contrastive Integrated Gradients toward the L2-normalised breed centroids
  on the unit hypersphere.

The frozen protocol-v2 set is :data:`RANKING_METHODS`. The module also hosts
extended baselines used by the robustness analyses (currently the chi-square
association test in :data:`EXTENDED_RANKING_METHODS`); they obey the same
train-only invariants but are not part of the frozen comparison.

Protocol-v2 invariants
----------------------
* Every method consumes only ``X_train`` / ``y_train``. No validation or
  locked-test row is read and no ranking is fitted on held-out data.
* Model-based methods receive a model trained on the SAME training partition;
  their background/evaluation subsets are drawn exclusively from ``X_train``.
* ``MarkerRankingResult.ranked_indices`` is a deterministic descending-score
  permutation (ties broken by ascending feature index).

Feature attribution is not evidence of biological causality: a highly ranked
SNP is one on which the method relied under these experimental conditions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .fst import weir_cockerham_fst

#: canonical dispatcher method names
RANKING_METHODS: Tuple[str, ...] = (
    "fst",
    "random_forest",
    "vmgp_shap",
    "contrastive_ig",
)

#: extended (non-frozen) baselines evaluated under the same train-only rules
EXTENDED_RANKING_METHODS: Tuple[str, ...] = (
    "chi2_association",
    "contrastive_ig_mean",
    "contrastive_ig_probe",
)


@dataclass
class MarkerRankingResult:
    """Raw marker scores plus the induced descending ranking.

    Attributes
    ----------
    method:
        Dispatcher name of the ranking method (one of :data:`RANKING_METHODS`
        or :data:`EXTENDED_RANKING_METHODS`).
    scores:
        ``(P,)`` float64 raw importance/attribution scores (higher = more
        important). Scores are method-specific and not comparable across
        methods.
    ranked_indices:
        ``(P,)`` int64 feature indices sorted by descending ``scores``;
        always a permutation of ``range(P)``.
    """

    method: str
    scores: np.ndarray
    ranked_indices: np.ndarray


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _check_training_arrays(
    X_train: np.ndarray, y_train: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate the shape/finiteness of a training partition (no external I/O)."""
    X = np.asarray(X_train, dtype=np.float32)
    y = np.asarray(y_train)
    if X.ndim != 2:
        raise ValueError("X_train must be a 2D array (samples x markers).")
    if y.shape != (X.shape[0],):
        raise ValueError(
            f"y_train must have shape ({X.shape[0]},) matching X_train rows, "
            f"got {y.shape}."
        )
    if X.shape[0] == 0:
        raise ValueError("X_train must contain at least one sample.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X_train contains non-finite values.")
    return X, y


def _make_result(method: str, scores: np.ndarray) -> MarkerRankingResult:
    """Wrap a ``(P,)`` score vector into a validated :class:`MarkerRankingResult`."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1:
        raise ValueError(f"scores must be 1D (P,), got shape {scores.shape}.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores contain non-finite values.")
    ranked_indices = np.argsort(-scores, kind="stable").astype(np.int64)
    return MarkerRankingResult(
        method=method, scores=scores, ranked_indices=ranked_indices
    )


def _check_contrastive_training(X_train: np.ndarray, y_train: np.ndarray):
    """Validate train-only inputs for the contrastive attribution methods."""
    X, y = _check_training_arrays(X_train, y_train)
    y = np.asarray(y)
    if not np.issubdtype(y.dtype, np.integer):
        raise ValueError("y_train must contain integer breed labels.")
    if y.size and y.min() < 0:
        raise ValueError("y_train contains negative breed labels.")
    n_classes = int(y.max()) + 1 if y.size else 0
    return X, y, n_classes


def _mean_one_hot_baseline(X: np.ndarray):
    """Training-mean one-hot profile used as a valid IG reference ``(M, 4)``."""
    import torch

    from contrastive_learning.augmentation import GeneticAugmentation

    with torch.no_grad():
        x = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        return GeneticAugmentation.one_hot_encode(x).float().mean(dim=0)


def _stratified_sample_indices(
    y: np.ndarray, n: int, random_state: int
) -> np.ndarray:
    """Deterministic stratified subsample of ``n`` row indices from ``y``.

    Allocation is class-proportional with a largest-remainder correction and a
    minimum of one row per populated class (when ``n`` permits). Sampling
    within each class uses ``numpy.random.default_rng(random_state)`` so that
    repeated calls with the same seed return identical indices. If
    ``n >= len(y)`` every index is returned.
    """
    y = np.asarray(y)
    n_total = int(y.size)
    if n < 1:
        raise ValueError("n must be >= 1.")
    if n >= n_total:
        return np.arange(n_total, dtype=np.int64)

    classes = np.unique(y)
    n_classes = int(classes.size)
    rng = np.random.default_rng(int(random_state))

    if n < n_classes:
        # Not enough slots to cover every class: plain uniform subsample.
        return np.sort(rng.choice(n_total, size=n, replace=False).astype(np.int64))

    counts = {int(c): int((y == c).sum()) for c in classes}

    # One slot per class, then distribute the rest proportionally.
    alloc = {c: 1 for c in counts}
    remaining = n - n_classes
    total = float(n_total)
    extras = {c: remaining * counts[c] / total for c in counts}
    for c in counts:
        alloc[c] += int(np.floor(extras[c]))
    leftover = n - sum(alloc.values())
    fractional_order = sorted(
        counts, key=lambda c: (-(extras[c] - np.floor(extras[c])), c)
    )
    i = 0
    while leftover > 0:
        c = fractional_order[i % n_classes]
        if alloc[c] < counts[c]:
            alloc[c] += 1
            leftover -= 1
        i += 1

    chosen = []
    for c in sorted(counts):
        class_pos = np.flatnonzero(y == c)
        take = min(alloc[c], class_pos.size)
        chosen.append(rng.permutation(class_pos)[:take])
    return np.sort(np.concatenate(chosen)).astype(np.int64)


# ---------------------------------------------------------------------------
# Ranking methods
# ---------------------------------------------------------------------------
def rank_fst(
    X_train: np.ndarray, y_train: np.ndarray, n_classes: int
) -> MarkerRankingResult:
    """Locus-wise Weir & Cockerham (1984) F_ST ranking on training data only."""
    X, y = _check_training_arrays(X_train, y_train)
    scores = weir_cockerham_fst(X, y, n_classes)
    return _make_result("fst", scores)


def rank_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 200,
    random_state: int = 42,
    n_jobs: int = -1,
) -> MarkerRankingResult:
    """Random Forest impurity-importance ranking fitted on training data only."""
    if n_estimators < 1:
        raise ValueError("n_estimators must be >= 1.")
    X, y = _check_training_arrays(X_train, y_train)
    forest = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state, n_jobs=n_jobs
    )
    forest.fit(X, y)
    return _make_result("random_forest", forest.feature_importances_)


def rank_chi2_association(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int | None = None,
) -> MarkerRankingResult:
    """Per-marker chi-square association between genotype and breed (train-only).

    Extended classical baseline (not part of the frozen protocol-v2 set): each
    marker is scored by the chi-square statistic of the
    ``genotype (0/1/2) x breed`` contingency table computed on the fold's
    training partition only. In a task without an external phenotype this is
    the univariate association test that plays the role of a GWAS-style
    analysis, with the breed label as the outcome. Markers whose table is
    degenerate (a single genotype category or a single breed) receive a zero
    score; the raw statistic is used, so scores are not standardised across
    markers.

    Parameters
    ----------
    X_train:
        ``(N, P)`` dosage matrix with values in ``{0, 1, 2}``.
    y_train:
        ``(N,)`` non-negative integer breed labels.
    n_classes:
        Optional canonical class count used to size the contingency table
        (defaults to ``max(y_train) + 1``).
    """
    X, _ = _check_training_arrays(X_train, y_train)
    if not np.issubdtype(np.asarray(y_train).dtype, np.integer):
        raise ValueError("y_train must contain integer breed labels.")
    y = np.asarray(y_train, dtype=np.int64)
    if y.size and y.min() < 0:
        raise ValueError("y_train contains negative breed labels.")
    if n_classes is None:
        n_classes = int(y.max()) + 1 if y.size else 0
    n_classes = int(n_classes)
    if n_classes < 1:
        raise ValueError("n_classes must be >= 1.")

    G = np.rint(X).astype(np.int64)
    if G.min() < 0 or G.max() > 2:
        raise ValueError("X_train dosages must lie in {0, 1, 2}.")

    n_samples, n_features = G.shape
    class_counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    chi2 = np.zeros(n_features, dtype=np.float64)
    for genotype in (0, 1, 2):
        present = G == genotype
        row_counts = present.sum(axis=0).astype(np.float64)
        if not np.any(row_counts):
            continue
        observed = np.empty((n_classes, n_features), dtype=np.float64)
        for breed in range(n_classes):
            observed[breed] = present[y == breed].sum(axis=0)
        expected = row_counts[None, :] * class_counts[:, None] / float(n_samples)
        positive = expected > 0.0
        contribution = np.zeros_like(expected)
        contribution[positive] = (
            observed[positive] - expected[positive]
        ) ** 2 / expected[positive]
        chi2 += contribution.sum(axis=0)
    return _make_result("chi2_association", chi2)


def rank_vmgp_shap(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_background: int = 100,
    n_eval: int = 200,
    random_state: int = 42,
) -> MarkerRankingResult:
    """SHAP ``GradientExplainer`` attribution of the VMGP breed head (train-only).

    ``model`` is a ``VMGP_LightningSystem`` (or any model accepted by
    :class:`vae.attribution.BreedWrapper`). The explainer background and the
    evaluation subset are deterministic stratified samples of ``X_train``; no
    validation or locked-test row is used. Per-SNP scores are the mean absolute
    attribution over evaluation samples and breed outputs
    (:func:`vae.attribution.aggregate_shap_to_global`).
    """
    import shap
    import torch

    from vae.attribution import BreedWrapper, aggregate_shap_to_global

    if n_background < 1:
        raise ValueError("n_background must be >= 1.")
    if n_eval < 1:
        raise ValueError("n_eval must be >= 1.")

    X, y = _check_training_arrays(X_train, y_train)
    n_features = int(X.shape[1])

    background_idx = _stratified_sample_indices(y, n_background, random_state)
    eval_idx = _stratified_sample_indices(y, n_eval, random_state + 1)

    wrapper = BreedWrapper(model)
    wrapper.eval()

    device = next(wrapper.parameters()).device
    background = torch.as_tensor(X[background_idx], dtype=torch.float32, device=device)
    eval_samples = torch.as_tensor(X[eval_idx], dtype=torch.float32, device=device)

    explainer = shap.GradientExplainer(wrapper, background)
    shap_values = explainer.shap_values(eval_samples, rseed=int(random_state))
    scores = aggregate_shap_to_global(shap_values, n_features=n_features)
    return _make_result("vmgp_shap", scores)


def rank_contrastive_ig(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_steps: int = 50,
    batch_size: int = 32,
    device: str = "cpu",
) -> MarkerRankingResult:
    """Contrastive Integrated Gradients toward the breed centroids (train-only).

    Embeddings are extracted with the deterministic path (``augment=False``) and
    the L2-normalised per-breed centroid is computed from ``X_train`` only.
    Each training sample is attributed toward its OWN breed centroid, then
    scores are the per-breed mean followed by the equal-weight mean across
    breeds (see :func:`contrastive_learning.attribution.aggregate_attributions`).

    ``model`` is moved to ``device`` (an inference helper side effect).
    """
    import torch

    from contrastive_learning.attribution import (
        EncoderToCentroid,
        aggregate_attributions,
        compute_breed_centroids,
        compute_centroid_ig_batched,
    )
    from contrastive_learning.evaluation import extract_embeddings

    if n_steps < 1:
        raise ValueError("n_steps must be >= 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")

    X, y, n_classes = _check_contrastive_training(X_train, y_train)

    dev = torch.device(device)
    model = model.to(dev)
    model.eval()

    embeddings = extract_embeddings(model, X)
    centroids = compute_breed_centroids(embeddings, y, n_classes)
    wrapper = EncoderToCentroid(
        model.encoder, torch.as_tensor(centroids, dtype=torch.float32)
    ).to(dev)

    target_classes = torch.as_tensor(y, dtype=torch.long, device=dev)
    ig_all = compute_centroid_ig_batched(
        wrapper,
        X,
        target_classes,
        dev,
        n_steps=n_steps,
        batch_size=batch_size,
    )
    scores, _ = aggregate_attributions(ig_all, y, n_classes)
    return _make_result("contrastive_ig", scores)


def rank_contrastive_ig_mean(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_steps: int = 50,
    batch_size: int = 32,
    device: str = "cpu",
) -> MarkerRankingResult:
    """Centroid-path IG with a valid (training-mean) baseline (extended).

    Same centroid-path attribution as :func:`rank_contrastive_ig`, but the IG
    reference is the mean one-hot profile of the fold's training partition
    instead of the zero vector, which does not correspond to any genotype
    configuration. The comparison isolates the effect of the baseline on the
    ranking, since the embedding, the centroids and the aggregation are
    identical.
    """
    import torch

    from contrastive_learning.attribution import (
        EncoderToCentroid,
        aggregate_attributions,
        compute_breed_centroids,
        compute_centroid_ig_batched,
    )
    from contrastive_learning.evaluation import extract_embeddings

    if n_steps < 1:
        raise ValueError("n_steps must be >= 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")

    X, y, n_classes = _check_contrastive_training(X_train, y_train)

    dev = torch.device(device)
    model = model.to(dev)
    model.eval()

    embeddings = extract_embeddings(model, X)
    centroids = compute_breed_centroids(embeddings, y, n_classes)
    wrapper = EncoderToCentroid(
        model.encoder, torch.as_tensor(centroids, dtype=torch.float32)
    ).to(dev)

    baseline = _mean_one_hot_baseline(X)
    target_classes = torch.as_tensor(y, dtype=torch.long, device=dev)
    ig_all = compute_centroid_ig_batched(
        wrapper,
        X,
        target_classes,
        dev,
        n_steps=n_steps,
        batch_size=batch_size,
        baseline=baseline,
    )
    scores, _ = aggregate_attributions(ig_all, y, n_classes)
    return _make_result("contrastive_ig_mean", scores)


def rank_contrastive_ig_probe(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_steps: int = 50,
    batch_size: int = 32,
    device: str = "cpu",
    probe_epochs: int = 300,
    random_state: int = 42,
) -> MarkerRankingResult:
    """Probe-path IG with a valid (training-mean) baseline (extended).

    A supervised MLP probe is trained on the fold's training embeddings with a
    deterministic internal split (``random_state``), the IG reference is the
    mean one-hot training profile, and per-SNP scores are aggregated per-breed
    with equal weight (as in :func:`rank_contrastive_ig`). This combines a
    supervised read-out of the embedding space with a reference that
    corresponds to a plausible genotype profile.
    """
    import torch

    from contrastive_learning.attribution import (
        EncoderWithProbeMLP,
        aggregate_attributions,
        compute_ig_batched,
        train_probe_mlp,
    )
    from contrastive_learning.evaluation import extract_embeddings

    if n_steps < 1:
        raise ValueError("n_steps must be >= 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    if probe_epochs < 1:
        raise ValueError("probe_epochs must be >= 1.")

    X, y, n_classes = _check_contrastive_training(X_train, y_train)

    dev = torch.device(device)
    torch.manual_seed(int(random_state))
    model = model.to(dev)
    model.eval()

    embeddings = extract_embeddings(model, X)
    probe, scaler, _ = train_probe_mlp(
        embeddings, y, n_classes, dev, epochs=int(probe_epochs)
    )
    wrapper = EncoderWithProbeMLP(
        model.encoder, scaler.mean_, scaler.scale_, probe
    ).to(dev)

    baseline = _mean_one_hot_baseline(X)
    target_classes = torch.as_tensor(y, dtype=torch.long, device=dev)
    ig_all = compute_ig_batched(
        wrapper,
        X,
        target_classes,
        dev,
        n_steps=n_steps,
        batch_size=batch_size,
        baseline=baseline,
    )
    scores, _ = aggregate_attributions(ig_all, y, n_classes)
    return _make_result("contrastive_ig_probe", scores)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
_MODEL_METHODS = ("vmgp_shap", "contrastive_ig", "contrastive_ig_mean",
                  "contrastive_ig_probe")

_DISPATCH: Dict[str, object] = {
    "fst": rank_fst,
    "random_forest": rank_random_forest,
    "chi2_association": rank_chi2_association,
    "vmgp_shap": rank_vmgp_shap,
    "contrastive_ig": rank_contrastive_ig,
    "contrastive_ig_mean": rank_contrastive_ig_mean,
    "contrastive_ig_probe": rank_contrastive_ig_probe,
}


def compute_marker_ranking(
    method: str, X_train: np.ndarray, y_train: np.ndarray, **kwargs
) -> MarkerRankingResult:
    """Dispatch to a ranking method by name.

    Valid names are the frozen protocol-v2 methods (:data:`RANKING_METHODS`)
    plus the extended baselines (:data:`EXTENDED_RANKING_METHODS`).

    Model-based methods (``vmgp_shap``, ``contrastive_ig``) require a ``model``
    keyword argument. All methods consume only ``X_train`` / ``y_train`` plus
    method-specific configuration; no held-out data is accepted or read.
    """
    if method not in _DISPATCH:
        valid = list(RANKING_METHODS) + list(EXTENDED_RANKING_METHODS)
        raise ValueError(
            f"Unknown ranking method {method!r}; valid methods: {valid}."
        )
    func = _DISPATCH[method]
    if method in _MODEL_METHODS:
        if "model" not in kwargs:
            raise TypeError(
                f"ranking method {method!r} requires a 'model' keyword argument."
            )
        model = kwargs.pop("model")
        return func(model, X_train, y_train, **kwargs)
    return func(X_train, y_train, **kwargs)
