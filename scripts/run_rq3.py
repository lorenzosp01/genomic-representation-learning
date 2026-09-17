#!/usr/bin/env python
"""RQ3 marker-efficiency / RQ4 marker-selection-strategy experiment (protocol-v2).

Per development fold (0, 1, 2), using only the fold TRAINING partition for
ranking and the fold VALIDATION partition for evaluation (the locked test is
never accessed):

1. build the fold's train-only preprocessed data
   (``GenomicExperimentData.fold_data``);
2. compute marker rankings on ``(fold.X_train, fold.y_train)`` with
   ``fst`` / ``random_forest`` / ``vmgp_shap`` / ``contrastive_ig``
   (``genomic.ranking``, cached and resumable);
3. evaluate the frozen downstream evaluator
   ``RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)``
   on the full post-QC panel (S_full) and on every K in the frozen grid;
4. repeat every K with 5 random subsets per fold as a chance baseline.

Retention rule (fixed before interpreting results):
``P_min(method, epsilon)`` is the smallest K in the frozen grid whose mean
Macro-F1 across folds is ``>= (1 - epsilon) * mean(S_full)``; ``None`` if no
grid K qualifies. Both ``epsilon_P = 0.02`` and ``0.05`` are reported.

Artifacts are written atomically to ``results/rq3_marker_efficiency/`` and
``results/rq4_marker_ranking/``: ``protocol.json``, ``run_metrics.csv``,
``aggregate_metrics.csv``, ``summary.json`` and ``rankings/`` (full score
vectors, cache manifests and top-SNP lists carrying raw ``.bim`` SNP IDs).

Usage:
    python scripts/run_rq3.py                     # resume-safe (rankings cached)
    python scripts/run_rq3.py --force-rankings    # recompute all rankings
    python scripts/run_rq3.py --smoke             # tiny synthetic dry-run, K=[10,20]
    python scripts/run_rq4.py                     # companion alias (same pipeline)
"""

from __future__ import annotations

import argparse
import csv
import gc
import glob
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Make the repository root importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genomic.experiment_data import GenomicExperimentData, locked_test_overlap
from genomic.panel_evaluation import (
    aggregate_panel_records,
    evaluate_full_panel,
    evaluate_panel,
    minimum_panel_size,
)
from genomic.ranking import RANKING_METHODS, MarkerRankingResult, compute_marker_ranking
from genomic.rq2 import save_json_atomic
from genomic.splitting import load_split

from contrastive_learning.model import ContrastiveGeneticModel
from vae.lightning_module import VMGP_LightningSystem

# ---------------------------------------------------------------------------
# Frozen protocol constants
# ---------------------------------------------------------------------------
DATASET_ID = "ADAPTmap_genotypeTOP_20160222_full"
FAM_PATH = "data/ADAPTmap_genotypeTOP_20160222_full.fam"
BED_PATH = "data/ADAPTmap_genotypeTOP_20160222_full.bed"
BIM_PATH = "data/ADAPTmap_genotypeTOP_20160222_full.bim"
SPLITS_DIR = "splits"
COHORT_LABEL = "indmiss0p10"

RQ3_OUT_DIR = "results/rq3_marker_efficiency"
RQ4_OUT_DIR = "results/rq4_marker_ranking"
SMOKE_RQ3_OUT_DIR = "results/_smoke_rq3"
SMOKE_RQ4_OUT_DIR = "results/_smoke_rq4"

FOLDS: Tuple[int, ...] = (0, 1, 2)
K_GRID: Tuple[int, ...] = (50, 100, 200, 500, 1000, 2000, 5000)
EPSILON_P: Tuple[float, ...] = (0.02, 0.05)
METHODS: Tuple[str, ...] = ("fst", "random_forest", "vmgp_shap", "contrastive_ig")
RANDOM_METHOD = "random"
FULL_METHOD = "full"
RANDOM_REPLICATES = 5
RANDOM_SEED = 42
N_CLASSES_PRIMARY = 34
RF_RANDOM_STATE = 42

DEFAULT_VAE_CKPT_PATTERN = "checkpoints/LatDim96_fold{fold}/fold_{fold1}/epoch=*.ckpt"
DEFAULT_CONTRASTIVE_CKPT_PATTERN = "checkpoints/rq1_fold{fold}/fold_{fold1}/epoch=*.ckpt"

RUN_CSV_FIELDS = [
    "fold", "method", "K", "n_selected", "replicate", "selection_source",
    "macro_f1", "balanced_accuracy", "accuracy",
    "n_train", "n_val", "n_markers_fold",
]
AGG_CSV_FIELDS = [
    "method", "K", "n_observations",
    "macro_f1_mean", "macro_f1_std",
    "balanced_accuracy_mean", "balanced_accuracy_std",
    "accuracy_mean", "accuracy_std",
]


# ---------------------------------------------------------------------------
# Fold specification
# ---------------------------------------------------------------------------
@dataclass
class FoldSpec:
    """One development fold's data + optional trained models/checkpoint paths.

    Duck-typed so smoke tests can attach tiny synthetic arrays and toy models
    without touching PLINK data. ``retained_snp_metadata`` is the
    post-QC-aligned ``.bim`` metadata dict (SNP IDs, chromosome, position).
    """

    fold: int
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    retained_snp_metadata: Optional[Dict[str, np.ndarray]] = None
    retained_snp_indices: Optional[np.ndarray] = None
    vae_model: Optional[object] = None
    contrastive_model: Optional[object] = None
    vae_checkpoint: Optional[str] = None
    contrastive_checkpoint: Optional[str] = None

    @property
    def n_features(self) -> int:
        return int(self.X_train.shape[1])


# ---------------------------------------------------------------------------
# Atomic I/O helpers
# ---------------------------------------------------------------------------
def _write_csv_atomic(rows: Sequence[dict], path: str,
                      fieldnames: Optional[Sequence[str]] = None) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    os.replace(tmp, path)
    return path


def _save_npy_atomic(arr: np.ndarray, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.save(f, np.asarray(arr))
    os.replace(tmp, path)
    return path


def _find_fold_checkpoint(pattern: str) -> str:
    """Resolve a fold checkpoint glob; fail loudly on zero or ambiguity."""
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No checkpoint matches {pattern!r}.")
    if len(matches) > 1:
        raise RuntimeError(
            f"Ambiguous checkpoint pattern {pattern!r} matched {len(matches)} "
            f"files: {matches}"
        )
    return matches[0]


# ---------------------------------------------------------------------------
# Ranking computation + cache
# ---------------------------------------------------------------------------
def _ranking_params(
    method: str,
    n_classes: int,
    shap_kwargs: dict,
    ig_kwargs: dict,
) -> dict:
    """Method-specific ranking parameters excluding the (non-serializable) model."""
    if method == "fst":
        return {"n_classes": int(n_classes)}
    if method == "random_forest":
        return {}
    if method == "vmgp_shap":
        return dict(shap_kwargs)
    if method == "contrastive_ig":
        return dict(ig_kwargs)
    raise ValueError(
        f"Unknown ranking method {method!r}; valid: {list(RANKING_METHODS)}."
    )


def _ranking_call_kwargs(method: str, spec: FoldSpec, params: dict) -> dict:
    if method == "vmgp_shap":
        if spec.vae_model is None:
            raise RuntimeError(
                f"fold {spec.fold}: vmgp_shap requires a loaded VMGP model "
                "(checkpoint cache miss)."
            )
        return {"model": spec.vae_model, **params}
    if method == "contrastive_ig":
        if spec.contrastive_model is None:
            raise RuntimeError(
                f"fold {spec.fold}: contrastive_ig requires a loaded contrastive "
                "model (checkpoint cache miss)."
            )
        return {"model": spec.contrastive_model, **params}
    return dict(params)


def _checkpoint_for(method: str, spec: FoldSpec) -> Optional[str]:
    if method == "vmgp_shap":
        return spec.vae_checkpoint
    if method == "contrastive_ig":
        return spec.contrastive_checkpoint
    return None


def _cache_paths(rankings_dir: str, method: str, fold: int) -> Tuple[str, str]:
    base = os.path.join(rankings_dir, f"{method}_fold{fold}")
    return base + "_scores.npy", base + "_meta.json"


def _cache_matches(meta: dict, *, method: str, spec: FoldSpec, params: dict,
                   checkpoint: Optional[str]) -> bool:
    return (
        meta.get("method") == method
        and int(meta.get("fold", -1)) == int(spec.fold)
        and int(meta.get("n_features", -1)) == spec.n_features
        and int(meta.get("n_samples", -1)) == int(spec.X_train.shape[0])
        and meta.get("checkpoint") == checkpoint
        and meta.get("ranking_kwargs") == params
    )


def _cache_hit(spec: FoldSpec, method: str, rankings_dir: str, params: dict,
               checkpoint: Optional[str]) -> bool:
    """Cheap cache probe (metadata + score shape) without computing rankings."""
    scores_path, meta_path = _cache_paths(rankings_dir, method, spec.fold)
    if not (os.path.exists(scores_path) and os.path.exists(meta_path)):
        return False
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    if not _cache_matches(meta, method=method, spec=spec, params=params,
                          checkpoint=checkpoint):
        return False
    try:
        scores = np.load(scores_path, mmap_mode="r")
    except (OSError, ValueError):
        return False
    return scores.ndim == 1 and scores.shape[0] == spec.n_features


def _save_ranking_manifest(result: MarkerRankingResult, spec: FoldSpec,
                           out_dir: str, top_n: int) -> str:
    """Persist rank / panel index / score plus raw ``.bim`` SNP identity."""
    rankings_dir = os.path.join(out_dir, "rankings")
    os.makedirs(rankings_dir, exist_ok=True)
    top = min(int(top_n), spec.n_features)
    selected = result.ranked_indices[:top]

    metadata = spec.retained_snp_metadata
    meta_keys = (
        "raw_snp_index", "snp_id", "chromosome", "position",
        "allele_1", "allele_2",
    )
    rows: List[dict] = []
    for rank, j in enumerate(selected, start=1):
        j = int(j)
        row = {
            "rank": rank,
            "panel_index": j,
            "score": float(result.scores[j]),
        }
        if metadata is not None:
            for key in meta_keys:
                if key in metadata:
                    row[key] = metadata[key][j]
        rows.append(row)

    path = os.path.join(rankings_dir, f"{result.method}_fold{spec.fold}_top.csv")
    return _write_csv_atomic(rows, path)


def compute_fold_rankings(
    spec: FoldSpec,
    methods: Sequence[str],
    *,
    ks: Sequence[int],
    cache_dir: str,
    extra_manifest_dirs: Sequence[str] = (),
    force: bool = False,
    n_classes: int = N_CLASSES_PRIMARY,
    shap_kwargs: Optional[dict] = None,
    ig_kwargs: Optional[dict] = None,
) -> Dict[str, MarkerRankingResult]:
    """Compute (or load from cache) one fold's rankings, strictly train-only.

    Cached scores are validated against method, fold, marker count, sample
    count, checkpoint path and ranking parameters before reuse. Manifests
    (top-SNP lists) are written to ``cache_dir/rankings`` and every
    ``extra_manifest_dirs`` entry.
    """
    shap_kwargs = dict(shap_kwargs or {})
    ig_kwargs = dict(ig_kwargs or {})
    rankings_dir = os.path.join(cache_dir, "rankings")
    os.makedirs(rankings_dir, exist_ok=True)

    results: Dict[str, MarkerRankingResult] = {}
    for method in methods:
        params = _ranking_params(method, n_classes, shap_kwargs, ig_kwargs)
        checkpoint = _checkpoint_for(method, spec)
        scores_path, meta_path = _cache_paths(rankings_dir, method, spec.fold)

        result: Optional[MarkerRankingResult] = None
        if not force and _cache_hit(spec, method, rankings_dir, params, checkpoint):
            scores = np.asarray(np.load(scores_path), dtype=np.float64)
            result = MarkerRankingResult(
                method=method,
                scores=scores,
                ranked_indices=np.argsort(-scores, kind="stable").astype(np.int64),
            )

        if result is None:
            call_kwargs = _ranking_call_kwargs(method, spec, params)
            result = compute_marker_ranking(
                method, spec.X_train, spec.y_train, **call_kwargs
            )
            if result.scores.shape != (spec.n_features,):
                raise RuntimeError(
                    f"fold {spec.fold}: {method} returned shape "
                    f"{result.scores.shape}, expected ({spec.n_features},)."
                )
            _save_npy_atomic(result.scores, scores_path)
            save_json_atomic(
                {
                    "method": method,
                    "fold": int(spec.fold),
                    "n_features": spec.n_features,
                    "n_samples": int(spec.X_train.shape[0]),
                    "n_classes": int(n_classes),
                    "checkpoint": checkpoint,
                    "ranking_kwargs": params,
                },
                meta_path,
            )

        for out_dir in (cache_dir, *extra_manifest_dirs):
            _save_ranking_manifest(result, spec, out_dir, top_n=max(ks))
        results[method] = result
    return results


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def _record(spec: FoldSpec, method: str, k: int, n_selected: int, replicate: int,
            metrics: dict, selection_source: str) -> dict:
    return {
        "fold": int(spec.fold),
        "method": method,
        "K": int(k),
        "n_selected": int(n_selected),
        "replicate": int(replicate),
        "selection_source": selection_source,
        "macro_f1": float(metrics["macro_f1"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "accuracy": float(metrics["accuracy"]),
        "n_train": int(spec.X_train.shape[0]),
        "n_val": int(spec.X_val.shape[0]),
        "n_markers_fold": int(spec.n_features),
    }


def evaluate_fold(
    spec: FoldSpec,
    rankings: Dict[str, MarkerRankingResult],
    *,
    ks: Sequence[int],
    random_replicates: int,
    random_seed: int,
    n_classes: int,
    rf_random_state: int = RF_RANDOM_STATE,
) -> List[dict]:
    """S_full + deterministic-method panels + random chance baseline for one fold."""
    records: List[dict] = []

    full_metrics = evaluate_full_panel(
        spec.X_train, spec.y_train, spec.X_val, spec.y_val,
        random_state=rf_random_state, n_classes=n_classes,
    )
    records.append(_record(
        spec, FULL_METHOD, spec.n_features, spec.n_features, -1,
        full_metrics, "all",
    ))

    for method, result in rankings.items():
        for k in ks:
            metrics = evaluate_panel(
                spec.X_train, spec.y_train, spec.X_val, spec.y_val,
                result.ranked_indices[: int(k)],
                random_state=rf_random_state, n_classes=n_classes,
            )
            records.append(_record(spec, method, int(k), int(k), -1, metrics, "ranked"))

    for replicate in range(int(random_replicates)):
        rng = np.random.default_rng([int(random_seed), int(spec.fold), int(replicate)])
        for k in ks:
            indices = rng.choice(spec.n_features, size=int(k), replace=False)
            metrics = evaluate_panel(
                spec.X_train, spec.y_train, spec.X_val, spec.y_val,
                indices, random_state=rf_random_state, n_classes=n_classes,
            )
            records.append(_record(
                spec, RANDOM_METHOD, int(k), int(k), replicate, metrics, "random",
            ))

    return records


# ---------------------------------------------------------------------------
# Aggregation / summary / persistence
# ---------------------------------------------------------------------------
def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    ddof = 1 if arr.size > 1 else 0
    return float(arr.mean()), float(arr.std(ddof=ddof))


def build_summary(records: Sequence[dict], *, ks: Sequence[int],
                  methods: Sequence[str], epsilons: Sequence[float]) -> dict:
    """Build S_full, per-(method, K) aggregates and the epsilon_P P_min table."""
    panel_records = [r for r in records if r["method"] != FULL_METHOD]
    aggregates = aggregate_panel_records(panel_records)

    full_records = sorted(
        [r for r in records if r["method"] == FULL_METHOD], key=lambda r: r["fold"]
    )
    if not full_records:
        raise ValueError("no full-panel records available for S_full.")
    s_full_macro_f1, s_full_macro_f1_std = _mean_std(
        [r["macro_f1"] for r in full_records]
    )
    s_full_bal_acc, s_full_bal_acc_std = _mean_std(
        [r["balanced_accuracy"] for r in full_records]
    )
    s_full_acc, s_full_acc_std = _mean_std([r["accuracy"] for r in full_records])

    evaluated_methods = list(methods) + [RANDOM_METHOD]
    p_min: Dict[str, Dict[str, Optional[int]]] = {}
    thresholds: Dict[str, Dict[str, float]] = {}
    for method in evaluated_methods:
        p_min[method] = {}
        thresholds[method] = {}
        for eps in epsilons:
            key = f"{float(eps):.2f}"
            thresholds[method][key] = float((1.0 - float(eps)) * s_full_macro_f1)
            p_min[method][key] = minimum_panel_size(
                method, aggregates, s_full_macro_f1,
                epsilon=float(eps), k_grid=ks,
            )

    return {
        "s_full": {
            "per_fold": [
                {
                    "fold": int(r["fold"]),
                    "n_markers": int(r["n_markers_fold"]),
                    "macro_f1": float(r["macro_f1"]),
                    "balanced_accuracy": float(r["balanced_accuracy"]),
                    "accuracy": float(r["accuracy"]),
                }
                for r in full_records
            ],
            "macro_f1_mean": s_full_macro_f1,
            "macro_f1_std": s_full_macro_f1_std,
            "balanced_accuracy_mean": s_full_bal_acc,
            "balanced_accuracy_std": s_full_bal_acc_std,
            "accuracy_mean": s_full_acc,
            "accuracy_std": s_full_acc_std,
            "n_markers": [int(r["n_markers_fold"]) for r in full_records],
        },
        "epsilon_P": [float(e) for e in epsilons],
        "retention_thresholds": thresholds,
        "p_min": p_min,
        "panel_sizes": [int(k) for k in ks],
        "methods": list(methods),
        "random_baseline": {"method": RANDOM_METHOD, "replicates": len(set(
            r["replicate"] for r in panel_records if r["method"] == RANDOM_METHOD
        ))},
        "aggregates": aggregates,
        "n_folds": len(full_records),
        "locked_test_accessed": False,
    }


def _build_protocol(
    *,
    split_meta: dict,
    fold_n_features: Dict[int, int],
    checkpoints: dict,
    methods: Sequence[str],
    ks: Sequence[int],
    epsilons: Sequence[float],
    random_replicates: int,
    random_seed: int,
    shap_kwargs: dict,
    ig_kwargs: dict,
    smoke: bool,
) -> dict:
    from genomic.panel_evaluation import RF_N_ESTIMATORS, RF_RANDOM_STATE

    return {
        "dataset_id": DATASET_ID,
        "cohort": {
            "label": COHORT_LABEL,
            "description": "primary 34-breed development cohort",
            "n_classes": N_CLASSES_PRIMARY,
            "n_folds": len(fold_n_features),
        },
        "folds": {str(k): {"n_markers": int(v)} for k, v in sorted(fold_n_features.items())},
        "evaluator": {
            "type": "RandomForestClassifier",
            "n_estimators": RF_N_ESTIMATORS,
            "random_state": RF_RANDOM_STATE,
            "n_jobs": -1,
        },
        "full_panel_baseline": "all post-QC markers of each training fold (S_full)",
        "panel_sizes": [int(k) for k in ks],
        "epsilon_P": [float(e) for e in epsilons],
        "retention_rule": (
            "P_min = smallest grid K with mean Macro-F1 >= (1 - epsilon_P) * "
            "mean(S_full); None when no grid K qualifies"
        ),
        "methods": list(methods),
        "random_baseline": {
            "method": RANDOM_METHOD,
            "replicates": int(random_replicates),
            "seeding": "numpy.random.default_rng([seed, fold, replicate])",
            "seed": int(random_seed),
        },
        "ranking_kwargs": {
            "vmgp_shap": dict(shap_kwargs),
            "contrastive_ig": dict(ig_kwargs),
        },
        "checkpoints": checkpoints,
        "split_meta": split_meta,
        "smoke": bool(smoke),
        "locked_test_accessed": False,
    }


def _persist(out_dir: str, *, experiment: str, protocol: dict,
             records: Sequence[dict], summary: dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    save_json_atomic(
        {**protocol, "experiment": experiment},
        os.path.join(out_dir, "protocol.json"),
    )
    _write_csv_atomic(records, os.path.join(out_dir, "run_metrics.csv"),
                      fieldnames=RUN_CSV_FIELDS)
    _write_csv_atomic(summary["aggregates"],
                      os.path.join(out_dir, "aggregate_metrics.csv"),
                      fieldnames=AGG_CSV_FIELDS)
    save_json_atomic(summary, os.path.join(out_dir, "summary.json"))


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def _run_pipeline(
    fold_specs: Iterable[FoldSpec],
    *,
    ks: Sequence[int],
    methods: Sequence[str],
    random_replicates: int,
    random_seed: int,
    rq3_out: str,
    rq4_out: str,
    cache_dir: str,
    force_rankings: bool,
    n_classes: int,
    shap_kwargs: dict,
    ig_kwargs: dict,
    smoke: bool,
    split_meta: dict,
    load_models: Optional[Callable[[FoldSpec], None]] = None,
) -> dict:
    methods = tuple(methods)
    ks = tuple(int(k) for k in ks)
    records: List[dict] = []
    checkpoints: dict = {}
    fold_n_features: Dict[int, int] = {}

    for spec in fold_specs:
        if max(ks) > spec.n_features:
            raise ValueError(
                f"fold {spec.fold}: largest K={max(ks)} exceeds the fold's "
                f"{spec.n_features} post-QC markers."
            )
        if load_models is not None:
            load_models(spec)

        rankings = compute_fold_rankings(
            spec, methods,
            ks=ks,
            cache_dir=cache_dir,
            extra_manifest_dirs=(rq4_out,),
            force=force_rankings,
            n_classes=n_classes,
            shap_kwargs=shap_kwargs,
            ig_kwargs=ig_kwargs,
        )
        records.extend(evaluate_fold(
            spec, rankings,
            ks=ks,
            random_replicates=random_replicates,
            random_seed=random_seed,
            n_classes=n_classes,
        ))

        fold_n_features[int(spec.fold)] = spec.n_features
        if spec.vae_checkpoint is not None:
            checkpoints.setdefault("vae", {})[str(spec.fold)] = spec.vae_checkpoint
        if spec.contrastive_checkpoint is not None:
            checkpoints.setdefault("contrastive", {})[str(spec.fold)] = (
                spec.contrastive_checkpoint
            )

        spec.vae_model = None
        spec.contrastive_model = None
        gc.collect()

    summary = build_summary(records, ks=ks, methods=methods, epsilons=EPSILON_P)
    protocol = _build_protocol(
        split_meta=split_meta,
        fold_n_features=fold_n_features,
        checkpoints=checkpoints,
        methods=methods,
        ks=ks,
        epsilons=EPSILON_P,
        random_replicates=random_replicates,
        random_seed=random_seed,
        shap_kwargs=shap_kwargs,
        ig_kwargs=ig_kwargs,
        smoke=smoke,
    )
    _persist(rq3_out, experiment="rq3_marker_efficiency", protocol=protocol,
             records=records, summary=summary)
    _persist(rq4_out, experiment="rq4_marker_ranking", protocol=protocol,
             records=records, summary=summary)
    return summary


def _validate_methods(methods: Sequence[str]) -> None:
    invalid = [m for m in methods if m not in RANKING_METHODS]
    if invalid:
        raise ValueError(
            f"Unsupported ranking methods {invalid}; valid: {list(RANKING_METHODS)}."
        )


def _load_fold_models(
    spec: FoldSpec,
    *,
    methods: Sequence[str],
    n_classes: int,
    rankings_dir: str,
    force_rankings: bool,
    shap_kwargs: dict,
    ig_kwargs: dict,
) -> None:
    """Load only the checkpoints whose cached rankings are missing/invalid."""
    if "vmgp_shap" in methods:
        if spec.vae_checkpoint is None:
            raise RuntimeError(f"fold {spec.fold}: VMGP checkpoint path not resolved.")
        params = _ranking_params("vmgp_shap", n_classes, shap_kwargs, ig_kwargs)
        if force_rankings or not _cache_hit(
            spec, "vmgp_shap", rankings_dir, params, spec.vae_checkpoint
        ):
            model = VMGP_LightningSystem.load_from_checkpoint(
                spec.vae_checkpoint, map_location="cpu"
            )
            if int(model.hparams.num_snps) != spec.n_features:
                raise RuntimeError(
                    f"fold {spec.fold}: VMGP checkpoint has "
                    f"num_snps={int(model.hparams.num_snps)} but the fold has "
                    f"{spec.n_features} markers; refusing mismatched ranking."
                )
            if int(model.hparams.num_classes_breed) != n_classes:
                raise RuntimeError(
                    f"fold {spec.fold}: VMGP checkpoint has "
                    f"{int(model.hparams.num_classes_breed)} breed classes, "
                    f"expected {n_classes}."
                )
            spec.vae_model = model
    if "contrastive_ig" in methods:
        if spec.contrastive_checkpoint is None:
            raise RuntimeError(
                f"fold {spec.fold}: contrastive checkpoint path not resolved."
            )
        params = _ranking_params("contrastive_ig", n_classes, shap_kwargs, ig_kwargs)
        if force_rankings or not _cache_hit(
            spec, "contrastive_ig", rankings_dir, params, spec.contrastive_checkpoint
        ):
            model = ContrastiveGeneticModel.load_from_checkpoint(
                spec.contrastive_checkpoint, map_location="cpu"
            )
            if int(model.hparams.n_markers) != spec.n_features:
                raise RuntimeError(
                    f"fold {spec.fold}: contrastive checkpoint has "
                    f"n_markers={int(model.hparams.n_markers)} but the fold has "
                    f"{spec.n_features} markers; refusing mismatched ranking."
                )
            spec.contrastive_model = model


def run(
    *,
    rq3_out: str = RQ3_OUT_DIR,
    rq4_out: str = RQ4_OUT_DIR,
    folds: Sequence[int] = FOLDS,
    ks: Sequence[int] = K_GRID,
    methods: Sequence[str] = METHODS,
    random_replicates: int = RANDOM_REPLICATES,
    random_seed: int = RANDOM_SEED,
    force_rankings: bool = False,
    vae_ckpt_pattern: str = DEFAULT_VAE_CKPT_PATTERN,
    contrastive_ckpt_pattern: str = DEFAULT_CONTRASTIVE_CKPT_PATTERN,
    shap_kwargs: Optional[dict] = None,
    ig_kwargs: Optional[dict] = None,
) -> dict:
    """Production RQ3/RQ4 run over the protocol-v2 development folds."""
    _validate_methods(methods)
    shap_kwargs = dict(shap_kwargs or {})
    ig_kwargs = dict(ig_kwargs or {})

    split = load_split(DATASET_ID, 42, 42, FAM_PATH, out_dir=SPLITS_DIR,
                       cohort_label=COHORT_LABEL)
    ed = GenomicExperimentData.from_plink(split, BED_PATH, bim_path=BIM_PATH)
    n_classes = int(ed.n_classes)
    if n_classes != N_CLASSES_PRIMARY:
        raise RuntimeError(
            f"protocol-v2 primary cohort expects {N_CLASSES_PRIMARY} breeds, "
            f"found {n_classes}."
        )

    rankings_dir = os.path.join(rq3_out, "rankings")

    def _specs() -> Iterable[FoldSpec]:
        for k in folds:
            fd = ed.fold_data(int(k))
            for source in (fd.train_source_index, fd.val_source_index):
                if locked_test_overlap(source, split) != 0:
                    raise RuntimeError(
                        f"locked-test row leaked into fold {k} data preparation."
                    )
            spec = FoldSpec(
                fold=int(k),
                X_train=fd.X_train,
                y_train=fd.y_train,
                X_val=fd.X_val,
                y_val=fd.y_val,
                retained_snp_metadata=fd.retained_snp_metadata,
                retained_snp_indices=fd.retained_snp_indices,
            )
            if "vmgp_shap" in methods:
                spec.vae_checkpoint = _find_fold_checkpoint(
                    vae_ckpt_pattern.format(fold=spec.fold, fold1=spec.fold + 1)
                )
            if "contrastive_ig" in methods:
                spec.contrastive_checkpoint = _find_fold_checkpoint(
                    contrastive_ckpt_pattern.format(
                        fold=spec.fold, fold1=spec.fold + 1
                    )
                )
            yield spec

    def _load(spec: FoldSpec) -> None:
        _load_fold_models(
            spec,
            methods=methods,
            n_classes=n_classes,
            rankings_dir=rankings_dir,
            force_rankings=force_rankings,
            shap_kwargs=shap_kwargs,
            ig_kwargs=ig_kwargs,
        )

    return _run_pipeline(
        _specs(),
        ks=ks,
        methods=methods,
        random_replicates=random_replicates,
        random_seed=random_seed,
        rq3_out=rq3_out,
        rq4_out=rq4_out,
        cache_dir=rq3_out,
        force_rankings=force_rankings,
        n_classes=n_classes,
        shap_kwargs=shap_kwargs,
        ig_kwargs=ig_kwargs,
        smoke=False,
        split_meta=split.meta,
        load_models=_load,
    )


# ---------------------------------------------------------------------------
# Smoke mode (tiny synthetic dry-run, no PLINK/checkpoints)
# ---------------------------------------------------------------------------
def _build_smoke_specs(
    *,
    n_train: int = 60,
    n_val: int = 30,
    n_features: int = 60,
    n_classes: int = 3,
    seed: int = 0,
) -> List[FoldSpec]:
    rng = np.random.default_rng(seed)
    specs: List[FoldSpec] = []
    for fold in range(3):
        X_train = rng.integers(0, 3, size=(n_train, n_features)).astype(np.float32)
        y_train = np.tile(np.arange(n_classes), int(np.ceil(n_train / n_classes)))
        y_train = y_train[:n_train].astype(np.int64)
        rng.shuffle(y_train)

        X_val = rng.integers(0, 3, size=(n_val, n_features)).astype(np.float32)
        y_val = np.tile(np.arange(n_classes), int(np.ceil(n_val / n_classes)))
        y_val = y_val[:n_val].astype(np.int64)
        rng.shuffle(y_val)

        metadata = {
            "raw_snp_index": np.arange(n_features, dtype=np.int64),
            "snp_id": np.array([f"rs{i:05d}" for i in range(n_features)]),
            "chromosome": np.array(["1"] * n_features),
            "position": np.array([str(i * 100) for i in range(n_features)]),
            "allele_1": np.array(["A"] * n_features),
            "allele_2": np.array(["G"] * n_features),
        }

        import torch

        torch.manual_seed(seed + fold)
        vae_model = VMGP_LightningSystem(
            num_snps=n_features,
            num_classes_breed=n_classes,
            latent_dim=8,
            lr=1e-4,
        )
        torch.manual_seed(seed + fold)
        contrastive_model = ContrastiveGeneticModel(
            n_markers=n_features, embedding_dim=3
        )

        specs.append(FoldSpec(
            fold=fold,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            retained_snp_metadata=metadata,
            retained_snp_indices=np.arange(n_features, dtype=np.int64),
            vae_model=vae_model,
            contrastive_model=contrastive_model,
        ))
    return specs


def run_smoke(
    *,
    rq3_out: str = SMOKE_RQ3_OUT_DIR,
    rq4_out: str = SMOKE_RQ4_OUT_DIR,
    ks: Sequence[int] = (10, 20),
    methods: Sequence[str] = METHODS,
    random_replicates: int = RANDOM_REPLICATES,
    random_seed: int = RANDOM_SEED,
    force_rankings: bool = False,
    shap_kwargs: Optional[dict] = None,
    ig_kwargs: Optional[dict] = None,
) -> dict:
    """Tiny synthetic end-to-end dry-run on toy models (never touches PLINK data)."""
    _validate_methods(methods)
    if shap_kwargs is None:
        shap_kwargs = {"n_background": 8, "n_eval": 6, "random_state": 42}
    if ig_kwargs is None:
        ig_kwargs = {"n_steps": 4, "batch_size": 8, "device": "cpu"}

    n_classes = 3
    specs = _build_smoke_specs(
        n_features=max(60, max(int(k) for k in ks) + 1), n_classes=n_classes
    )
    return _run_pipeline(
        specs,
        ks=ks,
        methods=methods,
        random_replicates=random_replicates,
        random_seed=random_seed,
        rq3_out=rq3_out,
        rq4_out=rq4_out,
        cache_dir=rq3_out,
        force_rankings=force_rankings,
        n_classes=n_classes,
        shap_kwargs=dict(shap_kwargs),
        ig_kwargs=dict(ig_kwargs),
        smoke=True,
        split_meta={"synthetic": True, "seed": 0},
        load_models=None,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="tiny synthetic dry-run (default K=[10,20])")
    parser.add_argument("--folds", type=int, nargs="+", default=list(FOLDS))
    parser.add_argument("--ks", type=int, nargs="+", default=None,
                        help=f"panel sizes (default: {list(K_GRID)})")
    parser.add_argument("--methods", nargs="+", choices=list(METHODS),
                        default=list(METHODS))
    parser.add_argument("--random-replicates", type=int, default=RANDOM_REPLICATES)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--force-rankings", action="store_true",
                        help="recompute all rankings instead of reusing the cache")
    parser.add_argument("--rq3-out-dir", default=RQ3_OUT_DIR)
    parser.add_argument("--rq4-out-dir", default=RQ4_OUT_DIR)
    parser.add_argument("--vae-ckpt-pattern", default=DEFAULT_VAE_CKPT_PATTERN)
    parser.add_argument("--contrastive-ckpt-pattern",
                        default=DEFAULT_CONTRASTIVE_CKPT_PATTERN)
    parser.add_argument("--shap-background", type=int, default=100)
    parser.add_argument("--shap-eval", type=int, default=200)
    parser.add_argument("--shap-random-state", type=int, default=42)
    parser.add_argument("--ig-steps", type=int, default=50)
    parser.add_argument("--ig-batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    shap_kwargs = {
        "n_background": int(args.shap_background),
        "n_eval": int(args.shap_eval),
        "random_state": int(args.shap_random_state),
    }
    ig_kwargs = {
        "n_steps": int(args.ig_steps),
        "batch_size": int(args.ig_batch_size),
        "device": str(args.device),
    }

    if args.smoke:
        if args.rq3_out_dir == RQ3_OUT_DIR:
            args.rq3_out_dir = SMOKE_RQ3_OUT_DIR
        if args.rq4_out_dir == RQ4_OUT_DIR:
            args.rq4_out_dir = SMOKE_RQ4_OUT_DIR
        ks = list(args.ks) if args.ks is not None else [10, 20]
        t0 = time.time()
        summary = run_smoke(
            rq3_out=args.rq3_out_dir,
            rq4_out=args.rq4_out_dir,
            ks=ks,
            methods=args.methods,
            random_replicates=int(args.random_replicates),
            random_seed=int(args.random_seed),
            force_rankings=bool(args.force_rankings),
            shap_kwargs=shap_kwargs,
            ig_kwargs=ig_kwargs,
        )
    else:
        ks = list(args.ks) if args.ks is not None else list(K_GRID)
        t0 = time.time()
        summary = run(
            rq3_out=args.rq3_out_dir,
            rq4_out=args.rq4_out_dir,
            folds=tuple(int(k) for k in args.folds),
            ks=ks,
            methods=args.methods,
            random_replicates=int(args.random_replicates),
            random_seed=int(args.random_seed),
            force_rankings=bool(args.force_rankings),
            vae_ckpt_pattern=args.vae_ckpt_pattern,
            contrastive_ckpt_pattern=args.contrastive_ckpt_pattern,
            shap_kwargs=shap_kwargs,
            ig_kwargs=ig_kwargs,
        )

    print(f"Total wall-clock: {time.time() - t0:.1f}s")
    print(f"S_full (mean Macro-F1) = "
          f"{summary['s_full']['macro_f1_mean']:.4f}"
          f"±{summary['s_full']['macro_f1_std']:.4f}")
    for method, per_eps in summary["p_min"].items():
        print(f"  P_min[{method}]: " + ", ".join(
            f"eps={eps} -> {p_min}" for eps, p_min in per_eps.items()
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
