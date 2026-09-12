"""Model-neutral RQ2 sample-efficiency infrastructure.

Implements the frozen RQ2 protocol on top of the development-only raw-data API
(:meth:`GenomicExperimentData.development_raw_data`): the fixed 15-breed cohort,
the nested per-breed sampling (``S5 ⊂ S10 ⊂ … ⊂ S30``), fresh preprocessing on
each ``S_N``, the 15-class evaluation sample, aggregation over the 15
observations per N, and the frozen ``epsilon_N = 0.02`` minimum-sample rule.

Locked-test rows are never received, materialized or indexed here: this module
operates exclusively on :class:`~genomic.experiment_data.DevelopmentRawData`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .experiment_data import DevelopmentRawData
from .labels import CanonicalLabelMapper
from .preprocessing import GenomicPreprocessor

# ---------------------------------------------------------------------------
# Frozen RQ2 protocol constants
# ---------------------------------------------------------------------------
RQ2_BREEDS: List[str] = [
    "ABR", "ALP", "ANG", "BOE", "BRK", "BUR", "CRE", "LNR",
    "NBN", "OSS", "RAN", "SAA", "SEA", "SID", "WAD",
]
RQ2_FOLDS: Tuple[int, ...] = (0, 1, 2)
RQ2_SEEDS: Tuple[int, ...] = (42, 43, 44, 45, 46)
RQ2_N_GRID: Tuple[int, ...] = (5, 10, 15, 20, 25, 30)
RQ2_MAX_N: int = 30
EPSILON_N: float = 0.02

REQUIRED_RECORD_KEYS = (
    "fold", "replicate_seed", "N", "macro_f1", "balanced_accuracy", "accuracy",
    "per_class_recall", "confusion_matrix", "retained_snp_count",
    "best_epoch", "completed_epochs", "early_stopping_triggered",
    "runtime_s", "class_names",
)


def assert_rq2_breeds(breeds: Sequence[str]) -> None:
    """Assert the cohort is exactly the frozen 15-breed RQ2 list."""
    got = list(breeds)
    if got != RQ2_BREEDS:
        raise AssertionError(
            f"RQ2 cohort mismatch.\n expected: {RQ2_BREEDS}\n got:      {got}"
        )


def breed_availability_table(split) -> List[dict]:
    """Per-RQ2-breed training count in each fold + the fold minimum."""
    counts = split.train_counts_per_breed()
    rows = []
    for b in RQ2_BREEDS:
        if b not in counts:
            raise ValueError(f"RQ2 breed {b!r} not present in the split.")
        fs = [int(c) for c in counts[b]]
        rows.append({
            "breed": b,
            "fold0_train_count": fs[0],
            "fold1_train_count": fs[1],
            "fold2_train_count": fs[2],
            "min_train_count": min(fs),
        })
    return rows


# ---------------------------------------------------------------------------
# Fold partitions (development-only, restricted to the 15 RQ2 breeds)
# ---------------------------------------------------------------------------
def rq2_fold_partitions(dev_raw: DevelopmentRawData, fold: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (train_positions, val_positions) for one fold.

    Positions index ``dev_raw.X_dev``. Both partitions are restricted to the 15
    RQ2 breeds. Validation is the UNCHANGED original fold-validation subset; it
    is never resampled.
    """
    if not 0 <= fold < dev_raw.n_folds:
        raise ValueError(f"fold {fold} out of range [0, {dev_raw.n_folds})")
    in_rq2 = np.isin(dev_raw.dev_breed, RQ2_BREEDS)
    train_pos = np.flatnonzero((dev_raw.dev_fold != fold) & in_rq2)
    val_pos = np.flatnonzero((dev_raw.dev_fold == fold) & in_rq2)
    return train_pos, val_pos


def sample_nested_rq2(dev_raw: DevelopmentRawData, *, fold: int, seed: int,
                      ns: Optional[Sequence[int]] = None,
                      max_n: int = RQ2_MAX_N
                      ) -> Tuple[Dict[int, np.ndarray], Dict[str, np.ndarray]]:
    """Deterministic nested per-breed sampling for one (fold, replicate).

    A fold-specific RNG is derived via ``SeedSequence([seed, fold])``. For each
    RQ2 breed (fixed canonical order) one permutation of the fold-training pool
    is drawn and the first ``max_n`` individuals are kept; ``S_N`` is the union
    of the first ``N`` of each breed (so ``S5 ⊂ … ⊂ S30``).

    Returns ``(sampled_positions_by_n, per_breed_order)`` where positions index
    ``dev_raw.X_dev``.
    """
    ns = list(RQ2_N_GRID if ns is None else ns)
    if any(n < 1 or n > max_n for n in ns):
        raise ValueError(f"sample sizes must be in [1, {max_n}], got {ns}")

    train_pos, _ = rq2_fold_partitions(dev_raw, fold)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(fold)]))

    per_breed: Dict[str, np.ndarray] = {}
    for b in RQ2_BREEDS:
        b_pos = train_pos[dev_raw.dev_breed[train_pos] == b]
        if len(b_pos) < max_n:
            raise ValueError(
                f"breed {b} has only {len(b_pos)} fold-{fold} training rows "
                f"(< {max_n})"
            )
        per_breed[b] = rng.permutation(b_pos)[:max_n]

    sampled: Dict[int, np.ndarray] = {}
    for n in ns:
        parts = [per_breed[b][:n] for b in RQ2_BREEDS]
        sampled[int(n)] = np.sort(np.concatenate(parts))
    return sampled, per_breed


# ---------------------------------------------------------------------------
# Fresh-preprocessor sample assembly
# ---------------------------------------------------------------------------
@dataclass
class RQ2PreprocessedSample:
    """One RQ2 run's preprocessed training/validation matrices (15 classes)."""

    fold: int
    replicate_seed: int
    n: int
    preprocessor: GenomicPreprocessor
    class_names: List[str]
    n_classes: int
    X_train: np.ndarray
    y_train: np.ndarray
    train_positions: np.ndarray
    train_source_index: np.ndarray
    train_breed: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    val_positions: np.ndarray
    val_source_index: np.ndarray
    val_breed: np.ndarray

    @property
    def n_features(self) -> int:
        return int(self.X_train.shape[1])

    @property
    def n_train(self) -> int:
        return int(self.X_train.shape[0])

    @property
    def n_val(self) -> int:
        return int(self.X_val.shape[0])


def preprocessing_provenance(pre: GenomicPreprocessor) -> dict:
    return {
        "marker_missingness_threshold": pre.marker_missingness_threshold,
        "maf_threshold": pre.maf_threshold,
        "ld_window": pre.ld_window,
        "ld_r2_threshold": pre.ld_r2_threshold,
        "n_features_in": int(pre.n_features_in_),
        "n_retained_snps": int(len(pre.retained_snp_indices_)),
    }


def build_rq2_sample(dev_raw: DevelopmentRawData, *,
                     fold: int, seed: int, n: int,
                     train_positions: np.ndarray, val_positions: np.ndarray,
                     mapper: CanonicalLabelMapper,
                     preprocessor_kwargs: Optional[dict] = None) -> RQ2PreprocessedSample:
    """Fit a FRESH preprocessor on ``S_N`` only and transform S_N + fixed val.

    Never fits on S30 and reuses for smaller N; each call is independent.
    """
    pre = GenomicPreprocessor(**(preprocessor_kwargs or {}))
    pre.fit(dev_raw.X_dev[train_positions])

    X_train = pre.transform(dev_raw.X_dev[train_positions])
    X_val = pre.transform(dev_raw.X_dev[val_positions])
    y_train = mapper.encode(dev_raw.dev_breed[train_positions])
    y_val = mapper.encode(dev_raw.dev_breed[val_positions])

    return RQ2PreprocessedSample(
        fold=int(fold), replicate_seed=int(seed), n=int(n),
        preprocessor=pre,
        class_names=list(mapper.classes), n_classes=mapper.n_classes,
        X_train=X_train, y_train=y_train,
        train_positions=np.asarray(train_positions),
        train_source_index=dev_raw.dev_source_index[train_positions],
        train_breed=dev_raw.dev_breed[train_positions],
        X_val=X_val, y_val=y_val,
        val_positions=np.asarray(val_positions),
        val_source_index=dev_raw.dev_source_index[val_positions],
        val_breed=dev_raw.dev_breed[val_positions],
    )


# ---------------------------------------------------------------------------
# Aggregation + frozen minimum-sample rule
# ---------------------------------------------------------------------------
def _require_obs(records: Sequence[dict], n: int, folds=RQ2_FOLDS, seeds=RQ2_SEEDS) -> List[dict]:
    recs = [r for r in records if int(r["N"]) == int(n)]
    keys = {(int(r["fold"]), int(r["replicate_seed"])) for r in recs}
    expected = {(int(f), int(s)) for f in folds for s in seeds}
    if len(recs) != len(expected) or keys != expected:
        raise ValueError(
            f"N={n}: expected exactly {len(expected)} observations "
            f"(folds {tuple(folds)} x seeds {tuple(seeds)}), got {len(recs)} "
            f"(keys={sorted(keys)})"
        )
    return recs


def aggregate_rq2(records: Sequence[dict], *, ns: Sequence[int] = RQ2_N_GRID,
                  folds=RQ2_FOLDS, seeds=RQ2_SEEDS,
                  epsilon: float = EPSILON_N) -> dict:
    """Aggregate exactly 15 observations per N and compute the frozen N_min.

    Raises if any N lacks a complete 3 folds x 5 seeds set, if any metric is
    non-finite, or if the N=max reference aggregate is unavailable/invalid.
    ``N_min`` is guaranteed to be one of ``ns`` (N=max is S_ref by definition).
    """
    ns = [int(n) for n in ns]
    aggregates = []
    for n in ns:
        recs = _require_obs(records, n, folds=folds, seeds=seeds)
        row = {"N": n, "n_observations": len(recs)}
        for field, prefix in (("macro_f1", "macro_f1"),
                              ("balanced_accuracy", "balanced_accuracy"),
                              ("accuracy", "accuracy")):
            vals = np.asarray([float(r[field]) for r in recs], dtype=np.float64)
            if not np.all(np.isfinite(vals)):
                raise ValueError(f"N={n}: non-finite {field} values {vals.tolist()}")
            ddof = 1 if vals.size > 1 else 0
            row[f"{prefix}_mean"] = float(vals.mean())
            row[f"{prefix}_std"] = float(vals.std(ddof=ddof))
        aggregates.append(row)

    aggregates.sort(key=lambda r: r["N"])
    max_n = max(ns)
    ref = [r for r in aggregates if r["N"] == max_n]
    if len(ref) != 1:
        raise ValueError(f"reference aggregate for N={max_n} unavailable")
    s_ref = ref[0]["macro_f1_mean"]
    if not np.isfinite(s_ref):
        raise ValueError(f"reference mean Macro-F1 (N={max_n}) is non-finite")

    threshold = (1.0 - float(epsilon)) * s_ref
    qualifying = [r["N"] for r in aggregates if r["macro_f1_mean"] >= threshold]
    if not qualifying:
        raise RuntimeError(
            "No N satisfies the minimum-sample criterion; this is mathematically "
            "impossible because N=max is S_ref. Refusing to return a null N_min."
        )
    n_min = int(min(qualifying))
    if n_min not in ns:
        raise ValueError(f"computed N_min={n_min} is not in the frozen grid {ns}")

    return {
        "aggregates": aggregates,
        "epsilon_N": float(epsilon),
        "S_ref": float(s_ref),
        "threshold": float(threshold),
        "N_min": n_min,
        "reference_N": max_n,
    }


# ---------------------------------------------------------------------------
# Atomic persistence helpers
# ---------------------------------------------------------------------------
def save_json_atomic(obj, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, path)
    return path


def load_json(path: str):
    with open(path) as f:
        return json.load(f)


def validate_run_record(record: dict) -> None:
    """Raise if a run record is incomplete/corrupt (never silently accept)."""
    missing = [k for k in REQUIRED_RECORD_KEYS if k not in record]
    if missing:
        raise ValueError(f"run record missing keys: {missing}")
    if list(record["class_names"]) != RQ2_BREEDS:
        raise ValueError("run record class_names are not the canonical 15 RQ2 breeds")
    if len(record["per_class_recall"]) != len(RQ2_BREEDS):
        raise ValueError("run record per_class_recall has the wrong length")
    cm = np.asarray(record["confusion_matrix"])
    if cm.shape != (len(RQ2_BREEDS), len(RQ2_BREEDS)):
        raise ValueError(f"run record confusion_matrix shape {cm.shape} != (15, 15)")
    for field in ("macro_f1", "balanced_accuracy", "accuracy"):
        if not np.isfinite(float(record[field])):
            raise ValueError(f"run record {field} is non-finite")
    for field in ("fold", "replicate_seed", "N", "best_epoch", "completed_epochs"):
        if int(record[field]) < 0:
            raise ValueError(f"run record {field} is invalid")
