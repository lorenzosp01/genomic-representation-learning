#!/usr/bin/env python
"""RQ4 extended robustness analyses (thesis Section 6.4-bis).

Two train-only analyses on the primary caprine cohort, complementing the
frozen RQ3/RQ4 protocol without touching the locked test:

A) Evaluator sensitivity. Each ranking's panels are re-evaluated with
   additional downstream classifiers (logistic regression, linear SVM,
   gradient boosting) on the same folds, complementing the frozen Random
   Forest evaluator. Tests whether the method ordering is evaluator-specific.

B) Model-native panel evaluation (attribution faithfulness). For every
   panel, the *same architecture* is retrained from scratch on the panel
   markers only, using the frozen RQ1 hyperparameters, and evaluated on the
   fold validation partition: the VMGP breed head for all methods, and the
   contrastive KNN for all methods (train embeddings -> validation
   embeddings). This tests whether the top-ranked markers carry the signal
   for the model family that the neural attributions describe, beyond the
   Random Forest used in the frozen protocol.

   Note: a previous occlusion-style implementation (non-selected markers
   replaced by their training mode) was discarded because it produces
   out-of-distribution inputs and collapses the model to chance; retraining
   on the panel is the faithful analogue of "refit the evaluator on the
   panel" already used for the Random Forest.

C) Extended classical association baseline. A chi-square test of
   independence between genotype (0/1/2) and breed label is computed on the
   fold training partition alone and used as an additional ranking family;
   its panels are evaluated with the same frozen Random Forest downstream
   protocol, and its overlap with the frozen rankings is measured. Outputs
   go to ``results/rq4_extended/association/``. This analysis is caprine
   only (it complements the primary RQ4 comparison).

D) Full-panel references for the alternative evaluators. The frozen
   Random Forest reference (``S_full``) is not transferable to the logistic
   regression / linear SVM / gradient boosting evaluators of analysis A, so
   their own full-post-QC-panel references are computed on the same folds
   and stored under ``results/rq4_extended/evaluator_full_panel/``. They
   allow retention and ``P_min`` to be recomputed per evaluator.

Rankings are loaded from the cached protocol-v2 RQ3 scores; every panel is
built and evaluated on the fold partitions only. Checkpoints produced by
this script are deleted after each run to avoid clutter.

Usage:
    uv run python scripts/run_rq4_extended.py                # A + B + C + D
    uv run python scripts/run_rq4_extended.py --analysis a   # only A
    uv run python scripts/run_rq4_extended.py --analysis b   # only B
    uv run python scripts/run_rq4_extended.py --analysis c   # only C
    uv run python scripts/run_rq4_extended.py --analysis d   # only D
    uv run python scripts/run_rq4_extended.py --analysis b --ks-b 200 --vae-epochs 2 --con-epochs 2   # smoke
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import glob
import json
import os
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genomic.classification_metrics import compute_classification_metrics
from genomic.experiment_data import GenomicExperimentData
from genomic.rq2 import save_json_atomic
from genomic.splitting import load_split

DATASET_ID = "ADAPTmap_genotypeTOP_20160222_full"
FAM_PATH = "data/ADAPTmap_genotypeTOP_20160222_full.fam"
BED_PATH = "data/ADAPTmap_genotypeTOP_20160222_full.bed"
BIM_PATH = "data/ADAPTmap_genotypeTOP_20160222_full.bim"
SPLITS_DIR = "splits"
COHORT_LABEL = "indmiss0p10"

RANKS_DIR = "results/rq3_marker_efficiency/rankings"
OUT_DIR = "results/rq4_extended"
CKPT_DIR = "checkpoints/rq4_extended"
METHODS = ("fst", "random_forest", "vmgp_shap", "contrastive_ig")
K_A = (50, 200, 500, 2000)
K_B = (200, 500, 2000)
K_C = (50, 100, 200, 500, 1000, 2000, 5000)
ASSOCIATION_METHOD = "chi2_association"
GOAT_RQ3_SUMMARY = os.path.join("results", "rq3_marker_efficiency", "summary.json")

RUN_FIELDS = ["analysis", "fold", "method", "K", "evaluator",
              "macro_f1", "balanced_accuracy", "accuracy", "runtime_s"]


def _load_ranked(method: str, fold: int) -> np.ndarray:
    scores_path = os.path.join(RANKS_DIR, f"{method}_fold{fold}_scores.npy")
    meta_path = os.path.join(RANKS_DIR, f"{method}_fold{fold}_meta.json")
    if not (os.path.exists(scores_path) and os.path.exists(meta_path)):
        raise FileNotFoundError(f"missing cached ranking {scores_path}")
    scores = np.load(scores_path)
    with open(meta_path) as f:
        meta = json.load(f)
    if meta["method"] != method or int(meta["fold"]) != fold:
        raise RuntimeError(f"ranking metadata mismatch for {method} fold {fold}")
    return np.argsort(-scores, kind="stable")


def _classifiers():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    return {
        "lr": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=42),
        ),
        "svm": make_pipeline(
            StandardScaler(),
            LinearSVC(C=1.0, dual="auto", max_iter=5000, random_state=42),
        ),
        "gb": HistGradientBoostingClassifier(random_state=42),
    }


def _record(analysis, fold, method, k, evaluator, metrics, runtime):
    return {
        "analysis": analysis,
        "fold": int(fold),
        "method": method,
        "K": int(k),
        "evaluator": evaluator,
        "macro_f1": float(metrics["macro_f1"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "accuracy": float(metrics["accuracy"]),
        "runtime_s": float(runtime),
    }


# ---------------------------------------------------------------------------
# A. Evaluator sensitivity (panels refit with classical classifiers)
# ---------------------------------------------------------------------------
def analysis_a(ed, folds, ks, logger=print):
    labels = np.arange(ed.n_classes, dtype=np.int64)
    clfs = _classifiers()
    records = []
    for k in folds:
        fd = ed.fold_data(k)
        logger(f"[A] fold {k}: train={fd.X_train.shape[0]} val={fd.X_val.shape[0]}")
        rankings = {m: _load_ranked(m, k) for m in METHODS}
        for method in METHODS:
            for panel_k in ks:
                idx = rankings[method][:panel_k]
                X_tr = fd.X_train[:, idx]
                X_v = fd.X_val[:, idx]
                for name, clf in clfs.items():
                    t0 = time.time()
                    clf.fit(X_tr, fd.y_train)
                    metrics = compute_classification_metrics(
                        fd.y_val, clf.predict(X_v), labels=labels
                    )
                    records.append(_record(
                        "A_evaluator", k, method, panel_k, name,
                        metrics, time.time() - t0,
                    ))
                    logger(f"  {method:15s} K={panel_k:5d} {name:3s} "
                           f"MacroF1={metrics['macro_f1']:.4f}")
    return records


# ---------------------------------------------------------------------------
# B. Model-native panel evaluation: same architecture retrained on the panel
# ---------------------------------------------------------------------------
def _panel_fold(fd, idx: np.ndarray):
    """Copy of a FoldData with both matrices restricted to the panel columns."""
    return dataclasses.replace(
        fd,
        X_train=np.ascontiguousarray(fd.X_train[:, idx]),
        X_val=np.ascontiguousarray(fd.X_val[:, idx]),
    )


class _PanelExperimentData:
    """Minimal adapter exposing fold_data(k) restricted to a panel."""

    def __init__(self, ed, idx: np.ndarray):
        self._ed = ed
        self._idx = np.asarray(idx, dtype=np.int64)

    @property
    def split(self):
        return self._ed.split

    @property
    def n_classes(self):
        return self._ed.n_classes

    @property
    def class_names(self):
        return self._ed.class_names

    def fold_data(self, k):
        return _panel_fold(self._ed.fold_data(k), self._idx)


def _vae_config(accelerator: str) -> dict:
    return {
        "batch_size": 64, "alpha": 0.5, "lr": 1e-4, "latent_dim": 96,
        "weight_breed": 1.0, "weight_continent": 0.0, "weight_caseina": 0.0,
        "accelerator": accelerator, "devices": 1,
    }


def _contrastive_config(accelerator: str) -> dict:
    return {
        "embedding_dim": 3, "flip_max": 0.99, "mask_max": 0.99,
        "learning_rate": 0.001, "lr_decay_factor": 0.99,
        "lr_decay_interval": 10, "accelerator": accelerator,
        "devices": 1, "precision": "32-true",
    }


def analysis_b(ed, folds, ks, *, accelerator=None, vae_epochs=200,
               con_epochs=5000, logger=print):
    import torch
    from sklearn.neighbors import KNeighborsClassifier

    from contrastive_learning.evaluation import extract_embeddings
    from contrastive_learning.experiment import run_experiment as run_contrastive
    from contrastive_learning.model import ContrastiveGeneticModel
    from vae.data_module import build_vae_fold
    from vae.experiment import run_experiment as run_vae
    from vae.lightning_module import VMGP_LightningSystem

    if accelerator is None:
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    labels = np.arange(ed.n_classes, dtype=np.int64)
    os.makedirs(os.path.join(OUT_DIR, "confusion_matrices"), exist_ok=True)
    records = []

    def _vae_logits(model, X, batch=512):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, X.shape[0], batch):
                xb = torch.as_tensor(X[i:i + batch], dtype=torch.float32)
                out.append(model(xb)["logits_breed"].numpy())
        return np.concatenate(out)

    for k in folds:
        fd = ed.fold_data(k)

        # Full-input baselines from the trained fold checkpoints (forward only).
        vae_ckpt = sorted(glob.glob(
            f"checkpoints/LatDim96_fold{k}/fold_{k + 1}/epoch=*.ckpt"))
        con_ckpt = sorted(glob.glob(
            f"checkpoints/rq1_fold{k}/fold_{k + 1}/epoch=*.ckpt"))
        if len(vae_ckpt) != 1 or len(con_ckpt) != 1:
            raise RuntimeError(f"fold {k}: checkpoint globbing failed "
                               f"(vae={vae_ckpt}, con={con_ckpt})")
        vae_ref = VMGP_LightningSystem.load_from_checkpoint(
            vae_ckpt[0], map_location="cpu")
        con_ref = ContrastiveGeneticModel.load_from_checkpoint(
            con_ckpt[0], map_location="cpu")

        t0 = time.time()
        m = compute_classification_metrics(
            fd.y_val, _vae_logits(vae_ref, fd.X_val).argmax(axis=1), labels=labels)
        records.append(_record("B_native", k, "full_panel", fd.n_features,
                               "vmgp_retrained", m, time.time() - t0))
        t0 = time.time()
        Z_tr = extract_embeddings(con_ref, fd.X_train)
        Z_v = extract_embeddings(con_ref, fd.X_val)
        knn = KNeighborsClassifier(n_neighbors=3).fit(Z_tr, fd.y_train)
        m = compute_classification_metrics(fd.y_val, knn.predict(Z_v), labels=labels)
        records.append(_record("B_native", k, "full_panel", fd.n_features,
                               "contrastive_retrained", m, time.time() - t0))
        logger(f"[B] fold {k}: full-input baseline "
               f"VMGP={records[-2]['macro_f1']:.4f} "
               f"KNN={records[-1]['macro_f1']:.4f}")
        del vae_ref, con_ref
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for method in METHODS:
            ranked = _load_ranked(method, k)
            for panel_k in ks:
                idx = ranked[:panel_k]
                pfd = _panel_fold(fd, idx)
                tag = f"{method}_K{panel_k}_fold{k}"

                # -- VMGP retrained on the panel --
                vfd = build_vae_fold(
                    pfd, class_names_breed=list(ed.class_names),
                    metadata_path=None, attitudine_mapping=None,
                    use_continent=False, use_caseina=False, use_attitudine=False,
                )
                name_v = f"b_vae_{tag}"
                t0 = time.time()
                res = run_vae(
                    name=name_v, experiment_data=None,
                    config=_vae_config(accelerator),
                    max_epochs=int(vae_epochs), classifier_config="breed_only",
                    balanced=False, cap_samples=False, folds=[vfd],
                    checkpoint_dir=CKPT_DIR,
                    cm_dir=os.path.join(OUT_DIR, "confusion_matrices"),
                )
                m_vae = {
                    "macro_f1": float(res["Macro_F1 (mean)"]),
                    "balanced_accuracy": float(res["Balanced_Accuracy (mean)"]),
                    "accuracy": float(res["Accuracy (mean)"]),
                }
                records.append(_record("B_native", k, method, panel_k,
                                       "vmgp_retrained", m_vae, time.time() - t0))
                shutil.rmtree(os.path.join(CKPT_DIR, name_v), ignore_errors=True)

                # -- Contrastive retrained on the panel --
                name_c = f"b_con_{tag}"
                t0 = time.time()
                res = run_contrastive(
                    name=name_c,
                    experiment_data=_PanelExperimentData(ed, idx),
                    config=_contrastive_config(accelerator),
                    max_epochs=int(con_epochs),
                    folds=[int(k)],
                    checkpoint_dir=CKPT_DIR,
                )
                m_con = {
                    "macro_f1": float(res["Macro_F1 (mean)"]),
                    "balanced_accuracy": float(res["Balanced_Accuracy (mean)"]),
                    "accuracy": float(res["Accuracy (mean)"]),
                }
                records.append(_record("B_native", k, method, panel_k,
                                       "contrastive_retrained", m_con,
                                       time.time() - t0))
                shutil.rmtree(os.path.join(CKPT_DIR, name_c), ignore_errors=True)
                logger(f"  {method:15s} K={panel_k:5d} "
                       f"VMGP={m_vae['macro_f1']:.4f} "
                       f"KNN={m_con['macro_f1']:.4f}")
    return records


# ---------------------------------------------------------------------------
# C. Extended classical association baseline (chi-square) under the frozen
#    Random Forest downstream protocol
# ---------------------------------------------------------------------------
def analysis_c(ed, folds, ks, out_subdir, logger=print):
    """Chi-square association ranking evaluated with the frozen RF protocol.

    The ranking is computed from ``(X_train, y_train)`` only; panels are
    evaluated on the fold validation partition with the same downstream
    RandomForestClassifier(200, seed 42) used by RQ3/RQ4. The summary stores
    the operational minimum panel sizes against the frozen caprine
    ``S_full`` and the mean-over-folds Jaccard overlap with the four frozen
    ranking families.
    """
    from genomic.panel_evaluation import (
        evaluate_panel,
        minimum_panel_size,
        retention_threshold,
    )
    from genomic.ranking import compute_marker_ranking

    records = []
    ranked_by_fold = {}
    for k in folds:
        fd = ed.fold_data(k)
        t0 = time.time()
        result = compute_marker_ranking(ASSOCIATION_METHOD, fd.X_train, fd.y_train)
        ranked = result.ranked_indices
        ranked_by_fold[int(k)] = ranked
        logger(f"[C] fold {k}: association ranking in {time.time() - t0:.1f}s "
               f"(top score={result.scores[ranked[0]]:.1f})")
        for panel_k in ks:
            t0 = time.time()
            metrics = evaluate_panel(
                fd.X_train, fd.y_train, fd.X_val, fd.y_val,
                ranked[:panel_k], n_classes=ed.n_classes,
            )
            records.append(_record(
                "C_association", k, ASSOCIATION_METHOD, panel_k, "rf",
                metrics, time.time() - t0,
            ))
            logger(f"  {ASSOCIATION_METHOD} K={panel_k:5d} rf "
                   f"MacroF1={metrics['macro_f1']:.4f}")

    aggregates = _aggregate(records)
    with open(GOAT_RQ3_SUMMARY) as f:
        frozen_summary = json.load(f)
    s_full = float(frozen_summary["s_full"]["macro_f1_mean"])
    n_markers = float(np.mean(frozen_summary["s_full"]["n_markers"]))
    k_grid = sorted(int(k) for k in ks)
    p_min = {
        f"{eps:g}": minimum_panel_size(
            ASSOCIATION_METHOD, aggregates, s_full, epsilon=eps, k_grid=k_grid
        )
        for eps in (0.02, 0.05)
    }
    thresholds = {
        f"{eps:g}": retention_threshold(s_full, eps) for eps in (0.02, 0.05)
    }
    jaccard = {}
    for panel_k in (50, 200, 500):
        row = {}
        for method in METHODS:
            vals = []
            for k in folds:
                other_scores = np.load(
                    os.path.join(RANKS_DIR, f"{method}_fold{k}_scores.npy")
                )
                other = np.argsort(-other_scores, kind="stable")
                a = set(ranked_by_fold[int(k)][:panel_k].tolist())
                b = set(other[:panel_k].tolist())
                union = a | b
                vals.append(len(a & b) / len(union) if union else 0.0)
            row[method] = 100.0 * float(np.mean(vals))
        jaccard[f"K={panel_k}"] = row
    chance = {
        f"K={panel_k}": 100.0 * panel_k / (2.0 * n_markers - panel_k)
        for panel_k in (50, 200, 500)
    }

    os.makedirs(out_subdir, exist_ok=True)
    _write_csv_atomic(records, os.path.join(out_subdir, "run_metrics.csv"),
                      RUN_FIELDS)
    agg_fields = list(aggregates[0].keys()) if aggregates else []
    _write_csv_atomic(aggregates,
                      os.path.join(out_subdir, "aggregate_metrics.csv"),
                      agg_fields)
    save_json_atomic({
        "experiment": "rq4_extended_association_baseline",
        "method": ASSOCIATION_METHOD,
        "description": "chi-square genotype x breed association test (train-only)",
        "folds": [int(k) for k in folds],
        "panel_sizes": k_grid,
        "s_full": s_full,
        "retention_thresholds": thresholds,
        "p_min": p_min,
        "jaccard_vs_frozen_mean_over_folds_pct": jaccard,
        "chance_jaccard_pct": chance,
        "locked_test_accessed": False,
    }, os.path.join(out_subdir, "summary.json"))

    logger(f"[C] p_min: {p_min} | thresholds: {thresholds}")
    for label, row in jaccard.items():
        logger(f"[C] Jaccard {label}: "
               + ", ".join(f"{m}={v:.1f}%" for m, v in row.items()))
    return records


# ---------------------------------------------------------------------------
# D. Full-panel references for the alternative evaluators (companion of A)
# ---------------------------------------------------------------------------
def analysis_d(ed, folds, out_subdir, logger=print):
    """Full post-QC panel references for the frozen and alternative evaluators.

    The frozen ``S_full`` belongs to the Random Forest evaluator; retention
    and ``P_min`` under the logistic regression / linear SVM / gradient
    boosting evaluators of analysis A need their own references on the same
    development folds. Every fit uses the complete post-QC marker axis of the
    fold and the validation partition for scoring.
    """
    from sklearn.ensemble import RandomForestClassifier

    from genomic.panel_evaluation import RF_N_ESTIMATORS, RF_RANDOM_STATE

    labels = np.arange(ed.n_classes, dtype=np.int64)
    evaluators = {
        "rf": RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS, random_state=RF_RANDOM_STATE,
            n_jobs=-1,
        ),
        **_classifiers(),
    }
    records = []
    for k in folds:
        fd = ed.fold_data(k)
        for name, clf in evaluators.items():
            t0 = time.time()
            clf.fit(fd.X_train, fd.y_train)
            metrics = compute_classification_metrics(
                fd.y_val, clf.predict(fd.X_val), labels=labels
            )
            records.append(_record("D_full", k, "full_panel", fd.n_features,
                                   name, metrics, time.time() - t0))
            logger(f"[D] fold {k} {name:3s} n_markers={fd.n_features} "
                   f"MacroF1={metrics['macro_f1']:.4f} "
                   f"({time.time() - t0:.1f}s)")

    per_evaluator = {}
    for name in evaluators:
        recs = [r for r in records if r["evaluator"] == name]
        vals = np.asarray([r["macro_f1"] for r in recs], dtype=np.float64)
        per_evaluator[name] = {
            "macro_f1_mean": float(vals.mean()),
            "macro_f1_std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
            "balanced_accuracy_mean": float(
                np.mean([r["balanced_accuracy"] for r in recs])
            ),
            "accuracy_mean": float(np.mean([r["accuracy"] for r in recs])),
            "per_fold": [
                {"fold": r["fold"], "n_markers": r["K"],
                 "macro_f1": r["macro_f1"]}
                for r in recs
            ],
        }
    os.makedirs(out_subdir, exist_ok=True)
    _write_csv_atomic(records, os.path.join(out_subdir, "run_metrics.csv"),
                      RUN_FIELDS)
    save_json_atomic({
        "experiment": "rq4_extended_evaluator_full_panel",
        "description": (
            "Full post-QC panel references for the frozen Random Forest and "
            "the alternative evaluators of analysis A, computed on the same "
            "development folds with the same preprocessing."
        ),
        "evaluators": per_evaluator,
        "locked_test_accessed": False,
    }, os.path.join(out_subdir, "summary.json"))
    for name, stats in per_evaluator.items():
        logger(f"[D] {name:3s} S_full = {stats['macro_f1_mean']:.4f} "
               f"± {stats['macro_f1_std']:.4f}")
    return records


# ---------------------------------------------------------------------------
# Aggregation / persistence
# ---------------------------------------------------------------------------
def _aggregate(records, group_keys=("analysis", "evaluator", "method", "K")):
    groups = {}
    for r in records:
        key = tuple(r[k] for k in group_keys)
        groups.setdefault(key, []).append(r)
    rows = []
    for key, recs in sorted(groups.items(), key=lambda kv: [str(x) for x in kv[0]]):
        row = dict(zip(group_keys, key))
        row["n_observations"] = len(recs)
        for metric in ("macro_f1", "balanced_accuracy", "accuracy"):
            vals = np.asarray([r[metric] for r in recs], dtype=np.float64)
            ddof = 1 if vals.size > 1 else 0
            row[f"{metric}_mean"] = float(vals.mean())
            row[f"{metric}_std"] = float(vals.std(ddof=ddof))
        rows.append(row)
    return rows


def _write_csv_atomic(rows, path, fieldnames):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    os.replace(tmp, path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", choices=["a", "b", "c", "d", "all"], default="all")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--ks-a", type=int, nargs="+", default=list(K_A))
    parser.add_argument("--ks-b", type=int, nargs="+", default=list(K_B))
    parser.add_argument("--ks-c", type=int, nargs="+", default=list(K_C))
    parser.add_argument("--vae-epochs", type=int, default=200)
    parser.add_argument("--con-epochs", type=int, default=5000)
    parser.add_argument("--device", default=None,
                        help="accelerator for the retraining ('gpu'/'cpu')")
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args(argv)

    split = load_split(DATASET_ID, 42, 42, FAM_PATH, out_dir=SPLITS_DIR,
                       cohort_label=COHORT_LABEL)
    ed = GenomicExperimentData.from_plink(split, BED_PATH, bim_path=BIM_PATH)
    print(f"cohort: {ed.n_samples} rows, {ed.n_classes} breeds, "
          f"folds={args.folds}")

    ks_a = tuple(int(k) for k in args.ks_a)
    ks_b = tuple(int(k) for k in args.ks_b)
    ks_c = tuple(int(k) for k in args.ks_c)
    records = []
    if args.analysis in ("a", "all"):
        records.extend(analysis_a(ed, args.folds, ks_a))
    if args.analysis in ("b", "all"):
        records.extend(analysis_b(
            ed, args.folds, ks_b,
            accelerator=args.device,
            vae_epochs=args.vae_epochs, con_epochs=args.con_epochs,
        ))
    if args.analysis in ("c", "all"):
        analysis_c(ed, args.folds, ks_c,
                   os.path.join(args.out_dir, "association"))
    if args.analysis in ("d", "all"):
        analysis_d(ed, args.folds,
                   os.path.join(args.out_dir, "evaluator_full_panel"))

    if not records:
        print("No A/B records produced; analyses C/D outputs (if requested) "
              f"are under {args.out_dir}.")
        return 0

    aggregates = _aggregate(records)
    os.makedirs(args.out_dir, exist_ok=True)
    run_path = os.path.join(args.out_dir, "run_metrics.csv")
    _write_csv_atomic(records, run_path, RUN_FIELDS)
    agg_fields = list(aggregates[0].keys()) if aggregates else []
    _write_csv_atomic(aggregates,
                      os.path.join(args.out_dir, "aggregate_metrics.csv"),
                      agg_fields)
    save_json_atomic({
        "experiment": "rq4_extended_robustness",
        "analysis": args.analysis,
        "folds": [int(k) for k in args.folds],
        "ks_evaluator_sensitivity": [int(k) for k in ks_a],
        "ks_native": [int(k) for k in ks_b],
        "methods": list(METHODS),
        "evaluators": ["rf (from RQ3)", "lr", "svm", "gb",
                       "vmgp_retrained", "contrastive_retrained"],
        "vae_epochs": int(args.vae_epochs),
        "con_epochs": int(args.con_epochs),
        "aggregates": aggregates,
        "locked_test_accessed": False,
    }, os.path.join(args.out_dir, "summary.json"))

    print(f"\nSaved: {run_path}")
    print(f"Saved: {os.path.join(args.out_dir, 'aggregate_metrics.csv')}")
    print(f"Saved: {os.path.join(args.out_dir, 'summary.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
