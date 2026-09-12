"""PRIMARY RQ2 sample-efficiency runner (VMGP only).

For every (fold, replicate seed, N) the runner:
  1. derives the nested per-breed sample S_N from the development-only raw data;
  2. fits a FRESH :class:`GenomicPreprocessor` on S_N only and transforms S_N and
     the unchanged 15-breed validation partition;
  3. trains the frozen RQ1 VMGP configuration (latent_dim=96, alpha=0.5,
     MSE + 1e-4*KL, Adam 1e-4, max_epochs=200, EarlyStopping patience=15,
     16-mixed, best-val-loss checkpoint restore) with the RQ2-specific training
     DataLoader semantics (``drop_last=False``);
  4. persists metrics / CM / per-class recall / provenance atomically and
     deletes the temporary per-run checkpoint by default.

The locked test is never received or materialized: everything operates on
:class:`~genomic.experiment_data.DevelopmentRawData`.
"""

from __future__ import annotations

import os
import shutil
import time
from typing import Dict, Optional, Sequence

import numpy as np
import torch

from genomic.labels import CanonicalLabelMapper
from genomic.rq2 import (
    RQ2_BREEDS,
    RQ2_FOLDS,
    RQ2_N_GRID,
    RQ2_SEEDS,
    RQ2PreprocessedSample,
    aggregate_rq2,
    build_rq2_sample,
    load_json,
    preprocessing_provenance,
    rq2_fold_partitions,
    sample_nested_rq2,
    save_json_atomic,
    validate_run_record,
)
from genomic.training_diagnostics import ConvergenceHistoryCallback

from .data_module import VAEFoldData
from .experiment import run_experiment

RQ2_LATENT_DIM = 96
RQ2_MAX_EPOCHS = 200
RQ2_PRECISION = "16-mixed"


def rq2_vae_config(*, accelerator: Optional[str] = None, devices: int = 1) -> dict:
    """Frozen selected RQ1 VMGP configuration (no hyperparameter search)."""
    if accelerator is None:
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    return {
        "batch_size": 64,
        "alpha": 0.5,
        "lr": 1e-4,
        "latent_dim": RQ2_LATENT_DIM,
        "weight_breed": 1.0,
        "weight_continent": 0.0,
        "weight_caseina": 0.0,
        "accelerator": accelerator,
        "devices": devices,
    }


def build_rq2_vae_fold_data(sample: RQ2PreprocessedSample) -> VAEFoldData:
    """Adapt an :class:`RQ2PreprocessedSample` to the frozen VAE fold container."""
    n_tr, n_va = sample.n_train, sample.n_val
    return VAEFoldData(
        fold=sample.fold,
        preprocessor=sample.preprocessor,
        num_snps=sample.n_features,
        num_classes_breed=sample.n_classes,
        class_names_breed=list(sample.class_names),
        num_classes_continent=0,
        num_classes_caseina=0,
        num_classes_attitudine=0,
        class_names_continent=[],
        class_names_caseina=[],
        class_names_attitudine=[],
        X_train=sample.X_train,
        y_breed_train=sample.y_train,
        y_continent_train=np.full(n_tr, -1, dtype=np.int64),
        y_caseina_train=np.full(n_tr, -1, dtype=np.int64),
        y_attitudine_train=np.full(n_tr, -1, dtype=np.int64),
        X_val=sample.X_val,
        y_breed_val=sample.y_val,
        y_continent_val=np.full(n_va, -1, dtype=np.int64),
        y_caseina_val=np.full(n_va, -1, dtype=np.int64),
        y_attitudine_val=np.full(n_va, -1, dtype=np.int64),
        train_source_index=sample.train_source_index,
        val_source_index=sample.val_source_index,
        train_breed=sample.train_breed,
        val_breed=sample.val_breed,
    )


def _run_name(fold: int, seed: int, n: int) -> str:
    return f"rq2_fold{fold}_seed{seed}_N{n}"


def _cleanup_side_effects(name: str, ckpt_dir: str, keep_checkpoints: bool) -> None:
    """Remove the temporary checkpoint (unless kept) and run_experiment plots/CSV."""
    if not keep_checkpoints:
        shutil.rmtree(os.path.join(ckpt_dir, name), ignore_errors=True)
    for path in (f"plot_vae_{name}_kfold.png",
                 os.path.join("results", "confusion_matrices", f"cm_{name}.csv")):
        if os.path.exists(path):
            os.remove(path)


def run_rq2_single(dev_raw, mapper: CanonicalLabelMapper, *,
                   fold: int, seed: int, n: int,
                   train_positions: np.ndarray, val_positions: np.ndarray,
                   out_dir: str, ckpt_dir: str = "checkpoints",
                   keep_checkpoints: bool = False,
                   run_experiment_fn=None, accelerator: Optional[str] = None,
                   devices: int = 1) -> dict:
    """Run one RQ2 (fold, seed, N) run, persist it atomically, clean up."""
    run_experiment_fn = run_experiment_fn or run_experiment

    sample = build_rq2_sample(
        dev_raw, fold=fold, seed=seed, n=n,
        train_positions=train_positions, val_positions=val_positions,
        mapper=mapper,
    )
    vfd = build_rq2_vae_fold_data(sample)

    cb = ConvergenceHistoryCallback()
    name = _run_name(fold, seed, n)
    cfg = rq2_vae_config(accelerator=accelerator, devices=devices)

    t0 = time.time()
    res = run_experiment_fn(
        name=name,
        experiment_data=None,          # folds are supplied explicitly
        config=cfg,
        max_epochs=RQ2_MAX_EPOCHS,
        classifier_config="breed_only",
        balanced=False,
        cap_samples=False,
        folds=[vfd],
        history_callback=cb,
        drop_last=False,               # RQ2-specific intentional deviation
    )
    runtime = time.time() - t0

    if cb.stopped_epoch is None:
        raise RuntimeError(
            f"{name}: convergence history did not record completed epochs; "
            "refusing to persist an incomplete RQ2 run record."
        )

    record = {
        "model": "vae",
        "latent_dim": RQ2_LATENT_DIM,
        "fold": int(fold),
        "replicate_seed": int(seed),
        "N": int(n),
        "macro_f1": float(res["Macro_F1 (mean)"]),
        "balanced_accuracy": float(res["Balanced_Accuracy (mean)"]),
        "accuracy": float(res["Accuracy (mean)"]),
        "per_class_recall": [float(x) for x in np.asarray(res["Per_Class_Recall (mean)"]).ravel()],
        "confusion_matrix": np.asarray(res["Confusion_Matrix"], dtype=np.int64).tolist(),
        "class_names": list(sample.class_names),
        "n_classes": int(sample.n_classes),
        "retained_snp_count": int(res["Num SNPs"]),
        "best_epoch": int(res["Best_Epochs"][0]),
        "completed_epochs": int(cb.stopped_epoch),
        "early_stopping_triggered": bool(cb.early_stopping_triggered),
        "max_epochs": RQ2_MAX_EPOCHS,
        "runtime_s": float(runtime),
        "n_train": int(sample.n_train),
        "n_val": int(sample.n_val),
        "train_source_index": [int(x) for x in sample.train_source_index],
        "val_source_index": [int(x) for x in sample.val_source_index],
        "preprocessing": preprocessing_provenance(sample.preprocessor),
        "checkpoint_kept": bool(keep_checkpoints),
    }
    validate_run_record(record)

    base = f"fold{fold}_seed{seed}_N{n}"
    save_json_atomic(
        {"class_names": record["class_names"], "confusion_matrix": record["confusion_matrix"]},
        os.path.join(out_dir, "confusion_matrices", f"{base}.json"),
    )
    save_json_atomic(
        {"class_names": record["class_names"], "per_class_recall": record["per_class_recall"]},
        os.path.join(out_dir, "per_class_recall", f"{base}.json"),
    )
    # Authoritative record written last (atomic) so its presence == complete run.
    save_json_atomic(record, os.path.join(out_dir, "runs", f"{base}.json"))

    _cleanup_side_effects(name, ckpt_dir, keep_checkpoints)
    return record


def load_completed_runs(out_dir: str) -> Dict[str, dict]:
    """Load and validate previously completed runs; raise on partial/corrupt."""
    runs_dir = os.path.join(out_dir, "runs")
    completed: Dict[str, dict] = {}
    if not os.path.isdir(runs_dir):
        return completed
    for fn in sorted(os.listdir(runs_dir)):
        if not fn.endswith(".json"):
            continue
        path = os.path.join(runs_dir, fn)
        record = load_json(path)                      # raises on corrupt JSON
        validate_run_record(record)                   # raises on invalid record
        key = fn[:-5]
        expected = f"fold{record['fold']}_seed{record['replicate_seed']}_N{record['N']}"
        if key != expected:
            raise ValueError(f"run record filename mismatch: {fn} vs {expected}")
        for sub in ("confusion_matrices", "per_class_recall"):
            if not os.path.exists(os.path.join(out_dir, sub, fn)):
                raise FileNotFoundError(
                    f"incomplete run artifact: {fn} missing {sub}/{fn}"
                )
        completed[key] = record
    return completed


def run_rq2_experiment(dev_raw, *, out_dir: str, ckpt_dir: str = "checkpoints",
                       folds: Sequence[int] = RQ2_FOLDS, seeds: Sequence[int] = RQ2_SEEDS,
                       ns: Sequence[int] = RQ2_N_GRID, resume: bool = True,
                       force: bool = False, keep_checkpoints: bool = False,
                       run_experiment_fn=None,
                       accelerator: Optional[str] = None,
                       devices: int = 1) -> dict:
    """Run the full 3 x 5 x |N| RQ2 grid with resume-safe persistence."""
    mapper = CanonicalLabelMapper.from_breeds(RQ2_BREEDS)
    os.makedirs(out_dir, exist_ok=True)

    completed = load_completed_runs(out_dir) if (resume and not force) else {}
    records: Dict[str, dict] = dict(completed)

    for fold in folds:
        for seed in seeds:
            sampled, per_breed = sample_nested_rq2(dev_raw, fold=fold, seed=seed, ns=ns)
            _, val_pos = rq2_fold_partitions(dev_raw, fold)

            manifest = {
                "fold": int(fold),
                "replicate_seed": int(seed),
                "breeds": list(RQ2_BREEDS),
                "max_n": int(max(ns)),
                "rng": "numpy.default_rng(numpy.random.SeedSequence([seed, fold]))",
                "per_breed_source_index": {
                    b: [int(dev_raw.dev_source_index[p]) for p in per_breed[b]]
                    for b in RQ2_BREEDS
                },
                "sampled_source_index": {
                    str(int(n)): [int(dev_raw.dev_source_index[p]) for p in sampled[int(n)]]
                    for n in ns
                },
                "val_source_index": [int(dev_raw.dev_source_index[p]) for p in val_pos],
                "val_size": int(len(val_pos)),
            }
            save_json_atomic(
                manifest,
                os.path.join(out_dir, "sampling_manifests", f"fold{fold}_seed{seed}.json"),
            )

            for n in ns:
                key = f"fold{fold}_seed{seed}_N{n}"
                if key in completed and not force:
                    continue
                records[key] = run_rq2_single(
                    dev_raw, mapper,
                    fold=fold, seed=seed, n=n,
                    train_positions=sampled[int(n)], val_positions=val_pos,
                    out_dir=out_dir, ckpt_dir=ckpt_dir,
                    keep_checkpoints=keep_checkpoints,
                    run_experiment_fn=run_experiment_fn,
                    accelerator=accelerator, devices=devices,
                )

    summary = aggregate_rq2(list(records.values()), ns=ns, folds=folds, seeds=seeds)
    return summary
