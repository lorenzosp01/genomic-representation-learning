#!/usr/bin/env python
"""Definitive RQ2 sample-efficiency experiment (VMGP only).

3 folds x 5 replicate seeds x N in [5,10,15,20,25,30] = 90 runs.

Usage:
    python scripts/run_rq2.py                 # resume-safe (skips completed runs)
    python scripts/run_rq2.py --force         # re-run all runs
    python scripts/run_rq2.py --keep-checkpoints
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genomic.experiment_data import GenomicExperimentData
from genomic.rq2 import (
    RQ2_BREEDS,
    RQ2_FOLDS,
    RQ2_N_GRID,
    RQ2_SEEDS,
    assert_rq2_breeds,
    breed_availability_table,
    save_json_atomic,
)
from genomic.splitting import load_split

from vae.rq2_experiment import load_completed_runs, run_rq2_experiment

DATASET_ID = "ADAPTmap_genotypeTOP_20160222_full"
FAM_PATH = "data/ADAPTmap_genotypeTOP_20160222_full.fam"
BED_PATH = "data/ADAPTmap_genotypeTOP_20160222_full.bed"
BIM_PATH = "data/ADAPTmap_genotypeTOP_20160222_full.bim"
SPLITS_DIR = "splits"
COHORT_LABEL = "indmiss0p10"
OUT_DIR = "results/rq2_sample_efficiency"
CKPT_DIR = "checkpoints"

RUN_CSV_FIELDS = [
    "fold", "replicate_seed", "N", "macro_f1", "balanced_accuracy", "accuracy",
    "retained_snp_count", "best_epoch", "completed_epochs",
    "early_stopping_triggered", "runtime_s", "n_train", "n_val",
    "max_epochs", "latent_dim",
]


def _write_run_metrics_csv(records, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RUN_CSV_FIELDS)
        writer.writeheader()
        for r in sorted(records, key=lambda x: (x["N"], x["fold"], x["replicate_seed"])):
            writer.writerow({k: r[k] for k in RUN_CSV_FIELDS})
    return path


def _write_aggregate_csv(aggregates, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = list(aggregates[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(aggregates)
    return path


def run(model: str, *, out_dir: str = OUT_DIR, ckpt_dir: str = CKPT_DIR,
        resume: bool = True, force: bool = False,
        keep_checkpoints: bool = False) -> dict:
    if model != "vae":
        raise ValueError("RQ2 primary experiment supports only the VMGP (vae) model")

    split = load_split(DATASET_ID, 42, 42, FAM_PATH, out_dir=SPLITS_DIR,
                       cohort_label=COHORT_LABEL)
    ed = GenomicExperimentData.from_plink(split, BED_PATH, bim_path=BIM_PATH)

    # Cohort validation + thesis availability artifact.
    assert_rq2_breeds(split.rq2_breeds(30))
    availability = breed_availability_table(split)
    for row in availability:
        if row["min_train_count"] < 30:
            raise RuntimeError(
                f"breed {row['breed']} has min fold-training count "
                f"{row['min_train_count']} < 30"
            )
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "breed_availability.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(availability[0].keys()))
        writer.writeheader()
        writer.writerows(availability)

    save_json_atomic({
        "experiment": "rq2_sample_efficiency",
        "model": "vae",
        "latent_dim": 96,
        "alpha": 0.5,
        "max_epochs": 200,
        "early_stopping": {"monitor": "val_loss", "mode": "min", "patience": 15},
        "precision": "16-mixed",
        "optimizer": {"name": "Adam", "lr": 1e-4},
        "train_dataloader": {"batch_size": 64, "shuffle": True,
                             "drop_last": False, "num_workers": 4, "pin_memory": True},
        "breeds": list(RQ2_BREEDS),
        "n_grid": list(RQ2_N_GRID),
        "seeds": list(RQ2_SEEDS),
        "folds": list(RQ2_FOLDS),
        "epsilon_N": 0.02,
        "dataset_id": DATASET_ID,
        "cohort_label": COHORT_LABEL,
        "split_meta": split.meta,
        "environment": {
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "locked_test_accessed": False,
    }, os.path.join(out_dir, "protocol.json"))

    # RQ2 operates exclusively on development-only raw data.
    dev_raw = ed.development_raw_data()

    t0 = time.time()
    summary = run_rq2_experiment(
        dev_raw, out_dir=out_dir, ckpt_dir=ckpt_dir,
        resume=resume, force=force, keep_checkpoints=keep_checkpoints,
    )
    summary["total_wall_clock_s"] = time.time() - t0

    records = list(load_completed_runs(out_dir).values())
    _write_run_metrics_csv(records, os.path.join(out_dir, "run_metrics.csv"))
    _write_aggregate_csv(summary["aggregates"], os.path.join(out_dir, "aggregate_metrics.csv"))
    save_json_atomic(summary, os.path.join(out_dir, "summary.json"))
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["vae"], default="vae")
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--force", action="store_true",
                        help="re-run all RQ2 runs (ignores completed records)")
    parser.add_argument("--keep-checkpoints", action="store_true")
    args = parser.parse_args(argv)

    summary = run(
        args.model, out_dir=args.out_dir,
        resume=True, force=args.force,
        keep_checkpoints=args.keep_checkpoints,
    )
    print(f"epsilon_N = {summary['epsilon_N']}")
    print(f"S_ref (mean Macro-F1 @ N={summary['reference_N']}) = {summary['S_ref']:.4f}")
    print(f"threshold = {summary['threshold']:.4f}")
    print(f"N_min = {summary['N_min']}")
    for row in summary["aggregates"]:
        print(f"  N={row['N']:>2}: Macro-F1={row['macro_f1_mean']:.4f}±{row['macro_f1_std']:.4f} "
              f"BalAcc={row['balanced_accuracy_mean']:.4f}±{row['balanced_accuracy_std']:.4f} "
              f"Acc={row['accuracy_mean']:.4f}±{row['accuracy_std']:.4f} "
              f"(n={row['n_observations']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
