#!/usr/bin/env python
"""Phase 1 — bounded RQ2 sample-efficiency mini-sweep (ISGC Sheep HapMap).

Overnight/unattended design mirrors ``run_sheep_rq1.py``: headless Agg,
no ``plt.show``, PNG(300 DPI)+PDF figures under
``results/sheep_cross_species/plots/``, guarded plotting, resume-safe run
records, and progress teed to stdout + ``results/sheep_cross_species/execution.log``.

Frozen decisions (cross-species adaptation of the goat RQ2 protocol):
    * N in {5, 10, 15} individuals per breed (strictly below the minimum
      fold-training count of 17 -> no oversampling or pseudo-replication);
    * 3 subsample seeds (42, 43, 44); nested S5 subset S10 subset S15 per
      (fold, seed), one permutation per breed;
    * VMGP latent_dim=96, pure hyperparameter transfer (batch_size=64,
      alpha=0.5, lr=1e-4, max_epochs=200, EarlyStopping patience=15,
      16-mixed, best-val-loss restore);
    * a FRESH GenomicPreprocessor is fitted on each subsample and used to
      transform it + the fixed validation fold;
    * retention is relative to the full-sample VMGP CV baseline in
      ``results/sheep_cross_species/rq1/summary.json``;
    * the locked test is never materialized here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Imports the headless matplotlib setup; must precede any pyplot consumer.
from sheep_phase1_utils import (  # noqa: E402
    PLOTS_DIR,
    TeeLogger,
    guarded_plot,
    save_figure,
)

from genomic.labels import CanonicalLabelMapper  # noqa: E402
from genomic.rq2 import (  # noqa: E402
    build_rq2_sample,
    preprocessing_provenance,
    save_json_atomic,
)
from run_sheep_rq1 import load_sheep_ed, vae_config  # noqa: E402

from vae.experiment import run_experiment as run_vae_experiment  # noqa: E402
from vae.rq2_experiment import build_rq2_vae_fold_data  # noqa: E402

OUT_DIR = "results/sheep_cross_species/rq2_mini"
CKPT_DIR = "checkpoints/sheep/rq2_mini"
SMOKE_OUT_DIR = "results/sheep_cross_species/_smoke/rq2_mini"
SMOKE_CKPT_DIR = "checkpoints/sheep/_smoke/rq2_mini"
SMOKE_PLOTS_DIR = "results/sheep_cross_species/_smoke/plots"
RQ1_SUMMARY = "results/sheep_cross_species/rq1/summary.json"

FOLDS = (0, 1, 2)
SEEDS = (42, 43, 44)
N_GRID = (5, 10, 15)
RQ2_MAX_EPOCHS = 200


def load_json(path):
    with open(path) as f:
        return json.load(f)


def sheep_fold_positions(dev_raw, breeds, fold):
    in_breeds = np.isin(dev_raw.dev_breed, breeds)
    train_pos = np.flatnonzero((dev_raw.dev_fold != fold) & in_breeds)
    val_pos = np.flatnonzero((dev_raw.dev_fold == fold) & in_breeds)
    return train_pos, val_pos


def sample_nested_sheep(dev_raw, breeds, *, fold, seed, ns):
    """Nested per-breed sampling, mirroring genomic.rq2.sample_nested_rq2."""
    max_n = max(int(n) for n in ns)
    train_pos, _ = sheep_fold_positions(dev_raw, breeds, fold)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(fold)]))

    per_breed = {}
    for b in breeds:
        b_pos = train_pos[dev_raw.dev_breed[train_pos] == b]
        if len(b_pos) < max_n:
            raise ValueError(
                f"breed {b}: only {len(b_pos)} fold-{fold} training rows (< {max_n})"
            )
        per_breed[b] = rng.permutation(b_pos)[:max_n]

    sampled = {}
    for n in ns:
        parts = [per_breed[b][: int(n)] for b in breeds]
        sampled[int(n)] = np.sort(np.concatenate(parts))
    return sampled, per_breed


def run_single(dev_raw, mapper, *, fold, seed, n, train_positions, val_positions,
               out_dir, ckpt_dir, max_epochs) -> dict:
    sample = build_rq2_sample(
        dev_raw, fold=fold, seed=seed, n=n,
        train_positions=train_positions, val_positions=val_positions,
        mapper=mapper,
    )
    vfd = build_rq2_vae_fold_data(sample)

    name = f"fold{fold}_seed{seed}_N{n}"
    t0 = time.time()
    res = run_vae_experiment(
        name=name,
        experiment_data=None,
        config=vae_config(),
        max_epochs=int(max_epochs),
        classifier_config="breed_only",
        balanced=False,
        cap_samples=False,
        folds=[vfd],
        checkpoint_dir=ckpt_dir,
        cm_dir=os.path.join(out_dir, "confusion_matrices"),
        drop_last=False,
    )
    runtime = time.time() - t0

    return {
        "model": "vae",
        "latent_dim": 96,
        "fold": int(fold),
        "replicate_seed": int(seed),
        "N": int(n),
        "macro_f1": float(res["Macro_F1 (mean)"]),
        "balanced_accuracy": float(res["Balanced_Accuracy (mean)"]),
        "accuracy": float(res["Accuracy (mean)"]),
        "best_epoch": int(res["Best_Epochs"][0]),
        "max_epochs": int(max_epochs),
        "runtime_s": float(runtime),
        "n_train": int(sample.n_train),
        "n_val": int(sample.n_val),
        "retained_snp_count": int(res["Num SNPs"]),
        "n_classes": int(sample.n_classes),
        "preprocessing": preprocessing_provenance(sample.preprocessor),
        "train_source_index": [int(x) for x in sample.train_source_index],
        "val_source_index": [int(x) for x in sample.val_source_index],
    }


def _mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    ddof = 1 if arr.size > 1 else 0
    return float(arr.mean()), float(arr.std(ddof=ddof))


def plot_retention_curve(summary, logger, plots_dir=PLOTS_DIR):
    import matplotlib.pyplot as plt

    by_n = summary["by_n"]
    Ns = [r["N"] for r in by_n]
    ret = [r["retention_pct"] for r in by_n]
    ret_std = [r["retention_pct_std"] for r in by_n]
    mf = [r["macro_f1_mean"] for r in by_n]
    mf_std = [r["macro_f1_std"] for r in by_n]
    baseline = summary["baseline_macro_f1_full_sample"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    ax.errorbar(Ns, ret, yerr=ret_std, marker="o", capsize=4, linewidth=2,
                color="C0", label="sheep retention")
    ax.axhline(100.0, color="grey", linestyle=":", label="full-sample baseline")
    ax.axhline(98.0, color="green", linestyle="--", alpha=0.7,
               label="98% (epsilon=0.02)")
    ax.axhline(95.0, color="orange", linestyle="--", alpha=0.7,
               label="95% (epsilon=0.05)")
    for x, y in zip(Ns, ret):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8)
    ax.set_xlabel("N individuals per breed")
    ax.set_ylabel("Macro-F1 retention (%)")
    ax.set_title("RQ2 mini-sweep — retention vs N")
    ax.set_xticks(Ns)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.errorbar(Ns, mf, yerr=mf_std, marker="s", capsize=4, linewidth=2,
                color="C1", label="sheep Macro-F1")
    ax.axhline(baseline, color="grey", linestyle=":",
               label=f"full sample={baseline:.4f}")
    ax.set_xlabel("N individuals per breed")
    ax.set_ylabel("Macro-F1")
    ax.set_title("RQ2 mini-sweep — absolute Macro-F1 vs N")
    ax.set_xticks(Ns)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("RQ2 cross-species mini-sweep — ISGC Sheep HapMap "
                 "(28 breeds, 3 folds x 3 seeds)")
    fig.tight_layout()
    save_figure(fig, "rq2_mini_retention_curve", logger, plots_dir=plots_dir)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="1 fold, 1 seed, N=5, 2 epochs (plumbing test)")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--ckpt-dir", default=None)
    parser.add_argument("--rq1-summary", default=RQ1_SUMMARY)
    parser.add_argument("--max-epochs", type=int, default=None)
    args = parser.parse_args(argv)

    logger = TeeLogger()
    t_start = time.time()

    smoke = args.smoke
    out_dir = args.out_dir or (SMOKE_OUT_DIR if smoke else OUT_DIR)
    ckpt_dir = args.ckpt_dir or (SMOKE_CKPT_DIR if smoke else CKPT_DIR)
    folds = (0,) if smoke else FOLDS
    seeds = (42,) if smoke else SEEDS
    ns = (5,) if smoke else N_GRID
    max_epochs = args.max_epochs or (2 if smoke else RQ2_MAX_EPOCHS)
    plots_dir = SMOKE_PLOTS_DIR if smoke else PLOTS_DIR

    os.makedirs(os.path.join(out_dir, "runs"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "sampling_manifests"), exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    if os.path.exists(args.rq1_summary):
        baseline_macro_f1 = float(
            load_json(args.rq1_summary)["cv"]["vae"]["macro_f1_mean"]
        )
    else:
        raise FileNotFoundError(
            f"missing full-sample VMGP baseline: {args.rq1_summary}; "
            "run scripts/run_sheep_rq1.py first"
        )

    ed = load_sheep_ed()
    dev_raw = ed.development_raw_data()
    breeds = sorted(set(dev_raw.dev_breed.tolist()))
    mapper = ed.label_mapper
    logger.log("=" * 78)
    logger.log(f"RQ2 SHEEP MINI-SWEEP — start {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log(f"breeds={len(breeds)} | folds={list(folds)} | seeds={list(seeds)} | "
               f"N={list(ns)} | max_epochs={max_epochs}")
    logger.log(f"full-sample VMGP baseline MacroF1 = {baseline_macro_f1:.4f} "
               f"({args.rq1_summary})")
    logger.log("=" * 78)

    records = []
    for fold in folds:
        _, val_pos = sheep_fold_positions(dev_raw, breeds, fold)
        for seed in seeds:
            sampled, per_breed = sample_nested_sheep(
                dev_raw, breeds, fold=fold, seed=seed, ns=ns
            )
            manifest = {
                "fold": int(fold),
                "replicate_seed": int(seed),
                "breeds": list(breeds),
                "max_n": int(max(ns)),
                "rng": "numpy.default_rng(numpy.random.SeedSequence([seed, fold]))",
                "per_breed_source_index": {
                    b: [int(dev_raw.dev_source_index[p]) for p in per_breed[b]]
                    for b in breeds
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
                os.path.join(out_dir, "sampling_manifests",
                             f"fold{fold}_seed{seed}.json"),
            )

            for n in ns:
                path = os.path.join(out_dir, "runs",
                                    f"fold{fold}_seed{seed}_N{n}.json")
                if os.path.exists(path) and not args.force:
                    record = load_json(path)
                    logger.log(f"[cache] fold={fold} seed={seed} N={n} "
                               f"MacroF1={record['macro_f1']:.4f}")
                else:
                    record = run_single(
                        dev_raw, mapper,
                        fold=fold, seed=seed, n=n,
                        train_positions=sampled[int(n)], val_positions=val_pos,
                        out_dir=out_dir, ckpt_dir=ckpt_dir,
                        max_epochs=max_epochs,
                    )
                    save_json_atomic(record, path)
                    logger.log(f"[run]   fold={fold} seed={seed} N={n} "
                               f"train={record['n_train']} val={record['n_val']} "
                               f"snps={record['retained_snp_count']} "
                               f"MacroF1={record['macro_f1']:.4f} "
                               f"({record['runtime_s']:.0f}s)")
                records.append(record)

    by_n = []
    for n in ns:
        recs = [r for r in records if int(r["N"]) == int(n)]
        row = {
            "N": int(n),
            "n_observations": len(recs),
            "n_seeds": len(set(r["replicate_seed"] for r in recs)),
            "n_folds": len(set(r["fold"] for r in recs)),
        }
        for metric in ("macro_f1", "balanced_accuracy", "accuracy"):
            mean, std = _mean_std([r[metric] for r in recs])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
        row["retention_pct"] = 100.0 * row["macro_f1_mean"] / baseline_macro_f1
        ret_per_run = [100.0 * float(r["macro_f1"]) / baseline_macro_f1 for r in recs]
        row["retention_pct_mean"], row["retention_pct_std"] = _mean_std(ret_per_run)
        row["retention_pct_per_run"] = ret_per_run
        by_n.append(row)

    summary = {
        "phase": 1,
        "experiment": "rq2_sheep_mini_sweep",
        "dataset_id": "ISGC_sheep_hapmap",
        "design": "bounded cross-species sample-efficiency mini-sweep",
        "protocol": {
            "N_grid": list(ns),
            "seeds": list(seeds),
            "folds": list(folds),
            "max_epochs": int(max_epochs),
            "latent_dim": 96,
            "sampling": "nested per breed (S_N1 subset S_N2), without replacement",
            "preprocessing": "fresh GenomicPreprocessor per run, fitted on the subsample only",
            "early_stopping_patience": 15,
            "precision": "16-mixed",
            "reference": "full-sample VMGP development-CV Macro-F1 (RQ1 sheep)",
            "reference_source": args.rq1_summary,
        },
        "baseline_macro_f1_full_sample": float(baseline_macro_f1),
        "by_n": by_n,
        "records": sorted(
            records, key=lambda r: (int(r["N"]), int(r["fold"]), int(r["replicate_seed"]))
        ),
        "rq2_full_grid_executed": False,
        "rq2_limitation": (
            "N grid truncated to {5,10,15}: minimum fold-training count per breed "
            "is 17, so N>=20 is not feasible for edge breeds; reference is N=15, "
            "hence N_min is bounded by construction"
        ),
        "locked_test_accessed": False,
    }
    save_json_atomic(summary, os.path.join(out_dir, "summary.json"))

    guarded_plot(lambda: plot_retention_curve(summary, logger, plots_dir),
                 "rq2_mini_retention_curve", logger)

    logger.log("")
    logger.log("=" * 78)
    logger.log("RQ2 CROSS-SPECIES MINI-SWEEP — ISGC Sheep HapMap (N in {5,10,15})")
    logger.log("=" * 78)
    logger.log(f"full-sample VMGP baseline MacroF1 = {baseline_macro_f1:.4f}")
    logger.log(f"{'N':>3} | {'obs':>3} | {'MacroF1':>16} | {'BalAcc':>8} | "
               f"{'Acc':>8} | {'retention':>9}")
    for row in by_n:
        logger.log(f"{row['N']:>3} | {row['n_observations']:>3} | "
                   f"{row['macro_f1_mean']:.4f}±{row['macro_f1_std']:.4f} | "
                   f"{row['balanced_accuracy_mean']:.4f} | "
                   f"{row['accuracy_mean']:.4f} | "
                   f"{row['retention_pct']:>7.1f}%")
    logger.log(f"\nSaved: {os.path.join(out_dir, 'summary.json')}")
    logger.log(f"Total wall clock: {time.time() - t_start:.0f}s")
    logger.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        with open("results/sheep_cross_species/execution.log", "a") as f:
            f.write("FATAL rq2_mini:\n" + traceback.format_exc() + "\n")
        traceback.print_exc()
        raise SystemExit(1)
