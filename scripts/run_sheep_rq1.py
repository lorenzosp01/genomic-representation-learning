#!/usr/bin/env python
"""Phase 1 — RQ1 cross-species evaluation on the ISGC Sheep HapMap cohort.

Overnight/unattended design:
    * headless matplotlib (Agg) via ``sheep_phase1_utils``, ``plt.show`` no-op;
    * figures saved as PNG (300 DPI) + PDF under
      ``results/sheep_cross_species/plots/``; plotting errors never crash;
    * progress + metrics teed to stdout and
      ``results/sheep_cross_species/execution.log``;
    * resume-safe per-fold CV records and guarded one-shot locked test.

Frozen protocol:
    * pure hyperparameter transfer: VMGP ``latent_dim=96`` (28 classes) and
      contrastive ``embedding_dim=3`` (unit sphere S^2), goat configuration;
    * VMGP CV: batch_size=64, alpha=0.5, lr=1e-4, max_epochs=200,
      EarlyStopping(val_loss, patience=15), 16-mixed, best-val-loss restore;
    * contrastive CV: max_epochs=5000, EarlyStopping(val_loss, patience=200);
    * GenomicPreprocessor fitted on the fold TRAINING partition only;
    * locked test (278) transformed + evaluated exactly once, after all
      development-fold and final training is complete.

All outputs live under ``results/sheep_cross_species/`` and
``checkpoints/sheep/``; goat artifacts are never touched.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import time
import traceback

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Must come before any module that imports matplotlib.pyplot (sets Agg).
from sheep_phase1_utils import (  # noqa: E402
    PLOTS_DIR,
    TeeLogger,
    guarded_plot,
    save_figure,
)

from genomic.classification_metrics import compute_classification_metrics  # noqa: E402
from genomic.experiment_data import GenomicExperimentData  # noqa: E402
from genomic.rq2 import save_json_atomic  # noqa: E402
from genomic.splitting import load_split  # noqa: E402

import contrastive_learning.experiment as contrastive_experiment  # noqa: E402
import vae.experiment as vae_experiment  # noqa: E402
from contrastive_learning.evaluation import equal_earth_projection  # noqa: E402
from contrastive_learning.final import (  # noqa: E402
    evaluate_contrastive_final,
    train_contrastive_final,
)
from contrastive_learning.model import ContrastiveGeneticModel  # noqa: E402
from vae.final import evaluate_vae_final, train_vae_final  # noqa: E402
from vae.lightning_module import VMGP_LightningSystem  # noqa: E402

SHEEP_DATASET_ID = "ISGC_sheep_hapmap"
SHEEP_FAM = "data/sheep/sheep_hapmap_raw.fam"
SHEEP_BED = "data/sheep/sheep_hapmap_raw.bed"
SHEEP_BIM = "data/sheep/sheep_hapmap_raw.bim"
SHEEP_SPLITS_DIR = "data/sheep/splits"
SHEEP_COHORT_LABEL = "indmiss0p10"

OUT_DIR = "results/sheep_cross_species/rq1"
CKPT_DIR = "checkpoints/sheep/rq1"
SMOKE_OUT_DIR = "results/sheep_cross_species/_smoke/rq1"
SMOKE_CKPT_DIR = "checkpoints/sheep/_smoke/rq1"
SMOKE_PLOTS_DIR = "results/sheep_cross_species/_smoke/plots"

FOLDS = (0, 1, 2)
SEED = 42
LATENT_DIM = 96
EMBEDDING_DIM = 3
VAE_MAX_EPOCHS = 200
CONTRASTIVE_MAX_EPOCHS = 5000
EXPECTED_BREEDS = 28


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _device() -> str:
    return "gpu" if torch.cuda.is_available() else "cpu"


def vae_config() -> dict:
    """Frozen goat RQ1 VMGP configuration (pure transfer)."""
    return {
        "batch_size": 64,
        "alpha": 0.5,
        "lr": 1e-4,
        "latent_dim": LATENT_DIM,
        "weight_breed": 1.0,
        "weight_continent": 0.0,
        "weight_caseina": 0.0,
        "accelerator": _device(),
        "devices": 1,
    }


def contrastive_config() -> dict:
    """Frozen goat contrastive configuration (pure transfer)."""
    return {
        "embedding_dim": EMBEDDING_DIM,
        "flip_max": 0.99,
        "mask_max": 0.99,
        "learning_rate": 0.001,
        "lr_decay_factor": 0.99,
        "lr_decay_interval": 10,
        "accelerator": _device(),
        "devices": 1,
        "precision": "32-true",
    }


def load_sheep_ed() -> GenomicExperimentData:
    split = load_split(
        SHEEP_DATASET_ID, 42, 42, SHEEP_FAM,
        out_dir=SHEEP_SPLITS_DIR, cohort_label=SHEEP_COHORT_LABEL,
    )
    ed = GenomicExperimentData.from_plink(split, SHEEP_BED, bim_path=SHEEP_BIM)
    if ed.n_classes != EXPECTED_BREEDS:
        raise RuntimeError(
            f"sheep cohort must have {EXPECTED_BREEDS} breeds, found {ed.n_classes}"
        )
    return ed


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _metrics_dict(cm) -> dict:
    return {
        "macro_f1": float(cm["macro_f1"]),
        "balanced_accuracy": float(cm["balanced_accuracy"]),
        "accuracy": float(cm["accuracy"]),
    }


def _mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    ddof = 1 if arr.size > 1 else 0
    return float(arr.mean()), float(arr.std(ddof=ddof))


class VAECurveCallback(pl.Callback):
    """Record per-epoch train/val metrics (loss, rec, KL, breed CE/acc)."""

    def __init__(self):
        self._by_epoch = {}

    def _snapshot(self, trainer) -> None:
        row = {"epoch": int(trainer.current_epoch) + 1}
        for key, value in trainer.callback_metrics.items():
            try:
                row[str(key)] = float(value)
            except Exception:
                continue
        self._by_epoch[int(trainer.current_epoch)] = row

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        self._snapshot(trainer)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        self._snapshot(trainer)

    def rows(self):
        return [self._by_epoch[k] for k in sorted(self._by_epoch)]


def rf_fold_metrics(ed, fold, n_classes, random_state=42) -> dict:
    fd = ed.fold_data(fold)
    t0 = time.time()
    clf = RandomForestClassifier(
        n_estimators=200, random_state=random_state, n_jobs=-1
    )
    clf.fit(fd.X_train, fd.y_train)
    preds = clf.predict(fd.X_val)
    cm = compute_classification_metrics(
        fd.y_val, preds, labels=np.arange(n_classes)
    )
    out = _metrics_dict(cm)
    out.update({
        "fold": int(fold),
        "n_train": int(fd.X_train.shape[0]),
        "n_val": int(fd.X_val.shape[0]),
        "n_markers": int(fd.n_features),
        "runtime_s": float(time.time() - t0),
    })
    return out


def _move_vae_pca_plot(name: str, logger: TeeLogger, plots_dir: str = PLOTS_DIR) -> None:
    src = f"plot_vae_{name.replace(' ', '_')}_kfold.png"
    if not os.path.exists(src):
        return
    try:
        os.makedirs(plots_dir, exist_ok=True)
        dst = os.path.join(plots_dir, f"rq1_vmgp_pca_bestfold_{name}.png")
        shutil.move(src, dst)
        logger.log(f"[plots] saved {dst}")
    except Exception:
        logger.log(f"[plots] ERROR moving PCA plot {src}:\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# Training wrappers
# ---------------------------------------------------------------------------
def run_vae_fold(ed, vfd, fold, out_dir, ckpt_dir, max_epochs, logger,
                 plots_dir=PLOTS_DIR):
    name = f"sheepLatDim96_fold{fold}"
    curve_cb = VAECurveCallback()
    t0 = time.time()
    res = vae_experiment.run_experiment(
        name=name,
        experiment_data=ed,
        config=vae_config(),
        max_epochs=max_epochs,
        classifier_config="breed_only",
        balanced=False,
        cap_samples=False,
        folds=[vfd],
        checkpoint_dir=ckpt_dir,
        cm_dir=os.path.join(out_dir, "confusion_matrices"),
        history_callback=curve_cb,
    )
    _move_vae_pca_plot(name, logger, plots_dir=plots_dir)

    curves_dir = os.path.join(out_dir, "curves")
    os.makedirs(curves_dir, exist_ok=True)
    save_json_atomic(curve_cb.rows(), os.path.join(curves_dir, f"vae_fold{fold}.json"))

    record = {
        "fold": int(fold),
        "macro_f1": float(res["Macro_F1 (mean)"]),
        "balanced_accuracy": float(res["Balanced_Accuracy (mean)"]),
        "accuracy": float(res["Accuracy (mean)"]),
        "best_epoch": int(res["Best_Epochs"][0]),
        "median_best_epoch": float(res["Median_Best_Epoch"]),
        "max_epochs": int(max_epochs),
        "n_snps": int(res["Num SNPs"]),
        "n_classes": int(res["Num Classes Breed"]),
        "runtime_s": float(time.time() - t0),
    }
    return record


def run_contrastive_fold(ed, fold, out_dir, ckpt_dir, max_epochs, logger):
    t0 = time.time()
    name = f"sheepEmb3_fold{fold}"
    res = contrastive_experiment.run_experiment(
        name=name,
        experiment_data=ed,
        config=contrastive_config(),
        max_epochs=max_epochs,
        folds=[fold],
        checkpoint_dir=ckpt_dir,
    )
    fd = ed.fold_data(fold)

    # Persist the learned embedding space (validation fold) for the thesis plot.
    try:
        emb_dir = os.path.join(out_dir, "embeddings")
        os.makedirs(emb_dir, exist_ok=True)
        np.savez(
            os.path.join(emb_dir, f"contrastive_fold{fold}.npz"),
            Z=np.asarray(res["_best_Z"], dtype=np.float32),
            y=np.asarray(res["_best_y"], dtype=np.int64),
        )
    except Exception:
        logger.log(f"[plots] ERROR saving embeddings fold {fold}:\n"
                   f"{traceback.format_exc()}")

    record = {
        "fold": int(fold),
        "macro_f1": float(res["Macro_F1 (mean)"]),
        "balanced_accuracy": float(res["Balanced_Accuracy (mean)"]),
        "accuracy": float(res["Accuracy (mean)"]),
        "best_epoch": int(res["Best_Epochs"][0]),
        "median_best_epoch": float(res["Median_Best_Epoch"]),
        "max_epochs": int(max_epochs),
        "n_markers": int(fd.n_features),
        "n_classes": int(ed.n_classes),
        "runtime_s": float(time.time() - t0),
    }
    return record


def _cv_summary(records, method):
    ordered = sorted(records, key=lambda r: r["fold"])
    out = {"method": method, "per_fold": ordered}
    for metric in ("macro_f1", "balanced_accuracy", "accuracy"):
        mean, std = _mean_std([r[metric] for r in ordered])
        out[f"{metric}_mean"] = mean
        out[f"{metric}_std"] = std
    best_epochs = [r["best_epoch"] for r in ordered if "best_epoch" in r]
    out["median_best_epoch"] = (
        float(np.median(best_epochs)) if best_epochs else None
    )
    return out


def _train_or_load_vae_final(dev, epochs, ckpt_path, marker_path, force, logger):
    if (not force and os.path.exists(ckpt_path) and os.path.exists(marker_path)):
        model = VMGP_LightningSystem.load_from_checkpoint(ckpt_path, map_location="cpu")
        marker = _load_json(marker_path)
        logger.log(f"[vae-final] loaded existing checkpoint {ckpt_path}")
        return {"model": model, **marker}
    trained = train_vae_final(
        dev, vae_config(), epochs=int(epochs), latent_dim=LATENT_DIM,
        checkpoint_path=ckpt_path, seed=SEED,
    )
    save_json_atomic({k: v for k, v in trained.items() if k != "model"}, marker_path)
    return trained


def _train_or_load_contrastive_final(dev, epochs, ckpt_path, marker_path, force, logger):
    if (not force and os.path.exists(ckpt_path) and os.path.exists(marker_path)):
        model = ContrastiveGeneticModel.load_from_checkpoint(ckpt_path, map_location="cpu")
        marker = _load_json(marker_path)
        logger.log(f"[contrastive-final] loaded existing checkpoint {ckpt_path}")
        return {"model": model, **marker}
    trained = train_contrastive_final(
        dev, contrastive_config(), epochs=int(epochs),
        embedding_dim=EMBEDDING_DIM, checkpoint_path=ckpt_path, seed=SEED,
    )
    save_json_atomic({k: v for k, v in trained.items() if k != "model"}, marker_path)
    return trained


# ---------------------------------------------------------------------------
# Figures (all wrapped in guarded_plot; failures never crash the run)
# ---------------------------------------------------------------------------
def plot_vae_training_curves(out_dir, logger, plots_dir=PLOTS_DIR):
    import matplotlib.pyplot as plt

    curves_dir = os.path.join(out_dir, "curves")
    histories = []
    for path in sorted(glob.glob(os.path.join(curves_dir, "vae_fold*.json"))):
        with open(path) as f:
            histories.append(json.load(f))
    if not histories:
        logger.log("[plots] no VMGP curve histories found; skipping training curves")
        return

    panels = [
        ("loss", "Total loss"),
        ("rec_loss", "Reconstruction term (MSE + 1e-4 KL)"),
        ("kl_loss", "KL divergence (nats/sample)"),
        ("loss_breed", "Breed classification loss (CE)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (key, title) in zip(axes.ravel(), panels):
        for i, hist in enumerate(histories):
            epochs = [r["epoch"] for r in hist]
            for split, style in (("train", "-"), ("val", "--")):
                vals = [r.get(f"{split}_{key}") for r in hist]
                pts = [(e, v) for e, v in zip(epochs, vals) if v is not None]
                if not pts:
                    continue
                xs, ys = zip(*pts)
                ax.plot(xs, ys, style, linewidth=1.0, alpha=0.35,
                        color=f"C{i}", label=f"fold {i} {split}" if key == "loss" else None)
            # mean across folds
            shared = {}
            for r in hist:
                val = r.get(f"train_{key}")
                if val is not None:
                    shared.setdefault(r["epoch"], []).append(val)
            if shared:
                xs = sorted(shared)
                ax.plot(xs, [float(np.mean(shared[e])) for e in xs], "k-",
                        linewidth=2.0, label="train mean" if key == "loss" else None)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("RQ1 sheep cross-species — VMGP d=96 training curves")
    fig.tight_layout()
    save_figure(fig, "rq1_vmgp_training_curves", logger, plots_dir=plots_dir)


def plot_contrastive_embedding_space(out_dir, class_names, logger,
                                     plots_dir=PLOTS_DIR):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    emb_dir = os.path.join(out_dir, "embeddings")
    paths = sorted(glob.glob(os.path.join(emb_dir, "contrastive_fold*.npz")))
    if not paths:
        logger.log("[plots] no contrastive embeddings found; skipping embedding plot")
        return
    data = [np.load(p) for p in paths]
    Z = np.concatenate([d["Z"] for d in data], axis=0)
    y = np.concatenate([d["y"] for d in data], axis=0)

    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) if i < 20 else plt.get_cmap("tab20b")(i - 20)
              for i in range(len(class_names))]

    fig = plt.figure(figsize=(13, 7.5))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    proj = equal_earth_projection(Z)
    ax2d = fig.add_subplot(1, 2, 2)

    for cls, name in enumerate(class_names):
        mask = y == cls
        if not mask.any():
            continue
        ax3d.scatter(Z[mask, 0], Z[mask, 1], Z[mask, 2],
                     s=8, alpha=0.65, color=colors[cls])
        ax2d.scatter(proj[mask, 0], proj[mask, 1],
                     s=8, alpha=0.65, color=colors[cls])

    # unit-sphere wireframe context
    u, v = np.mgrid[0:2 * np.pi:24j, 0:np.pi:12j]
    ax3d.plot_wireframe(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v),
                        np.cos(v), color="grey", linewidth=0.2, alpha=0.25)
    ax3d.set_title("Contrastive embeddings on S^2 (validation folds)")
    ax3d.set_xlabel("z1"); ax3d.set_ylabel("z2"); ax3d.set_zlabel("z3")
    ax3d.set_box_aspect([1, 1, 1])
    ax2d.set_title("Equal Earth projection of S^2")
    ax2d.set_xlabel("x'"); ax2d.set_ylabel("y'")
    ax2d.grid(alpha=0.3)

    handles = [Line2D([0], [0], marker="o", linestyle="", color=colors[i],
                      label=class_names[i], markersize=5)
               for i in range(len(class_names))]
    fig.legend(handles=handles, loc="lower center", fontsize=5.5, ncol=6,
               bbox_to_anchor=(0.5, -0.01), frameon=False)
    fig.suptitle("RQ1 sheep cross-species — contrastive embedding space (emb=3, KNN k=3)")
    fig.tight_layout(rect=[0, 0.10, 1, 0.95])
    save_figure(fig, "rq1_contrastive_embedding_space", logger,
                plots_dir=plots_dir)


def _plot_single_cm(cm, labels, title, stem, logger, plots_dir=PLOTS_DIR):
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=np.float64)
    row_sums = np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
    norm = cm / row_sums
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=5)
    ax.set_yticklabels(labels, fontsize=5)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    save_figure(fig, stem, logger, plots_dir=plots_dir)


def plot_locked_test_confusion_matrices(summary, class_names, logger,
                                        plots_dir=PLOTS_DIR):
    import matplotlib.pyplot as plt

    finals = summary.get("finals")
    if not finals:
        logger.log("[plots] no locked-test finals; skipping confusion matrices")
        return
    entries = [
        ("rf", finals["rf"], "Random Forest"),
        ("vmgp", finals["vae"], "VMGP d=96"),
        ("contrastive", finals["contrastive"], "Contrastive emb=3 (KNN k=3)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(24, 7.5))
    for ax, (key, metrics, title) in zip(axes, entries):
        cm = np.asarray(metrics["confusion_matrix"], dtype=np.float64)
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
        ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_title(f"{title}\nMacroF1={metrics['macro_f1']:.3f}")
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=90, fontsize=5)
        ax.set_yticklabels(class_names, fontsize=5)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
    fig.suptitle("RQ1 sheep cross-species — locked-test confusion matrices (278 animals)")
    fig.tight_layout()
    save_figure(fig, "rq1_locked_test_confusion_matrices", logger,
                plots_dir=plots_dir)

    for key, metrics, title in entries:
        _plot_single_cm(
            metrics["confusion_matrix"], class_names, title,
            f"rq1_locked_test_confusion_{key}", logger, plots_dir=plots_dir,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="single fold, 2 epochs, no finals (plumbing test)")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--ckpt-dir", default=None)
    parser.add_argument("--max-epochs-vae", type=int, default=None)
    parser.add_argument("--max-epochs-contrastive", type=int, default=None)
    parser.add_argument("--skip-finals", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args(argv)

    logger = TeeLogger()
    t_start = time.time()

    smoke = args.smoke
    out_dir = args.out_dir or (SMOKE_OUT_DIR if smoke else OUT_DIR)
    ckpt_dir = args.ckpt_dir or (SMOKE_CKPT_DIR if smoke else CKPT_DIR)
    folds = (0,) if smoke else FOLDS
    max_epochs_vae = args.max_epochs_vae or (2 if smoke else VAE_MAX_EPOCHS)
    max_epochs_contrastive = (
        args.max_epochs_contrastive or (2 if smoke else CONTRASTIVE_MAX_EPOCHS)
    )
    skip_finals = args.skip_finals or smoke
    skip_plots = args.skip_plots
    plots_dir = SMOKE_PLOTS_DIR if smoke else PLOTS_DIR

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    ed = load_sheep_ed()
    n_classes = int(ed.n_classes)
    logger.log("=" * 78)
    logger.log(f"RQ1 SHEEP CROSS-SPECIES — start {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log(f"cohort: {ed.n_samples} eligible | {n_classes} breeds | "
               f"n_folds={ed.split.n_folds}")
    logger.log(f"out_dir={out_dir} | ckpt_dir={ckpt_dir} | folds={list(folds)} | "
               f"vae_epochs<={max_epochs_vae} | contrastive_epochs<={max_epochs_contrastive}")
    logger.log("=" * 78)

    cv_dir = os.path.join(out_dir, "cv")
    os.makedirs(cv_dir, exist_ok=True)

    # --- 1. Random Forest full-array baseline ---------------------------
    rf_records = []
    for k in folds:
        path = os.path.join(cv_dir, f"rf_fold{k}.json")
        if os.path.exists(path) and not args.force:
            record = _load_json(path)
            logger.log(f"[rf] fold {k}: cached MacroF1={record['macro_f1']:.4f}")
        else:
            record = rf_fold_metrics(ed, k, n_classes)
            save_json_atomic(record, path)
            logger.log(f"[rf] fold {k}: MacroF1={record['macro_f1']:.4f} "
                       f"({record['runtime_s']:.0f}s)")
        rf_records.append(record)

    # --- 2. VMGP d=96 CV ------------------------------------------------
    vae_records = []
    vae_folds = None
    for k in folds:
        path = os.path.join(cv_dir, f"vae_fold{k}.json")
        if os.path.exists(path) and not args.force:
            record = _load_json(path)
            logger.log(f"[vae] fold {k}: cached MacroF1={record['macro_f1']:.4f}")
        else:
            if vae_folds is None:
                vae_folds = vae_experiment._build_folds(
                    ed, vae_config(), True, False, False, False, None
                )
            vfd = [f for f in vae_folds if f.fold == k][0]
            record = run_vae_fold(ed, vfd, k, out_dir, ckpt_dir, max_epochs_vae,
                                  logger, plots_dir=plots_dir)
            save_json_atomic(record, path)
            logger.log(f"[vae] fold {k}: MacroF1={record['macro_f1']:.4f} "
                       f"best_epoch={record['best_epoch']} ({record['runtime_s']:.0f}s)")
        vae_records.append(record)

    # --- 3. Contrastive emb=3 CV ---------------------------------------
    contrastive_records = []
    for k in folds:
        path = os.path.join(cv_dir, f"contrastive_fold{k}.json")
        if os.path.exists(path) and not args.force:
            record = _load_json(path)
            logger.log(f"[contrastive] fold {k}: cached MacroF1={record['macro_f1']:.4f}")
        else:
            record = run_contrastive_fold(
                ed, k, out_dir, ckpt_dir, max_epochs_contrastive, logger
            )
            save_json_atomic(record, path)
            logger.log(f"[contrastive] fold {k}: MacroF1={record['macro_f1']:.4f} "
                       f"best_epoch={record['best_epoch']} ({record['runtime_s']:.0f}s)")
        contrastive_records.append(record)

    summary = {
        "phase": 1,
        "dataset_id": SHEEP_DATASET_ID,
        "thesis_design": "goat-primary; sheep = cross-species generalization validation",
        "protocol": {
            "folds": list(folds),
            "seed": SEED,
            "vae": {"latent_dim": LATENT_DIM, "max_epochs": max_epochs_vae,
                    "batch_size": 64, "alpha": 0.5, "lr": 1e-4,
                    "early_stopping_patience": 15, "precision": "16-mixed"},
            "contrastive": {"embedding_dim": EMBEDDING_DIM,
                            "max_epochs": max_epochs_contrastive,
                            "early_stopping_patience": 200},
            "rf": {"n_estimators": 200, "random_state": 42, "n_jobs": -1},
            "hyperparameter_policy": "pure transfer from goat (no grid search)",
        },
        "split_meta": ed.split.meta,
        "cv": {
            "rf": _cv_summary(rf_records, "rf"),
            "vae": _cv_summary(vae_records, "vae"),
            "contrastive": _cv_summary(contrastive_records, "contrastive"),
        },
        "locked_test_accessed": False,
    }

    # --- 4. Finals + one-shot locked test ------------------------------
    guard_path = os.path.join(out_dir, "locked_test_evaluated.json")
    if skip_finals:
        logger.log("[finals] skipped (smoke/--skip-finals)")
    elif os.path.exists(guard_path) and not args.force:
        logger.log("[finals] locked test already evaluated; reusing existing results")
        summary["finals"] = _load_json(guard_path)["finals"]
        summary["locked_test_accessed"] = True
    else:
        dev = ed.build_final_development_data()
        vae_epochs = int(round(np.median([r["best_epoch"] for r in vae_records])))
        con_epochs = int(round(np.median([r["best_epoch"] for r in contrastive_records])))
        logger.log(f"[finals] dev rows={dev.n_samples} n_features={dev.n_features} "
                   f"| vae_epochs={vae_epochs} contrastive_epochs={con_epochs}")

        trained_vae = _train_or_load_vae_final(
            dev, vae_epochs, os.path.join(ckpt_dir, "vae_latent96_final.ckpt"),
            os.path.join(out_dir, "vae_final_trained.json"), args.force, logger,
        )
        trained_con = _train_or_load_contrastive_final(
            dev, con_epochs,
            os.path.join(ckpt_dir, "contrastive_emb3_final.ckpt"),
            os.path.join(out_dir, "contrastive_final_trained.json"), args.force, logger,
        )

        rf_final = RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        )
        rf_final.fit(dev.X_dev, dev.y_dev)

        test_data = ed.transform_locked_test(dev.preprocessor)
        vae_eval = evaluate_vae_final(trained_vae, test_data, output_dir=out_dir)
        con_eval = evaluate_contrastive_final(
            trained_con, dev, test_data, output_dir=out_dir
        )
        rf_pred = rf_final.predict(test_data.X_test)
        rf_cm = compute_classification_metrics(
            test_data.y_test, rf_pred, labels=np.arange(n_classes)
        )

        summary["finals"] = {
            "dev_rows": int(dev.n_samples),
            "dev_features": int(dev.n_features),
            "vae_epochs": int(vae_epochs),
            "contrastive_epochs": int(con_epochs),
            "locked_test_rows": int(test_data.n_samples),
            "rf": {**_metrics_dict(rf_cm),
                   "confusion_matrix": rf_cm["confusion_matrix"].tolist()},
            "vae": {k: vae_eval[k] for k in (
                "macro_f1", "balanced_accuracy", "accuracy", "confusion_matrix")},
            "contrastive": {k: con_eval[k] for k in (
                "macro_f1", "balanced_accuracy", "accuracy", "confusion_matrix")},
        }
        summary["locked_test_accessed"] = True
        save_json_atomic(
            {"evaluated_once": True, "finals": summary["finals"]}, guard_path
        )

    # --- 5. Figures (guarded; never crash) ------------------------------
    if not skip_plots:
        guarded_plot(lambda: plot_vae_training_curves(out_dir, logger, plots_dir),
                     "rq1_vmgp_training_curves", logger)
        guarded_plot(lambda: plot_contrastive_embedding_space(
            out_dir, list(ed.class_names), logger, plots_dir),
            "rq1_contrastive_embedding_space", logger)
        if "finals" in summary:
            guarded_plot(lambda: plot_locked_test_confusion_matrices(
                summary, list(ed.class_names), logger, plots_dir),
                "rq1_locked_test_confusion_matrices", logger)

    save_json_atomic(summary, os.path.join(out_dir, "summary.json"))

    # --- 6. Report ------------------------------------------------------
    logger.log("")
    logger.log("=" * 78)
    logger.log("RQ1 CROSS-SPECIES — ISGC Sheep HapMap (28 breeds)")
    logger.log("=" * 78)
    for name, key in (("Random Forest (full array)", "rf"),
                      ("VMGP d=96", "vae"),
                      ("Contrastive emb=3 (KNN k=3)", "contrastive")):
        cv = summary["cv"][key]
        logger.log(f"{name:32s} CV MacroF1={cv['macro_f1_mean']:.4f}"
                   f"±{cv['macro_f1_std']:.4f} | BalAcc={cv['balanced_accuracy_mean']:.4f}"
                   f" | Acc={cv['accuracy_mean']:.4f}")
    if "finals" in summary:
        logger.log("")
        logger.log("Locked test (278 rows, evaluated once):")
        for name, key in (("Random Forest", "rf"), ("VMGP d=96", "vae"),
                          ("Contrastive emb=3", "contrastive")):
            m = summary["finals"][key]
            logger.log(f"  {name:20s} MacroF1={m['macro_f1']:.4f} | "
                       f"BalAcc={m['balanced_accuracy']:.4f} | Acc={m['accuracy']:.4f}")
        logger.log(f"  (dev epochs: vae={summary['finals']['vae_epochs']}, "
                   f"contrastive={summary['finals']['contrastive_epochs']})")
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
            f.write("FATAL rq1:\n" + traceback.format_exc() + "\n")
        traceback.print_exc()
        raise SystemExit(1)
