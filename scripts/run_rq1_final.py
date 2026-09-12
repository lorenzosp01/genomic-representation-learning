#!/usr/bin/env python
"""Run PRIMARY RQ1 final-development training and locked-test evaluation.

Pipeline (scientifically sensitive):

  1. build/fit the shared full-development preprocessor (development rows only);
  2. verify locked-test usage is zero before any final training;
  3. train the requested final model(s) on ALL development rows
     (no validation split, no early stopping, no checkpoint selection);
  4. only after ALL requested model training is complete, explicitly enter the
     locked-test evaluation stage;
  5. transform the frozen locked test with the shared frozen preprocessor;
  6. evaluate each already-trained model exactly once;
  7. report that the frozen locked test (426 rows) was evaluated.

Locked-test metrics are descriptive final results only and never trigger
retraining, epoch selection, configuration selection or model selection.

Usage:
    python scripts/run_rq1_final.py --model both
    python scripts/run_rq1_final.py --model vae
    python scripts/run_rq1_final.py --model contrastive
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

# Make the repository root importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genomic.experiment_data import GenomicExperimentData, locked_test_overlap
from genomic.splitting import load_split

from contrastive_learning.final import (
    DEFAULT_CHECKPOINT as CONTRASTIVE_CHECKPOINT,
    DEFAULT_OUTPUT_DIR,
    FINAL_EMBEDDING_DIM,
    FINAL_EPOCHS as CONTRASTIVE_FINAL_EPOCHS,
    evaluate_contrastive_final,
    train_contrastive_final,
)
from vae.final import (
    DEFAULT_CHECKPOINT as VAE_CHECKPOINT,
    FINAL_EPOCHS as VAE_FINAL_EPOCHS,
    FINAL_LATENT_DIM,
    evaluate_vae_final,
    train_vae_final,
)

DATASET_ID = "ADAPTmap_genotypeTOP_20160222_full"
FAM_PATH = "data/ADAPTmap_genotypeTOP_20160222_full.fam"
BED_PATH = "data/ADAPTmap_genotypeTOP_20160222_full.bed"
BIM_PATH = "data/ADAPTmap_genotypeTOP_20160222_full.bim"
SPLITS_DIR = "splits"
COHORT_LABEL = "indmiss0p10"
OUT_DIR = DEFAULT_OUTPUT_DIR
CKPT_DIR = "checkpoints/rq1_final"
EXPECTED_LOCKED_TEST = 426


def _device() -> str:
    return "gpu" if torch.cuda.is_available() else "cpu"


def vae_config() -> dict:
    return {
        "batch_size": 64,
        "alpha": 0.5,
        "lr": 1e-4,
        "weight_breed": 1.0,
        "weight_continent": 0.0,
        "weight_caseina": 0.0,
        "accelerator": _device(),
        "devices": 1,
    }


def contrastive_config() -> dict:
    return {
        "embedding_dim": 3,
        "flip_max": 0.99,
        "mask_max": 0.99,
        "learning_rate": 0.001,
        "lr_decay_factor": 0.99,
        "lr_decay_interval": 10,
        "accelerator": _device(),
        "devices": 1,
    }


def persist_preprocessor(dev_data, out_dir: str = OUT_DIR) -> dict:
    """Persist the fitted full-development preprocessor + retained SNP identity."""
    import pandas as pd

    os.makedirs(out_dir, exist_ok=True)
    pre = dev_data.preprocessor

    with open(os.path.join(out_dir, "dev_preprocessor.pkl"), "wb") as f:
        pickle.dump(pre, f)

    summary = {
        "marker_missingness_threshold": pre.marker_missingness_threshold,
        "maf_threshold": pre.maf_threshold,
        "ld_window": pre.ld_window,
        "ld_r2_threshold": pre.ld_r2_threshold,
        "n_features_in": int(pre.n_features_in_),
        "n_retained_snps": int(len(pre.retained_snp_indices_)),
        "retained_snp_indices": pre.retained_snp_indices_.tolist(),
        "n_development": int(dev_data.n_samples),
    }
    with open(os.path.join(out_dir, "dev_preprocessor_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    meta = pre.retained_snp_metadata_
    if meta is not None:
        pd.DataFrame({
            "raw_snp_index": meta["raw_snp_index"],
            "snp_id": meta["snp_id"],
            "chromosome": meta["chromosome"],
            "position": meta["position"],
            "allele_1": meta["allele_1"],
            "allele_2": meta["allele_2"],
        }).to_csv(os.path.join(out_dir, "retained_snp_metadata.csv"), index=False)

    return summary


def run_final_with_data(model: str, ed: GenomicExperimentData, *,
                        out_dir: str = OUT_DIR, ckpt_dir: str = CKPT_DIR,
                        seed: int = 42) -> dict:
    """Build dev data, verify zero test usage, train, then evaluate the locked test."""
    # 1. full-development preparation (NO locked-test access)
    dev = ed.build_final_development_data()
    persist_preprocessor(dev, out_dir=out_dir)

    # 2. pre-final assertion: locked-test rows used == 0
    overlap = locked_test_overlap(dev.dev_source_index, ed.split)
    if overlap != 0:
        raise RuntimeError(
            f"locked-test rows leaked into final development preparation: {overlap}"
        )
    print(
        f"[pre-final] development={dev.n_samples} n_features={dev.n_features} "
        f"n_classes={dev.n_classes} | locked-test rows used=0"
    )

    # 3. final training on all development rows (no validation/ES/selection)
    trained = {}
    if model in ("vae", "both"):
        vae_ckpt = os.path.join(ckpt_dir, "vae_latent96_ep168.ckpt")
        trained["vae"] = train_vae_final(
            dev, vae_config(), epochs=VAE_FINAL_EPOCHS,
            latent_dim=FINAL_LATENT_DIM, checkpoint_path=vae_ckpt, seed=seed,
        )
        print(
            f"[vae] trained {trained['vae']['epochs']} epochs on {dev.n_samples} dev rows "
            f"({trained['vae']['runtime_s']:.1f}s) -> {vae_ckpt}"
        )
    if model in ("contrastive", "both"):
        ctr_ckpt = os.path.join(ckpt_dir, "contrastive_emb3_ep1216.ckpt")
        trained["contrastive"] = train_contrastive_final(
            dev, contrastive_config(), epochs=CONTRASTIVE_FINAL_EPOCHS,
            embedding_dim=FINAL_EMBEDDING_DIM, checkpoint_path=ctr_ckpt, seed=seed,
        )
        print(
            f"[contrastive] trained {trained['contrastive']['epochs']} epochs on "
            f"{dev.n_samples} dev rows ({trained['contrastive']['runtime_s']:.1f}s) "
            f"-> {ctr_ckpt}"
        )

    # 4-6. locked-test evaluation stage -- only after ALL training is complete
    test_data = ed.transform_locked_test(dev.preprocessor)
    results = {}
    if "vae" in trained:
        results["vae"] = evaluate_vae_final(trained["vae"], test_data, output_dir=out_dir)
    if "contrastive" in trained:
        results["contrastive"] = evaluate_contrastive_final(
            trained["contrastive"], dev, test_data, output_dir=out_dir
        )

    # 7. report the locked-test stage explicitly
    print(
        f"[locked-test] evaluated {test_data.n_samples} locked-test rows "
        f"(expected {EXPECTED_LOCKED_TEST}); metrics are descriptive only"
    )
    return results


def run_final(model: str, *, seed: int = 42) -> dict:
    split = load_split(
        DATASET_ID, 42, 42, FAM_PATH, out_dir=SPLITS_DIR, cohort_label=COHORT_LABEL
    )
    ed = GenomicExperimentData.from_plink(split, BED_PATH, bim_path=BIM_PATH)
    return run_final_with_data(model, ed, seed=seed)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["vae", "contrastive", "both"], default="both")
    args = parser.parse_args(argv)

    t0 = time.time()
    results = run_final(args.model)
    print(f"Total wall-clock: {time.time() - t0:.1f}s")
    print(json.dumps(
        {
            k: {
                "macro_f1": v["macro_f1"],
                "balanced_accuracy": v["balanced_accuracy"],
                "accuracy": v["accuracy"],
            }
            for k, v in results.items()
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
