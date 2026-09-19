#!/usr/bin/env python
"""Phase 1 completion report — consolidated stdout summary + figure inventory.

Reads the artifacts produced by ``scripts/run_sheep_rq1.py`` and
``scripts/run_sheep_rq2_mini.py`` and prints the thesis-ready completion
report; also appends it to ``results/sheep_cross_species/execution.log``.

Usage:
    uv run python scripts/sheep_phase1_report.py
"""

from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sheep_phase1_utils import PLOTS_DIR, TeeLogger  # noqa: E402

RQ1_SUMMARY = "results/sheep_cross_species/rq1/summary.json"
RQ2_SUMMARY = "results/sheep_cross_species/rq2_mini/summary.json"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def fmt(mean, std):
    return f"{mean:.4f} ± {std:.4f}"


def main() -> int:
    logger = TeeLogger()
    logger.log("")
    logger.log("#" * 78)
    logger.log("# PHASE 1 COMPLETION REPORT — SHEEP CROSS-SPECIES (RQ1 + RQ2 mini)")
    logger.log("#" * 78)

    if os.path.exists(RQ1_SUMMARY):
        rq1 = load_json(RQ1_SUMMARY)
        logger.log("")
        logger.log("[RQ1] Development 3-fold CV (mean ± std)")
        logger.log(f"{'method':38s} {'Macro-F1':>18s} {'Balanced Acc':>14s} "
                   f"{'Accuracy':>14s}")
        for label, key in (("Random Forest (full array)", "rf"),
                           ("VMGP d=96 (28 classes)", "vae"),
                           ("Contrastive emb=3 (KNN k=3)", "contrastive")):
            cv = rq1["cv"][key]
            logger.log(f"{label:38s} "
                       f"{fmt(cv['macro_f1_mean'], cv['macro_f1_std']):>18s} "
                       f"{fmt(cv['balanced_accuracy_mean'], cv['balanced_accuracy_std']):>14s} "
                       f"{fmt(cv['accuracy_mean'], cv['accuracy_std']):>14s}")
        if "finals" in rq1:
            logger.log("")
            logger.log(f"[RQ1] Locked test (one-shot, "
                       f"{rq1['finals']['locked_test_rows']} animals)")
            for label, key in (("Random Forest", "rf"), ("VMGP d=96", "vae"),
                               ("Contrastive emb=3", "contrastive")):
                m = rq1["finals"][key]
                logger.log(f"{label:38s} MacroF1={m['macro_f1']:.4f} | "
                           f"BalAcc={m['balanced_accuracy']:.4f} | "
                           f"Acc={m['accuracy']:.4f}")
            logger.log(f"(final dev epochs: vae={rq1['finals']['vae_epochs']}, "
                       f"contrastive={rq1['finals']['contrastive_epochs']})")
    else:
        logger.log(f"[RQ1] MISSING {RQ1_SUMMARY}")

    if os.path.exists(RQ2_SUMMARY):
        rq2 = load_json(RQ2_SUMMARY)
        logger.log("")
        logger.log(f"[RQ2 mini-sweep] full-sample VMGP baseline MacroF1 = "
                   f"{rq2['baseline_macro_f1_full_sample']:.4f}")
        logger.log(f"{'N':>3s} | {'obs':>3s} | {'Macro-F1':>16s} | "
                   f"{'BalAcc':>8s} | {'Acc':>8s} | {'retention %':>11s}")
        for row in rq2["by_n"]:
            logger.log(f"{row['N']:>3d} | {row['n_observations']:>3d} | "
                       f"{row['macro_f1_mean']:.4f}±{row['macro_f1_std']:.4f} | "
                       f"{row['balanced_accuracy_mean']:.4f} | "
                       f"{row['accuracy_mean']:.4f} | "
                       f"{row['retention_pct']:>10.1f}%")
    else:
        logger.log(f"[RQ2] MISSING {RQ2_SUMMARY}")

    logger.log("")
    logger.log(f"[Figures] inventory of {PLOTS_DIR}/")
    files = sorted(glob.glob(os.path.join(PLOTS_DIR, "*")))
    stems = sorted({os.path.splitext(f)[0] for f in files})
    for stem in stems:
        png = stem + ".png"
        pdf = stem + ".pdf"
        parts = []
        for path in (png, pdf):
            if os.path.exists(path):
                parts.append(f"{os.path.basename(path)} ({os.path.getsize(path)//1024} KiB)")
        logger.log(f"  - {os.path.basename(stem)}: " + " + ".join(parts))
    if not stems:
        logger.log("  (no figures found)")

    logger.log("#" * 78)
    logger.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
