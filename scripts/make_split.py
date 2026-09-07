#!/usr/bin/env python
"""Explicitly generate and persist a train/validation/test split.

Splits are produced on demand by this script and written under ``splits/``.
They are never created or silently replaced during DataModule setup.

The final protocol applies pre-split sample-level individual-missingness QC
(``missingness < individual_missingness_threshold``), then the breed-eligibility
rule (``cohort_min_breed_size``), then the frozen outer split + K-fold.

Usage:
    python scripts/make_split.py [--config configs/splits.yaml]
"""

from __future__ import annotations

import argparse
import os
import sys

import yaml

# Make the repository root importable so ``import genomic`` works regardless
# of the directory the script is invoked from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genomic.splitting import build_split, save_split
from genomic.cohort import compute_individual_missingness, individual_qc_mask


def _report(split) -> None:
    print("=" * 64)
    print("Split generated")
    print("=" * 64)
    print(f"  dataset_id      : {split.meta['dataset_id']}")
    print(f"  cohort_label    : {split.meta['cohort_label'] or '(none)'}")
    print(f"  protocol_version: {split.meta['protocol_version']}")
    print(f"  individual_missingness_threshold: {split.meta['individual_missingness_threshold']}")
    print(f"  individual_missingness_rule     : {split.meta['individual_missingness_rule']}")
    print(f"  cohort_min_breed_size           : {split.meta['cohort_min_breed_size']}")
    print(f"  raw_sample_count                : {split.meta['raw_sample_count']}")
    print(f"  eligible_after_individual_qc    : {split.meta['eligible_after_individual_qc_count']}")
    print(f"  final_primary_cohort_count      : {split.meta['final_primary_cohort_count']}")
    print(f"  outer_split_seed: {split.meta['outer_split_seed']}")
    print(f"  fold_seed       : {split.meta['fold_seed']}")
    print(f"  k_folds         : {split.meta['k_folds']}")
    print(f"  n_development   : {split.meta['n_development']}")
    print(f"  n_test          : {split.meta['n_test']}")
    print(f"  breeds          : {len(set(split.breed.tolist()))}")
    print(f"  source_sha256   : {split.meta['source_manifest_sha256']}")
    print(f"  cohort_sha256   : {split.meta['cohort_manifest_sha256']}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/splits.yaml")
    args = parser.parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    individual_missingness_mask = None
    threshold = cfg.get("individual_missingness_threshold")
    if threshold is not None:
        missingness = compute_individual_missingness(cfg["bed_path"], cfg["fam_path"])
        rule = cfg.get("individual_missingness_rule", "<")
        if rule != "<":
            raise ValueError(f"Unsupported individual_missingness_rule: {rule!r} (expected '<')")
        individual_missingness_mask = individual_qc_mask(missingness, float(threshold))

    split = build_split(
        cfg["fam_path"],
        dataset_id=cfg["dataset_id"],
        outer_split_seed=cfg["outer_split_seed"],
        fold_seed=cfg["fold_seed"],
        dev_frac=cfg["dev_frac"],
        test_frac=cfg["test_frac"],
        k_folds=cfg["k_folds"],
        cohort_min_breed_size=cfg["cohort_min_breed_size"],
        individual_missingness_mask=individual_missingness_mask,
        individual_missingness_threshold=float(threshold) if threshold is not None else None,
        individual_missingness_rule=cfg.get("individual_missingness_rule", "<"),
    )

    csv_path, meta_path = save_split(split, out_dir=cfg["out_dir"])
    print(f"Saved: {csv_path}")
    print(f"Saved: {meta_path}")
    _report(split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
