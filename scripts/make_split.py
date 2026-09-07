#!/usr/bin/env python
"""Explicitly generate and persist a train/validation/test split from a .fam file.

Splits are produced on demand by this script and written under ``splits/``.
They are never created or silently replaced during DataModule setup.

Usage:
    python scripts/make_split.py [--config configs/splits.yaml]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import yaml

# Make the repository root importable so ``import genomic`` works regardless
# of the directory the script is invoked from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genomic.splitting import build_split, save_split


def _report(split) -> None:
    dev = split.development_indices()
    test = split.test_indices()
    print("=" * 64)
    print("Split generated")
    print("=" * 64)
    print(f"  dataset_id      : {split.meta['dataset_id']}")
    print(f"  outer_split_seed: {split.meta['outer_split_seed']}")
    print(f"  fold_seed       : {split.meta['fold_seed']}")
    print(f"  k_folds         : {split.meta['k_folds']}")
    print(f"  cohort_min_breed_size   : {split.meta['cohort_min_breed_size']}")
    print(f"  n_total         : {split.meta['n_total']}")
    print(f"  n_eligible      : {split.meta['n_eligible']}")
    print(f"  n_development   : {split.meta['n_development']}")
    print(f"  n_test          : {split.meta['n_test']}")
    print(f"  breeds          : {len(set(split.breed.tolist()))}")
    print(f"  sha256          : {split.meta['source_manifest_sha256']}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/splits.yaml")
    args = parser.parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    split = build_split(
        cfg["fam_path"],
        dataset_id=cfg["dataset_id"],
        outer_split_seed=cfg["outer_split_seed"],
        fold_seed=cfg["fold_seed"],
        dev_frac=cfg["dev_frac"],
        test_frac=cfg["test_frac"],
        k_folds=cfg["k_folds"],
        cohort_min_breed_size=cfg["cohort_min_breed_size"],
    )

    csv_path, meta_path = save_split(split, out_dir=cfg["out_dir"])
    print(f"Saved: {csv_path}")
    print(f"Saved: {meta_path}")
    _report(split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
