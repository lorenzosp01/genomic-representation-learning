#!/usr/bin/env python
"""Ranking-overlap analysis for RQ4 (protocol-v2 artifacts).

Reads the persisted top-SNP manifests produced by ``scripts/run_rq3.py`` and
prints, for one fold:

1. the top-N SNP IDs (.bim names) prioritized by each ranking method;
2. the pairwise Jaccard overlap ``|A ∩ B| / |A ∪ B|`` between the methods for
   the requested panel sizes K.

Usage:
    python scripts/analyze_ranking_overlap.py
    python scripts/analyze_ranking_overlap.py --fold 0 --ks 50 200 500 --top 5
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_METHODS = ("fst", "random_forest", "vmgp_shap", "contrastive_ig")
DEFAULT_RANKINGS_DIR = "results/rq3_marker_efficiency/rankings"


def load_manifest(rankings_dir: str, method: str, fold: int) -> pd.DataFrame:
    path = os.path.join(rankings_dir, f"{method}_fold{fold}_top.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing ranking manifest: {path}")
    df = pd.read_csv(path)
    required = {"rank", "snp_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    return df.sort_values("rank").reset_index(drop=True)


def jaccard(a: set, b: set) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rankings-dir", default=DEFAULT_RANKINGS_DIR)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--ks", type=int, nargs="+", default=[50, 200, 500])
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args(argv)

    methods = list(args.methods)
    manifests = {m: load_manifest(args.rankings_dir, m, args.fold) for m in methods}
    max_k = max(args.ks)

    print("=" * 78)
    print(f"Ranking analysis — fold {args.fold} — {args.rankings_dir}")
    print("=" * 78)

    # 1. Top-N SNP IDs per method ------------------------------------------
    print(f"\nTop {args.top} SNP IDs per method (fold {args.fold})\n")
    top_ids = {
        m: manifests[m]["snp_id"].head(args.top).tolist() for m in methods
    }
    top_table = pd.DataFrame(
        top_ids, index=pd.RangeIndex(1, args.top + 1, name="rank")
    )
    print(top_table.to_string())

    # 2. Pairwise Jaccard matrices -----------------------------------------
    for k in args.ks:
        if k > len(manifests[methods[0]]):
            raise ValueError(
                f"K={k} exceeds the {len(manifests[methods[0]])} stored ranks."
            )
        sets = {m: set(manifests[m]["snp_id"].head(k)) for m in methods}
        matrix = pd.DataFrame(
            [[100.0 * jaccard(sets[a], sets[b]) for b in methods] for a in methods],
            index=methods,
            columns=methods,
        )
        print(f"\nJaccard overlap (%) — K = {k}  (|A ∩ B| / |A ∪ B|)\n")
        print(matrix.to_string(float_format=lambda v: f"{v:6.1f}"))

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
