#!/usr/bin/env python
"""Protocol-v2 split generator for the ISGC Sheep HapMap cohort (Phase 0).

Thesis design is goat-primary: ADAPTmap (caprine) is the primary reference
cohort, while this ovine cohort is the cross-species generalization validation
cohort (Chapter 6.5). The protocol is therefore mirrored EXACTLY from
``configs/splits.yaml``:

    1. pre-split individual QC: keep iff ``missingness < 0.10``
       (Protocol-v2 frozen rule; the same rule recorded in the goat metadata);
    2. breed eligibility: ``>= 30`` individuals POST-QC (frozen decision:
       any breed falling below 30 after QC is dropped);
    3. stratified outer development/test split (85/15, seed 42) and 3-fold
       cross-validation inside development (seed 42).

Splits are explicit artifacts: existing files are never silently replaced
(use ``--force`` to regenerate after an explicit protocol decision).

Usage:
    python scripts/make_split_sheep.py
    python scripts/make_split_sheep.py --force
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import yaml

# Make the repository root importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genomic.cohort import compute_individual_missingness, individual_qc_mask
from genomic.splitting import build_split, save_split

CONFIG_PATH = "configs/splits_sheep.yaml"
EXPECTED_DATASET_ID = "ISGC_sheep_hapmap"


def _read_breeds(fam_path: str) -> np.ndarray:
    """Breed labels from the PLINK ``.fam`` FID column (sheep convention)."""
    fam = np.loadtxt(fam_path, dtype=str, ndmin=2)
    if fam.ndim != 2 or fam.shape[1] < 2:
        raise ValueError(f"Malformed .fam file: {fam_path}")
    return fam[:, 0].astype(str)


def _cohort_stem(cfg: dict) -> str:
    label = "indmiss" + f"{float(cfg['individual_missingness_threshold']):.2f}".replace(
        ".", "p"
    )
    return (
        f"{cfg['dataset_id']}_{label}"
        f"_outer{int(cfg['outer_split_seed'])}_fold{int(cfg['fold_seed'])}"
    )


def _validate_config(cfg: dict) -> None:
    if cfg.get("dataset_id") != EXPECTED_DATASET_ID:
        raise ValueError(
            f"config dataset_id must be {EXPECTED_DATASET_ID!r}, "
            f"got {cfg.get('dataset_id')!r}"
        )
    if cfg.get("individual_missingness_rule") != "<":
        raise ValueError(
            "Protocol-v2 supports only individual_missingness_rule '<'; "
            f"got {cfg.get('individual_missingness_rule')!r}"
        )
    if int(cfg.get("cohort_min_breed_size", 0)) != 30:
        raise ValueError("Frozen sheep cohort rule is cohort_min_breed_size == 30.")
    if abs(float(cfg["dev_frac"]) + float(cfg["test_frac"]) - 1.0) > 1e-9:
        raise ValueError("dev_frac + test_frac must equal 1.0")
    if cfg.get("rq2_executed", False):
        raise ValueError("RQ2 is intentionally not executed on the sheep cohort.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument(
        "--force", action="store_true",
        help="overwrite an existing sheep split with the same stem",
    )
    args = parser.parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    _validate_config(cfg)

    out_dir = cfg["out_dir"]
    stem = _cohort_stem(cfg)
    csv_path = os.path.join(out_dir, stem + ".csv")
    meta_path = os.path.join(out_dir, stem + ".meta.json")
    if not args.force and (os.path.exists(csv_path) or os.path.exists(meta_path)):
        raise FileExistsError(
            f"Split already exists ({csv_path}). Refusing to overwrite silently; "
            "pass --force after an explicit protocol decision."
        )
    os.makedirs(out_dir, exist_ok=True)

    # --- 1. Individual QC (Protocol-v2: missingness < threshold) ----------
    breeds = _read_breeds(cfg["fam_path"])
    n_snps = sum(1 for _ in open(cfg["bim_path"]))
    missingness = compute_individual_missingness(cfg["bed_path"], cfg["fam_path"])
    if missingness.shape != breeds.shape:
        raise ValueError("missingness length does not match .fam rows")
    threshold = float(cfg["individual_missingness_threshold"])
    iqc_keep = individual_qc_mask(missingness, threshold)
    n_at_threshold = int(np.sum(np.isclose(missingness, threshold, atol=1e-12)))
    n_at_or_above = int(np.sum(missingness >= threshold))
    pre_counts = Counter(breeds.tolist())
    post_counts = Counter(breeds[iqc_keep].tolist())

    # --- 2 + 3. Post-QC breed eligibility, outer split, K-fold ------------
    split = build_split(
        cfg["fam_path"],
        dataset_id=cfg["dataset_id"],
        outer_split_seed=int(cfg["outer_split_seed"]),
        fold_seed=int(cfg["fold_seed"]),
        dev_frac=float(cfg["dev_frac"]),
        test_frac=float(cfg["test_frac"]),
        k_folds=int(cfg["k_folds"]),
        cohort_min_breed_size=int(cfg["cohort_min_breed_size"]),
        individual_missingness_mask=iqc_keep,
        individual_missingness_threshold=threshold,
        individual_missingness_rule=cfg["individual_missingness_rule"],
    )
    save_split(split, out_dir=out_dir)

    # --- Independent verification of the generated split ------------------
    retained = sorted(set(split.breed.tolist()))
    expected_retained = sorted(b for b, c in post_counts.items() if c >= 30)
    if retained != expected_retained:
        raise RuntimeError(
            "Generated split breeds do not match post-QC eligibility: "
            f"only_in_split={sorted(set(retained) - set(expected_retained))}, "
            f"missing_from_split={sorted(set(expected_retained) - set(retained))}"
        )
    if len(split.source_index) != int(split.meta["n_eligible"]):
        raise RuntimeError("split row count mismatch")

    train_counts = split.train_counts_per_breed()
    min_train_per_breed = {b: int(min(c)) for b, c in train_counts.items()}
    global_min_train = min(min_train_per_breed.values())
    breeds_at_min = sorted(b for b, v in min_train_per_breed.items() if v == global_min_train)
    dev_mask = split.outer_split == "development"
    fold_sizes = {
        int(k): int(((split.fold == k) & dev_mask).sum())
        for k in range(int(cfg["k_folds"]))
    }
    test_counts = Counter(split.breed[~dev_mask].tolist())
    dropped_by_qc = {
        b: {"pre_qc": int(pre_counts[b]), "post_qc": int(post_counts.get(b, 0))}
        for b in pre_counts
        if pre_counts[b] >= int(cfg["cohort_min_breed_size"])
        and post_counts.get(b, 0) < int(cfg["cohort_min_breed_size"])
    }
    dropped_pre_ge30 = sorted(
        (b for b, c in pre_counts.items() if c >= 30), key=lambda b: (post_counts.get(b, 0), b)
    )

    # --- Detailed Phase 0 report ------------------------------------------
    line = "=" * 78
    print(line)
    print("PHASE 0 — ISGC Sheep HapMap cohort & Protocol-v2 split")
    print(line)
    print(f"config                      : {args.config}")
    print(f"dataset_id                  : {cfg['dataset_id']}")
    print(f"raw samples (.fam rows)     : {len(breeds)}")
    print(f"raw SNPs                    : {n_snps}")
    print(f"individual QC rule          : missingness < {threshold:.2f} (Protocol-v2, goat-identical)")
    print(f"  missingness min/median/max: {missingness.min():.5f} / "
          f"{np.median(missingness):.5f} / {missingness.max():.5f}")
    print(f"  removed by individual QC  : {int((~iqc_keep).sum())} "
          f"({100.0 * (~iqc_keep).mean():.3f}%)")
    print(f"  missingness == threshold  : {n_at_threshold} (nothing changes with <=)")
    print(f"  missingness >= threshold  : {n_at_or_above}")
    print(f"breeds before QC            : {len(pre_counts)}")
    print(f"breeds after QC (>=1 sample): {len(post_counts)}")
    print(f"breeds eligible (>=30 post-QC): {len(expected_retained)}")
    print(f"total eligible animals      : {int(split.meta['n_eligible'])}")

    if dropped_by_qc:
        print("\nbreeds dropped by the POST-QC >=30 rule (were >=30 pre-QC):")
        for b in sorted(dropped_by_qc):
            d = dropped_by_qc[b]
            print(f"  {b:34s} pre={d['pre_qc']:3d} -> post={d['post_qc']:3d}")
    else:
        print("\nbreeds dropped by the POST-QC >=30 rule (were >=30 pre-QC): none")

    print("\neligible breeds and per-fold TRAIN counts:")
    for b in sorted(expected_retained, key=lambda x: (-post_counts[x], x)):
        counts = [int(c) for c in train_counts[b]]
        mn = min_train_per_breed[b]
        flag = "  <-- global min" if mn == global_min_train else ""
        print(f"  {b:34s} post_qc={post_counts[b]:3d} | "
              f"train_folds={counts} | min_train={mn:3d}{flag}")
    print(f"\nglobal minimum per-breed TRAIN count: {global_min_train} "
          f"({', '.join(breeds_at_min)})")
    print(f"minimum per-breed TEST count        : "
          f"{min(test_counts.values())} "
          f"({', '.join(sorted(b for b, c in test_counts.items() if c == min(test_counts.values())))})")

    print("\nsplit summary:")
    print(f"  outer split            : dev_frac={cfg['dev_frac']} test_frac={cfg['test_frac']} "
          f"seeds={cfg['outer_split_seed']}/{cfg['fold_seed']}")
    print(f"  n_development          : {int(split.meta['n_development'])}")
    print(f"  n_test                 : {int(split.meta['n_test'])}")
    print(f"  fold sizes             : {fold_sizes}")
    print(f"  source_manifest_sha256 : {split.meta['source_manifest_sha256']}")
    print(f"  cohort_manifest_sha256 : {split.meta['cohort_manifest_sha256']}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {meta_path}")

    report = {
        "phase": 0,
        "dataset_id": cfg["dataset_id"],
        "thesis_design": "goat-primary; sheep = cross-species generalization validation",
        "config": cfg,
        "raw_sample_count": len(breeds),
        "raw_snp_count": 49034,
        "individual_qc": {
            "threshold": threshold,
            "rule": "<",
            "removed": int((~iqc_keep).sum()),
            "at_threshold": n_at_threshold,
            "at_or_above": n_at_or_above,
            "missingness_min": float(missingness.min()),
            "missingness_median": float(np.median(missingness)),
            "missingness_max": float(missingness.max()),
        },
        "breeds_before_qc": len(pre_counts),
        "breeds_after_qc": len(post_counts),
        "breeds_dropped_by_post_qc_rule": dropped_by_qc,
        "breeds_pre_ge30": {
            b: {"pre_qc": int(pre_counts[b]), "post_qc": int(post_counts.get(b, 0))}
            for b in dropped_pre_ge30
        },
        "eligible_breeds": {
            b: {
                "post_qc": int(post_counts[b]),
                "train_counts": [int(c) for c in train_counts[b]],
                "min_train": min_train_per_breed[b],
                "test_count": int(test_counts.get(b, 0)),
            }
            for b in expected_retained
        },
        "eligible_animal_count": int(split.meta["n_eligible"]),
        "global_min_train_per_breed": global_min_train,
        "breeds_at_global_min_train": breeds_at_min,
        "n_development": int(split.meta["n_development"]),
        "n_test": int(split.meta["n_test"]),
        "fold_sizes": fold_sizes,
        "split_meta": split.meta,
        "artifacts": {"csv": csv_path, "meta": meta_path},
        "rq2_executed": False,
        "rq2_limitation": (
            "minimum eligible breed count is 30; edge breeds leave ~20 training "
            "individuals per breed per fold, below the RQ2 grid (5..30)"
        ),
        "locked_test_accessed": False,
    }
    report_path = os.path.join(out_dir, stem + "_phase0_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(f"Saved: {report_path}")
    print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
