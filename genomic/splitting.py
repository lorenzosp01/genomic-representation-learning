"""Model-independent dataset splitting for the thesis experimental framework.

This module produces a single outer development/test split plus a stratified
K-fold assignment inside the development set, and persists it so that the VMGP
model, the contrastive model and the classical baselines all consume identical
partitions.

The module operates only on the PLINK ``.fam`` sample manifest. It never reads
or reorders the genotype matrix; the original row order is preserved through
``source_index``, which maps every split row back to its original genotype
matrix row.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold

PROTOCOL_VERSION = 2

CSV_COLUMNS = [
    "source_index",
    "FID",
    "IID",
    "sample_id",
    "breed",
    "outer_split",
    "fold",
]


@dataclass(frozen=True)
class SplitIndex:
    """A persisted train/validation/test partition of a cohort.

    All arrays are aligned and ordered by the original ``.fam`` row order.
    ``source_index`` holds the row index into the original (unreordered)
    genotype matrix for every retained sample.

    ``outer_split`` is ``"development"`` or ``"test"``. ``fold`` is ``0..K-1``
    for development samples and ``-1`` for test samples.
    """

    source_index: np.ndarray  # int, original matrix row for each retained sample
    fid: np.ndarray  # str
    iid: np.ndarray  # str
    sample_id: np.ndarray  # str, "FID:IID"
    breed: np.ndarray  # str, raw FID (stratification key)
    outer_split: np.ndarray  # str, "development" | "test"
    fold: np.ndarray  # int, 0..K-1 (development) or -1 (test)
    meta: Dict

    # ------------------------------------------------------------------
    # Development-time access (what CV / training / tuning should use)
    # ------------------------------------------------------------------
    def development_indices(self) -> np.ndarray:
        return self.source_index[self.outer_split == "development"]

    def fold_train_indices(self, k: int) -> np.ndarray:
        mask = (self.outer_split == "development") & (self.fold != k)
        return self.source_index[mask]

    def fold_val_indices(self, k: int) -> np.ndarray:
        mask = (self.outer_split == "development") & (self.fold == k)
        return self.source_index[mask]

    # ------------------------------------------------------------------
    # Final evaluation only (explicitly requested)
    # ------------------------------------------------------------------
    def test_indices(self) -> np.ndarray:
        return self.source_index[self.outer_split == "test"]

    @property
    def n_folds(self) -> int:
        return int(self.meta["k_folds"])

    def train_counts_per_breed(self) -> Dict[str, List[int]]:
        """Training count per breed in each fold (in-memory, no re-split).

        Returns ``{breed: [train_count_fold0, ..., train_count_foldK-1]}`` for
        every retained (development) breed.
        """
        dev = self.outer_split == "development"
        counts: Dict[str, List[int]] = {}
        for b in sorted(set(self.breed.tolist())):
            bmask = self.breed == b
            counts[b] = [
                int((dev & bmask & (self.fold != k)).sum())
                for k in range(self.n_folds)
            ]
        return counts

    def rq2_breeds(self, min_train_per_fold: int = 30) -> List[str]:
        """Breeds whose training count in every fold is >= ``min_train_per_fold``.

        This is a pure breed filter over the persisted primary split: it reuses
        the existing fold assignments and never creates a second split.
        """
        counts = self.train_counts_per_breed()
        return [b for b, folds in counts.items() if min(folds) >= min_train_per_fold]


def _read_fam(fam_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read the PLINK ``.fam`` manifest in original row order.

    Returns (source_index, fid, iid, sample_id, breed).
    """
    fam = np.loadtxt(fam_path, dtype=str, ndmin=2)
    if fam.shape[0] == 0:
        raise ValueError(f"Empty .fam file: {fam_path}")
    if fam.shape[1] < 2:
        raise ValueError(f".fam file has fewer than 2 columns: {fam_path}")

    fid = fam[:, 0]
    iid = fam[:, 1]
    n = len(fid)
    source_index = np.arange(n, dtype=np.int64)
    sample_id = np.array([f"{f}:{i}" for f, i in zip(fid, iid)], dtype=object)
    breed = fid.copy()
    return source_index, fid, iid, sample_id, breed


def _manifest_sha256(source_index, fid, iid, breed) -> str:
    """Fingerprint of the ordered source manifest.

    Depends on source_index, FID, IID, breed and their row order.
    """
    h = hashlib.sha256()
    for si, f, i, b in zip(source_index, fid, iid, breed):
        h.update(f"{si}\t{f}\t{i}\t{b}\n".encode("utf-8"))
    return h.hexdigest()


def build_split(
    fam_path: str,
    *,
    dataset_id: str,
    outer_split_seed: int = 42,
    fold_seed: int = 42,
    dev_frac: float = 0.85,
    test_frac: float = 0.15,
    k_folds: int = 3,
    cohort_min_breed_size: Optional[int] = None,
    individual_missingness_mask: Optional[np.ndarray] = None,
    individual_missingness_threshold: Optional[float] = None,
    individual_missingness_rule: str = "<",
) -> SplitIndex:
    """Build a single outer development/test split plus a K-fold assignment.

    Parameters
    ----------
    fam_path:
        Path to the PLINK ``.fam`` file (FID = column 0, IID = column 1).
    dataset_id:
        Identifier used for file naming and metadata.
    outer_split_seed, fold_seed:
        Independent seeds for the outer dev/test split and the inner K-fold.
    dev_frac, test_frac:
        Target development/test fractions (must sum to 1.0).
    k_folds:
        Number of stratified folds inside the development set.
    cohort_min_breed_size:
        A-priori breed-eligibility threshold. Breeds with fewer individuals
        are excluded. ``None`` means no breed eligibility filtering. This is
        *only* an inclusion rule and is unrelated to the RQ2 training sample
        size N.
    individual_missingness_mask:
        Optional boolean keep-mask (length = raw sample count) encoding
        pre-split sample-level quality (e.g. ``missingness < threshold``). This
        is applied BEFORE breed eligibility. ``None`` means no sample-level QC.
    individual_missingness_threshold:
        The sample-missingness threshold used to build ``individual_missingness_mask``
        (recorded in metadata and used to derive the artifact cohort label).
    individual_missingness_rule:
        The comparison rule (default ``"<"``), recorded in metadata.
    """
    if not 0.0 < test_frac < 1.0:
        raise ValueError("test_frac must be in (0, 1)")
    if not 0.0 < dev_frac < 1.0:
        raise ValueError("dev_frac must be in (0, 1)")
    if abs((dev_frac + test_frac) - 1.0) > 1e-9:
        raise ValueError("dev_frac + test_frac must equal 1.0")
    if k_folds < 2:
        raise ValueError("k_folds must be >= 2")
    if cohort_min_breed_size is not None and cohort_min_breed_size < 1:
        raise ValueError("cohort_min_breed_size must be >= 1")
    if individual_missingness_threshold is not None and not 0.0 < individual_missingness_threshold < 1.0:
        raise ValueError("individual_missingness_threshold must be in (0, 1)")

    source_index, fid, iid, sample_id, breed = _read_fam(fam_path)
    n_total = len(source_index)

    duplicates = [s for s, c in Counter(sample_id.tolist()).items() if c > 1]
    if duplicates:
        raise ValueError(
            f"Duplicate sample_id (FID:IID) found in {fam_path}: {duplicates[:10]}"
        )

    manifest_sha = _manifest_sha256(source_index, fid, iid, breed)

    # --- Pre-split individual sample-level QC ----------------------------
    if individual_missingness_mask is not None:
        iqc_keep = np.asarray(individual_missingness_mask, dtype=bool)
        if iqc_keep.shape != (n_total,):
            raise ValueError(
                "individual_missingness_mask must have one entry per raw sample "
                f"(expected {n_total}, got {iqc_keep.size})."
            )
    else:
        iqc_keep = np.ones(n_total, dtype=bool)
    n_after_iqc = int(iqc_keep.sum())

    # --- A-priori breed cohort eligibility ------------------------------
    if cohort_min_breed_size is not None:
        counts = Counter(breed[iqc_keep].tolist())
        keep = iqc_keep & np.array([counts[b] >= cohort_min_breed_size for b in breed.tolist()])
    else:
        keep = iqc_keep
    elig = np.flatnonzero(keep)
    n_eligible = int(elig.size)
    elig_breed = breed[keep]

    # --- Validate sufficiency for outer split + K-fold ------------------
    counts_elig = Counter(elig_breed.tolist())
    bad = {b: c for b, c in counts_elig.items() if c < k_folds + 1}
    if bad:
        raise ValueError(
            "Insufficient samples per breed to support the outer "
            f"development/test split and {k_folds}-fold stratification. "
            f"Each retained breed needs at least k_folds+1 = {k_folds + 1} "
            f"individuals. Problematic breeds (breed: count): {bad}"
        )

    # --- Outer development/test split (stratified by breed) -------------
    rng = np.random.default_rng(outer_split_seed)
    test_mask = np.zeros(n_eligible, dtype=bool)
    for b in sorted(counts_elig):
        pos = np.flatnonzero(elig_breed == b)
        c = len(pos)
        n_test_b = int(min(max(1, round(test_frac * c)), c - k_folds))
        perm = rng.permutation(c)
        test_mask[pos[perm[:n_test_b]]] = True

    outer_split = np.where(test_mask, "test", "development")

    # --- Stratified K-fold inside development ---------------------------
    dev_pos = np.flatnonzero(~test_mask)
    fold = np.full(n_eligible, -1, dtype=np.int64)
    if dev_pos.size:
        dev_breed = elig_breed[dev_pos]
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=fold_seed)
        for k, (_, val_idx) in enumerate(skf.split(np.zeros(dev_pos.size), dev_breed)):
            fold[dev_pos[val_idx]] = k

    cohort_label = ""
    if individual_missingness_threshold is not None:
        cohort_label = "indmiss" + f"{individual_missingness_threshold:.2f}".replace(".", "p")

    meta = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset_id": dataset_id,
        "cohort_label": cohort_label,
        "individual_missingness_threshold": individual_missingness_threshold,
        "individual_missingness_rule": individual_missingness_rule if individual_missingness_threshold is not None else None,
        "outer_split_seed": outer_split_seed,
        "fold_seed": fold_seed,
        "n_folds": k_folds,
        "k_folds": k_folds,
        "dev_frac": dev_frac,
        "test_frac": test_frac,
        "cohort_min_breed_size": cohort_min_breed_size,
        "raw_sample_count": n_total,
        "eligible_after_individual_qc_count": n_after_iqc,
        "final_primary_cohort_count": n_eligible,
        "n_total": n_total,
        "n_eligible": n_eligible,
        "n_development": int(np.sum(~test_mask)),
        "n_test": int(np.sum(test_mask)),
        "source_manifest_sha256": manifest_sha,
        "cohort_manifest_sha256": _manifest_sha256(source_index[keep], fid[keep], iid[keep], breed[keep]),
    }

    return SplitIndex(
        source_index=elig,
        fid=fid[keep],
        iid=iid[keep],
        sample_id=sample_id[keep],
        breed=elig_breed,
        outer_split=outer_split,
        fold=fold,
        meta=meta,
    )


def _split_stem(meta: Dict) -> str:
    label = meta.get("cohort_label") or ""
    base = meta["dataset_id"] + (f"_{label}" if label else "")
    return f"{base}_outer{meta['outer_split_seed']}_fold{meta['fold_seed']}"


def save_split(split: SplitIndex, out_dir: str = "splits") -> Tuple[str, str]:
    """Persist a split as CSV + JSON metadata. Returns (csv_path, meta_path)."""
    os.makedirs(out_dir, exist_ok=True)
    stem = _split_stem(split.meta)
    csv_path = os.path.join(out_dir, stem + ".csv")
    meta_path = os.path.join(out_dir, stem + ".meta.json")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        for si, fi, ii, sid, br, os_, fo in zip(
            split.source_index,
            split.fid,
            split.iid,
            split.sample_id,
            split.breed,
            split.outer_split,
            split.fold,
        ):
            writer.writerow([si, fi, ii, sid, br, os_, fo])

    with open(meta_path, "w") as f:
        json.dump(split.meta, f, indent=2, sort_keys=True)

    return csv_path, meta_path


def load_split(
    dataset_id: str,
    outer_split_seed: int,
    fold_seed: int,
    fam_path: str,
    out_dir: str = "splits",
    cohort_label: str = "",
) -> SplitIndex:
    """Load a persisted split and verify it against the current ``.fam``.

    The manifest fingerprint is recomputed from ``fam_path`` and compared to the
    stored value; a mismatch means the source dataset changed since the split
    was created. The cohort fingerprint (over the eligible samples) is also
    recomputed from the CSV and verified.
    """
    base = dataset_id + (f"_{cohort_label}" if cohort_label else "")
    stem = f"{base}_outer{outer_split_seed}_fold{fold_seed}"
    csv_path = os.path.join(out_dir, stem + ".csv")
    meta_path = os.path.join(out_dir, stem + ".meta.json")
    if not (os.path.exists(csv_path) and os.path.exists(meta_path)):
        raise FileNotFoundError(f"Split files not found: {csv_path}, {meta_path}")

    with open(meta_path) as f:
        meta = json.load(f)

    source_index, fid, iid, sample_id, breed = _read_fam(fam_path)
    sha = _manifest_sha256(source_index, fid, iid, breed)
    if sha != meta["source_manifest_sha256"]:
        raise ValueError(
            "Source manifest fingerprint mismatch: the .fam file has changed "
            "since this split was created."
        )

    rows: List[dict] = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    si = np.array([int(r["source_index"]) for r in rows], dtype=np.int64)
    F = np.array([r["FID"] for r in rows], dtype=object)
    I = np.array([r["IID"] for r in rows], dtype=object)
    S = np.array([r["sample_id"] for r in rows], dtype=object)
    B = np.array([r["breed"] for r in rows], dtype=object)
    O = np.array([r["outer_split"] for r in rows], dtype=object)
    Fo = np.array([int(r["fold"]) for r in rows], dtype=np.int64)

    if len(rows) != meta["n_eligible"]:
        raise ValueError("Split CSV row count does not match metadata n_eligible.")
    n_test = int((O == "test").sum())
    n_dev = int((O == "development").sum())
    if n_test != meta["n_test"] or n_dev != meta["n_development"]:
        raise ValueError("Split CSV outer_split counts do not match metadata.")

    if "cohort_manifest_sha256" in meta:
        cohort_sha = _manifest_sha256(si, F, I, B)
        if cohort_sha != meta["cohort_manifest_sha256"]:
            raise ValueError(
                "Cohort manifest fingerprint mismatch: the eligible cohort "
                "differs from the one used to create this split."
            )

    return SplitIndex(
        source_index=si,
        fid=F,
        iid=I,
        sample_id=S,
        breed=B,
        outer_split=O,
        fold=Fo,
        meta=meta,
    )
