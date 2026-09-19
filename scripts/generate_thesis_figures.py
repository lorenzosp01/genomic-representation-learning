#!/usr/bin/env python
"""Generate thesis-ready figures from protocol-v2 result artifacts.

Headless (Agg) and non-interactive: every figure is written as vector
PDF + 300-DPI PNG under ``thesis/assets/``. Figures whose source run is
missing or incomplete are skipped with a warning; the script never
crashes for a missing artifact.

Figures produced (batch 1):
    fig_rq1_latentdim_selection
    fig_rq1_confusion_locked
    fig_rq2_learning_curve
    fig_rq3_panel_curves
    fig_rq4_jaccard
    fig_cross_species_rq1

Usage:
    uv run python scripts/generate_thesis_figures.py
"""

from __future__ import annotations

import json
import os
import sys
import traceback

import numpy as np

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
RESULTS = os.path.join(REPO, "results")
OUT_DIR = os.path.join(REPO, "thesis", "assets")

METHOD_ORDER = ["fst", "random_forest", "vmgp_shap", "contrastive_ig"]
METHOD_LABELS = {
    "fst": r"$F_{ST}$",
    "random_forest": "Random Forest",
    "vmgp_shap": "VMGP SHAP",
    "contrastive_ig": "Contrastive IG",
    "random": "Random panels",
}
METHOD_COLORS = {
    "fst": "#1f77b4",
    "random_forest": "#ff7f0e",
    "vmgp_shap": "#2ca02c",
    "contrastive_ig": "#d62728",
    "random": "#7f7f7f",
}
SPECIES_COLORS = {"goat": "#4c72b0", "sheep": "#dd8452"}

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "savefig.dpi": 300,
    "figure.dpi": 120,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_json(path):
    with open(path) as f:
        return json.load(f)


def save(fig, stem):
    os.makedirs(OUT_DIR, exist_ok=True)
    pdf = os.path.join(OUT_DIR, stem + ".pdf")
    png = os.path.join(OUT_DIR, stem + ".png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok]   {stem}.pdf + {stem}.png")


def warn(msg):
    print(f"[warn] {msg}")


def _series(aggs, method, key="macro_f1"):
    rows = sorted((r for r in aggs if r["method"] == method), key=lambda r: r["K"])
    return (
        np.array([r["K"] for r in rows]),
        np.array([r[f"{key}_mean"] for r in rows]),
        np.array([r[f"{key}_std"] for r in rows]),
    )


# ---------------------------------------------------------------------------
# 1. VMGP latent-dimension selection (goat)
# ---------------------------------------------------------------------------
def fig_rq1_latentdim_selection():
    src = os.path.join(RESULTS, "rq1_vae_grid_summary.json")
    if not os.path.exists(src):
        warn(f"missing {src}; skipping fig_rq1_latentdim_selection")
        return
    summary = load_json(src)["summary"]
    dims = sorted(int(k) for k in summary)
    metrics = [
        ("Macro_F1", "Macro-F1", "o-", "C0"),
        ("Balanced_Accuracy", "Balanced accuracy", "s--", "C1"),
        ("Accuracy", "Accuracy", "^:", "C2"),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for key, label, style, color in metrics:
        mean = [summary[str(d)][f"{key}_mean"] for d in dims]
        std = [summary[str(d)][f"{key}_std"] for d in dims]
        ax.errorbar(dims, mean, yerr=std, fmt=style, color=color,
                    capsize=3, linewidth=1.8, markersize=5, label=label)
    ax.axvline(96, color="grey", linestyle=":", linewidth=1.2)
    mf96 = summary["96"]["Macro_F1_mean"]
    ax.annotate("selected $d=96$", xy=(96, mf96), xytext=(70, mf96 - 0.004),
                fontsize=8, arrowprops=dict(arrowstyle="->", color="grey", lw=0.8))
    ax.set_xlabel("Latent dimension $d$")
    ax.set_ylabel("Development CV metric")
    ax.set_xticks(dims)
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, "fig_rq1_latentdim_selection")


# ---------------------------------------------------------------------------
# 2. Locked-test confusion matrices (goat)
# ---------------------------------------------------------------------------
def fig_rq1_confusion_locked():
    srcs = {
        "VMGP $d=96$": os.path.join(RESULTS, "rq1_final", "vae_final_summary.json"),
        "Contrastive emb$=3$": os.path.join(RESULTS, "rq1_final",
                                            "contrastive_final_summary.json"),
    }
    if not all(os.path.exists(p) for p in srcs.values()):
        warn("missing locked-test RQ1 summaries; skipping fig_rq1_confusion_locked")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.2))
    im = None
    for ax, (title, path) in zip(axes, srcs.items()):
        d = load_json(path)
        cm = np.asarray(d["confusion_matrix"], dtype=np.float64)
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
        im = ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=1.0)
        n = cm.shape[0]
        ticks = list(range(n))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([str(i) for i in ticks], rotation=90, fontsize=7)
        ax.set_yticklabels([str(i) for i in ticks], fontsize=7)
        ax.set_xlabel("predicted class index")
        ax.set_ylabel("true class index")
        ax.set_title(f"{title} (Macro-F1 = {d['macro_f1']:.3f})")
        ax.grid(False)
    fig.colorbar(im, ax=list(axes), fraction=0.025, pad=0.02,
                 label="row-normalized count")
    save(fig, "fig_rq1_confusion_locked")


# ---------------------------------------------------------------------------
# 3. RQ2 sample-efficiency learning curve (goat)
# ---------------------------------------------------------------------------
def fig_rq2_learning_curve():
    src = os.path.join(RESULTS, "rq2_sample_efficiency", "summary.json")
    if not os.path.exists(src):
        warn(f"missing {src}; skipping fig_rq2_learning_curve")
        return
    s = load_json(src)
    aggs = sorted(s["aggregates"], key=lambda r: r["N"])
    N = np.array([r["N"] for r in aggs])
    mean = np.array([r["macro_f1_mean"] for r in aggs])
    std = np.array([r["macro_f1_std"] for r in aggs])

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.errorbar(N, mean, yerr=std, fmt="o-", color="C0", capsize=3,
                linewidth=2.0, markersize=5, label="Macro-F1 (15 runs)")
    ax.axhline(s["S_ref"], color="grey", linestyle="--", linewidth=1.2,
               label=f"$S_{{\\mathrm{{ref}}}}$ ($N={s['reference_N']}$) = {s['S_ref']:.4f}")
    ax.axhline(s["threshold"], color="green", linestyle=":", linewidth=1.4,
               label=f"98% threshold = {s['threshold']:.4f}")
    if s.get("N_min") is not None:
        ax.axvline(s["N_min"], color="C3", linestyle=":", linewidth=1.5)
        ymin = min(mean) - 0.01
        ax.annotate(f"$N_{{\\min}} = {s['N_min']}$", xy=(s["N_min"], ymin),
                    xytext=(s["N_min"] + 2.5, ymin), color="C3", fontsize=9,
                    arrowprops=dict(arrowstyle="->", color="C3", lw=1.0))
    ax.set_xlabel("Training individuals per breed ($N$)")
    ax.set_ylabel("Macro-F1")
    ax.set_xticks(N)
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, "fig_rq2_learning_curve")


# ---------------------------------------------------------------------------
# 4. RQ3 marker-efficiency curves (goat)
# ---------------------------------------------------------------------------
def fig_rq3_panel_curves():
    src = os.path.join(RESULTS, "rq3_marker_efficiency", "summary.json")
    if not os.path.exists(src):
        warn(f"missing {src}; skipping fig_rq3_panel_curves")
        return
    s = load_json(src)
    aggs = s["aggregates"]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    handles = []
    for m in METHOD_ORDER:
        K, mean, std = _series(aggs, m)
        if not len(K):
            continue
        color = METHOD_COLORS[m]
        ax.errorbar(K, mean, yerr=std, fmt="o-", color=color, capsize=2.5,
                    linewidth=1.8, markersize=4.5, label=METHOD_LABELS[m])
        pm = s.get("p_min", {}).get(m, {}).get("0.02")
        if pm is not None and pm in set(K.tolist()):
            j = int(np.where(K == pm)[0][0])
            ax.scatter([pm], [mean[j]], marker="*", s=130, color=color,
                       edgecolor="black", linewidth=0.6, zorder=5)
            handles.append(Line2D([], [], marker="*", linestyle="",
                                  color=color, markeredgecolor="black",
                                  markersize=10, label=f"$P_{{\\min}}$ {METHOD_LABELS[m]}"))
    K, rmean, rstd = _series(aggs, "random")
    if len(K):
        ax.fill_between(K, rmean - rstd, rmean + rstd, color="grey", alpha=0.18)
        ax.plot(K, rmean, color="grey", linestyle="--", linewidth=1.2,
                label=METHOD_LABELS["random"])
    s_full = s["s_full"]["macro_f1_mean"]
    ax.axhline(s_full, color="black", linestyle="--", linewidth=1.4,
               label=f"$S_{{\\mathrm{{full}}}}$ = {s_full:.4f}")
    ax.set_xscale("log")
    ax.set_xticks([r["K"] for r in aggs if r["method"] == "fst"])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Panel size $K$ (SNPs)")
    ax.set_ylabel("Macro-F1")
    leg1 = ax.legend(loc="lower right", ncol=1)
    ax.add_artist(leg1)
    if handles:
        ax.legend(handles=handles, loc="upper left", ncol=1)
    fig.tight_layout()
    save(fig, "fig_rq3_panel_curves")


# ---------------------------------------------------------------------------
# 5. RQ4 Jaccard overlap matrices (mean over folds)
# ---------------------------------------------------------------------------
def _jaccard(a, b):
    union = a | b
    return 100.0 * len(a & b) / len(union) if union else 0.0


def fig_rq4_jaccard():
    import pandas as pd

    ranks_dir = os.path.join(RESULTS, "rq3_marker_efficiency", "rankings")
    Ks = (50, 200, 500)
    folds = (0, 1, 2)
    paths = {
        (m, f): os.path.join(ranks_dir, f"{m}_fold{f}_top.csv")
        for m in METHOD_ORDER for f in folds
    }
    if not all(os.path.exists(p) for p in paths.values()):
        warn("missing ranking manifests; skipping fig_rq4_jaccard")
        return
    data = {k: pd.read_csv(p).sort_values("rank")["snp_id"].tolist()
            for k, p in paths.items()}

    fig, axes = plt.subplots(1, len(Ks), figsize=(13.0, 4.2))
    labels = ["FST", "RF", "SHAP", "IG"]
    im = None
    for ax, k in zip(axes, Ks):
        mat = np.zeros((len(METHOD_ORDER), len(METHOD_ORDER)))
        for i, mi in enumerate(METHOD_ORDER):
            for j, mj in enumerate(METHOD_ORDER):
                vals = [
                    _jaccard(set(data[(mi, f)][:k]), set(data[(mj, f)][:k]))
                    for f in folds
                ]
                mat[i, j] = float(np.mean(vals))
        im = ax.imshow(mat, cmap="viridis", vmin=0.0, vmax=20.0)
        for i in range(len(METHOD_ORDER)):
            for j in range(len(METHOD_ORDER)):
                ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center",
                        fontsize=8, color="white" if mat[i, j] < 10 else "black")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_title(f"$K={k}$")
        ax.grid(False)
    fig.colorbar(im, ax=list(axes), fraction=0.025, pad=0.02,
                 label="Jaccard overlap (%, mean over folds)")
    chance = {k: 100.0 * k / (2 * 46442 - k) for k in Ks}
    print("       chance-level Jaccard (%): "
          + ", ".join(f"K={k}: {v:.3f}" for k, v in chance.items()))
    save(fig, "fig_rq4_jaccard")


# ---------------------------------------------------------------------------
# 6. Cross-species RQ1 comparison (goat vs sheep)
# ---------------------------------------------------------------------------
def _goat_cv():
    out = {}
    grid = os.path.join(RESULTS, "rq1_vae_grid_summary.json")
    if os.path.exists(grid):
        s = load_json(grid)["summary"]["96"]
        out["vae"] = (s["Macro_F1_mean"], s["Macro_F1_std"])
    con = os.path.join(RESULTS, "rq1_contrastive_summary.json")
    if os.path.exists(con):
        a = load_json(con)["aggregate"]
        out["contrastive"] = (a["Macro_F1_mean"], a["Macro_F1_std"])
    rq3 = os.path.join(RESULTS, "rq3_marker_efficiency", "summary.json")
    if os.path.exists(rq3):
        f = load_json(rq3)["s_full"]
        out["rf"] = (f["macro_f1_mean"], f["macro_f1_std"])
    return out


def _locked(path, key="macro_f1"):
    if os.path.exists(path):
        return load_json(path)[key]
    return None


def fig_cross_species_rq1():
    sheep_path = os.path.join(RESULTS, "sheep_cross_species", "rq1", "summary.json")
    if not os.path.exists(sheep_path):
        warn("missing sheep RQ1 summary; skipping fig_cross_species_rq1")
        return
    goat_cv = _goat_cv()
    sheep = load_json(sheep_path)
    sheep_cv = sheep["cv"]
    sheep_fin = sheep.get("finals", {})
    goat_fin = {
        "vae": _locked(os.path.join(RESULTS, "rq1_final", "vae_final_summary.json")),
        "contrastive": _locked(os.path.join(RESULTS, "rq1_final",
                                            "contrastive_final_summary.json")),
        "rf": None,
    }

    methods = [("rf", "Random Forest"), ("vae", "VMGP"), ("contrastive", "Contrastive")]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    width = 0.36
    x = np.arange(len(methods))

    ax = axes[0]
    for off, (species, series, color) in zip(
        (-width / 2, width / 2),
        (("goat", goat_cv, SPECIES_COLORS["goat"]),
         ("sheep", {m: (sheep_cv[m]["macro_f1_mean"], sheep_cv[m]["macro_f1_std"])
                    for m, _ in methods}, SPECIES_COLORS["sheep"])),
    ):
        vals = [series.get(m, (np.nan, 0.0))[0] for m, _ in methods]
        errs = [series.get(m, (np.nan, 0.0))[1] for m, _ in methods]
        ax.bar(x + off, vals, width, yerr=errs, capsize=3, color=color,
               label=species, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in methods])
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0.60, 1.02)
    ax.set_title("Development 3-fold CV")
    ax.legend(loc="lower right")

    ax = axes[1]
    goat_locked = {m: (goat_fin.get(m) if goat_fin.get(m) is not None else np.nan)
                   for m, _ in methods}
    sheep_locked = {m: sheep_fin.get(m, {}).get("macro_f1", np.nan)
                    for m, _ in methods}
    for off, (species, series, color) in zip(
        (-width / 2, width / 2),
        (("goat", goat_locked, SPECIES_COLORS["goat"]),
         ("sheep", sheep_locked, SPECIES_COLORS["sheep"])),
    ):
        vals = [series.get(m, np.nan) for m, _ in methods]
        ax.bar(x + off, vals, width, color=color, label=species, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in methods])
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0.60, 1.02)
    ax.set_xlim(-0.55, 2.55)
    ax.set_title("Locked test")
    ax.legend(loc="lower right")
    if np.isnan(goat_locked["rf"]):
        ax.annotate("goat RF:\ndevelopment CV\nonly (0.719)",
                    xy=(-width / 2 - 0.05, 0.70), fontsize=8,
                    color="dimgrey", ha="center", va="bottom")

    fig.suptitle("RQ1 cross-species behaviour: caprine (primary) vs ovine (validation)",
                 y=1.02)
    fig.tight_layout()
    save(fig, "fig_cross_species_rq1")


# ---------------------------------------------------------------------------
# 7. Dataset overview: population structure + cohort sizes
# ---------------------------------------------------------------------------
def _breed_palette(n):
    base = (list(plt.get_cmap("tab20").colors)
            + list(plt.get_cmap("tab20b").colors)
            + list(plt.get_cmap("tab20c").colors))
    return [base[i % len(base)] for i in range(n)]


def _species_dataset(species):
    from genomic.experiment_data import GenomicExperimentData
    from genomic.splitting import load_split

    cfg = {
        "goat": dict(dataset_id="ADAPTmap_genotypeTOP_20160222_full",
                     fam="data/ADAPTmap_genotypeTOP_20160222_full.fam",
                     bed="data/ADAPTmap_genotypeTOP_20160222_full.bed",
                     bim="data/ADAPTmap_genotypeTOP_20160222_full.bim",
                     splits="splits"),
        "sheep": dict(dataset_id="ISGC_sheep_hapmap",
                      fam="data/sheep/sheep_hapmap_raw.fam",
                      bed="data/sheep/sheep_hapmap_raw.bed",
                      bim="data/sheep/sheep_hapmap_raw.bim",
                      splits="data/sheep/splits"),
    }[species]
    split = load_split(cfg["dataset_id"], 42, 42,
                       os.path.join(REPO, cfg["fam"]),
                       out_dir=os.path.join(REPO, cfg["splits"]),
                       cohort_label="indmiss0p10")
    ed = GenomicExperimentData.from_plink(
        split, os.path.join(REPO, cfg["bed"]),
        bim_path=os.path.join(REPO, cfg["bim"]),
    )
    return split, ed


def fig_dataset_cohort():
    import pandas as pd
    from sklearn.decomposition import PCA

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
    for row, species in enumerate(("goat", "sheep")):
        split, ed = _species_dataset(species)
        fd = ed.fold_data(0)
        Z = PCA(n_components=2, svd_solver="randomized", random_state=42).fit_transform(
            fd.X_train - fd.X_train.mean(axis=0)
        )
        breeds = sorted(set(fd.train_breed.tolist()))
        palette = _breed_palette(len(breeds))
        color = {b: palette[i] for i, b in enumerate(breeds)}

        ax = axes[row, 0]
        for b in breeds:
            mask = fd.train_breed == b
            ax.scatter(Z[mask, 0], Z[mask, 1], s=6, alpha=0.7,
                       color=color[b], linewidths=0)
        ax.set_title(f"{species}: PCA of fold-0 training dosages", fontsize=13)
        ax.set_xlabel("PC1", fontsize=11)
        ax.set_ylabel("PC2", fontsize=11)
        ax.tick_params(labelsize=10)

        ax = axes[row, 1]
        counts = pd.Series(split.breed).value_counts()
        step = 20
        bins = np.arange(0, int(counts.max()) + step + 1, step)
        ax.hist(counts.values, bins=bins, edgecolor="white",
                color="#4c72b0" if species == "goat" else "#dd8452")
        ax.axvline(30, color="red", linestyle="--", linewidth=1.2,
                   label="eligibility threshold (30)")
        ax.set_xlabel("individuals per breed", fontsize=11)
        ax.set_ylabel("number of breeds", fontsize=11)
        ax.set_title(f"{species}: cohort size distribution", fontsize=13)
        ax.tick_params(labelsize=10)
        ax.legend(fontsize=9)
    fig.suptitle("Population structure and cohort composition", fontsize=15, y=0.995)
    fig.tight_layout()
    save(fig, "fig_dataset_cohort")


# ---------------------------------------------------------------------------
# 8. Learned latent spaces of the final goat models
# ---------------------------------------------------------------------------
def fig_rq1_latent_spaces():
    import torch
    from sklearn.decomposition import PCA

    from contrastive_learning.evaluation import (
        equal_earth_projection,
        extract_embeddings,
    )
    from contrastive_learning.model import ContrastiveGeneticModel
    from vae.lightning_module import VMGP_LightningSystem

    vae_ckpt = os.path.join(REPO, "checkpoints", "rq1_final", "vae_latent96_ep168.ckpt")
    con_ckpt = os.path.join(REPO, "checkpoints", "rq1_final",
                            "contrastive_emb3_ep1216.ckpt")
    if not (os.path.exists(vae_ckpt) and os.path.exists(con_ckpt)):
        warn("missing final goat checkpoints; skipping fig_rq1_latent_spaces")
        return

    split, ed = _species_dataset("goat")
    dev = ed.build_final_development_data()
    breeds = sorted(set(dev.dev_breed.tolist()))
    palette = _breed_palette(len(breeds))
    color = {b: palette[i] for i, b in enumerate(breeds)}

    vae = VMGP_LightningSystem.load_from_checkpoint(vae_ckpt, map_location="cpu").eval()
    mus = []
    with torch.no_grad():
        for i in range(0, dev.X_dev.shape[0], 256):
            xb = torch.FloatTensor(dev.X_dev[i:i + 256])
            mus.append(vae(xb)["mu"].numpy())
    mu = np.concatenate(mus)
    Zv = PCA(n_components=2, random_state=42).fit_transform(mu - mu.mean(axis=0))

    con = ContrastiveGeneticModel.load_from_checkpoint(con_ckpt, map_location="cpu").eval()
    proj = equal_earth_projection(extract_embeddings(con, dev.X_dev))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.6))
    for b in breeds:
        mask = dev.dev_breed == b
        ax1.scatter(Zv[mask, 0], Zv[mask, 1], s=6, alpha=0.7,
                    color=color[b], linewidths=0)
        ax2.scatter(proj[mask, 0], proj[mask, 1], s=6, alpha=0.7,
                    color=color[b], linewidths=0)
    ax1.set_title(r"VMGP $d=96$ latent space (PCA of $\mu$)")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax2.set_title(r"Contrastive emb$=3$ on $\mathbb{S}^2$ (Equal Earth projection)")
    ax2.set_xlabel("$x'$")
    ax2.set_ylabel("$y'$")

    handles = [Line2D([], [], marker="o", linestyle="", color=color[b],
                      markersize=4, label=b) for b in breeds]
    fig.legend(handles=handles, loc="lower center", ncol=9, fontsize=5,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Goat development embeddings of the two learned representations",
                 y=1.0)
    fig.tight_layout(rect=[0, 0.12, 1, 0.97])
    save(fig, "fig_rq1_latent_spaces")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
FIGURES = [
    fig_dataset_cohort,
    fig_rq1_latentdim_selection,
    fig_rq1_confusion_locked,
    fig_rq1_latent_spaces,
    fig_rq2_learning_curve,
    fig_rq3_panel_curves,
    fig_rq4_jaccard,
    fig_cross_species_rq1,
]


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Generating thesis figures into {OUT_DIR}")
    skipped = []
    for fn in FIGURES:
        try:
            fn()
        except Exception:
            skipped.append(fn.__name__)
            warn(f"{fn.__name__} failed:\n{traceback.format_exc()}")
    if skipped:
        print(f"[warn] figures with problems: {skipped}")
    else:
        print("[ok] all figures generated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
