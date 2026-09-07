"""Plotting helpers for the VMGP grid and single-experiment analyses.

Extracted from ``vae/vae_training.ipynb``. Each function reproduces the
corresponding notebook plotting code verbatim, with notebook-global state
replaced by explicit parameters. Default save paths match the original
notebook output paths.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


def plot_grid_heatmaps(
    df_grid: pd.DataFrame,
    metrics_to_plot: Dict[str, str],
    save_path: str = "grid_experiment_heatmaps.png",
) -> None:
    """Plot one heatmap per metric (pivot Min_Samples × Latent_Dim)."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    n_metrics = len(metrics_to_plot)
    n_rows = (n_metrics + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 6 * n_rows))
    axes = axes.flatten()

    for idx, (metric_col, metric_name) in enumerate(metrics_to_plot.items()):
        ax = axes[idx]

        pivot = df_grid.pivot_table(
            values=metric_col,
            index="Min_Samples",
            columns="Latent_Dim",
            aggfunc="mean",
        )

        cmap = "RdYlGn" if "MSE" not in metric_name else "RdYlGn_r"

        sns.heatmap(pivot, annot=True, fmt=".4f", cmap=cmap, ax=ax,
                    linewidths=0.5, linecolor="gray")
        ax.set_title(metric_name, fontweight="bold", fontsize=12)
        ax.set_xlabel("Latent Dimension")
        ax.set_ylabel("Min Samples per Breed")

    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(
        "📊 Griglia Esperimenti: Min Samples × Latent Dim\n"
        "(Solo dati reali, solo breed classification)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_tradeoff(
    df_grid: pd.DataFrame,
    min_samples_range: Sequence[int],
    latent_dim_range: Sequence[int],
    save_path: str = "tradeoff_analysis.png",
) -> None:
    """Plot the number-of-breeds vs performance trade-off (2×2 grid)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 1. Numero di razze per soglia min_samples
    ax1 = axes[0, 0]
    if "Num Classes Breed" in df_grid.columns:
        breed_counts = df_grid.groupby("Min_Samples")["Num Classes Breed"].first()
        ax1.bar(breed_counts.index, breed_counts.values, color="steelblue", alpha=0.8, width=7)
        ax1.set_xlabel("Min Samples per Breed", fontsize=11)
        ax1.set_ylabel("Numero di Razze", fontsize=11)
        ax1.set_title("Razze incluse per soglia", fontweight="bold")
        ax1.set_xticks(min_samples_range)
        for x, y in zip(breed_counts.index, breed_counts.values):
            ax1.text(x, y + 0.3, str(y), ha="center", fontweight="bold", fontsize=12)
        ax1.grid(True, alpha=0.3, axis="y")

    # 2. Raw Accuracy media per min_samples
    ax2 = axes[0, 1]
    if "Acc_Breed (mean)" in df_grid.columns:
        acc_by_samples = df_grid.groupby("Min_Samples")["Acc_Breed (mean)"].agg(["mean", "std"])
        ax2.errorbar(acc_by_samples.index, acc_by_samples["mean"],
                     yerr=acc_by_samples["std"], fmt="o-", capsize=5,
                     color="coral", linewidth=2, markersize=8)
        ax2.set_xlabel("Min Samples per Breed", fontsize=11)
        ax2.set_ylabel("Accuracy Breed (media ± std)", fontsize=11)
        ax2.set_title("Raw Accuracy media per soglia\n(⚠️ non comparabile direttamente)", fontweight="bold")
        ax2.set_xticks(min_samples_range)
        ax2.grid(True, alpha=0.3)

    # 3. Cohen's Kappa per min_samples (confronto equo)
    ax3 = axes[1, 0]
    if "Kappa_Breed (mean)" in df_grid.columns:
        kappa_by_samples = df_grid.groupby("Min_Samples")["Kappa_Breed (mean)"].agg(["mean", "std"])
        ax3.errorbar(kappa_by_samples.index, kappa_by_samples["mean"],
                     yerr=kappa_by_samples["std"], fmt="D-", capsize=5,
                     color="purple", linewidth=2, markersize=8)
        ax3.set_xlabel("Min Samples per Breed", fontsize=11)
        ax3.set_ylabel("Cohen's Kappa (media ± std)", fontsize=11)
        ax3.set_title("Cohen's Kappa per soglia\n(✅ chance-adjusted, comparabile)", fontweight="bold")
        ax3.set_xticks(min_samples_range)
        ax3.grid(True, alpha=0.3)

    # 4. Kappa media per latent_dim
    ax4 = axes[1, 1]
    if "Kappa_Breed (mean)" in df_grid.columns:
        kappa_by_dim = df_grid.groupby("Latent_Dim")["Kappa_Breed (mean)"].agg(["mean", "std"])
        ax4.errorbar(kappa_by_dim.index, kappa_by_dim["mean"],
                     yerr=kappa_by_dim["std"], fmt="s-", capsize=5,
                     color="seagreen", linewidth=2, markersize=8)
        ax4.set_xlabel("Latent Dimension", fontsize=11)
        ax4.set_ylabel("Cohen's Kappa (media ± std)", fontsize=11)
        ax4.set_title("Cohen's Kappa per latent dim\n(media su tutti i min_samples)", fontweight="bold")
        ax4.set_xticks(latent_dim_range)
        ax4.grid(True, alpha=0.3)

    plt.suptitle("📊 Trade-off Analysis", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_trends(
    df_grid: pd.DataFrame,
    min_samples_range: Sequence[int],
    latent_dim_range: Sequence[int],
    save_path: str = "trend_per_latent_dim.png",
) -> None:
    """Plot metric trend lines vs min_samples, one panel per metric."""
    import matplotlib.pyplot as plt

    metrics_lines = {
        "Kappa_Breed (mean)": "Cohen's Kappa (chance-adjusted)",
        "Acc_Breed (mean)": "Raw Breed Accuracy",
        "Generalization (GE) (mean)": "Generalization (GE)",
        "Silhouette (mean)": "Silhouette Score",
    }
    metrics_lines = {k: v for k, v in metrics_lines.items() if k in df_grid.columns}

    n_metrics = len(metrics_lines)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]

    for idx, (metric_col, metric_name) in enumerate(metrics_lines.items()):
        ax = axes[idx]

        for i, latent_dim in enumerate(latent_dim_range):
            subset = df_grid[df_grid["Latent_Dim"] == latent_dim].sort_values("Min_Samples")
            ax.plot(subset["Min_Samples"], subset[metric_col],
                    "o-", label=f"LD={latent_dim}", color=colors[i],
                    linewidth=2, markersize=8)

        ax.set_xlabel("Min Samples per Breed", fontsize=11)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(metric_name, fontweight="bold")
        ax.set_xticks(min_samples_range)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("📈 Trend per Dimensione Latente", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_cm_grid(
    grid_results: List[dict],
    min_samples_range: Sequence[int],
    latent_dim_range: Sequence[int],
    out_dir: str = "results/confusion_matrices",
) -> None:
    """Plot one normalised confusion-matrix figure per min_samples threshold."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    for min_samples in min_samples_range:
        subset = [r for r in grid_results if r.get("Min_Samples") == min_samples]
        if not subset:
            continue

        n_dims = len(latent_dim_range)
        fig, axes = plt.subplots(1, n_dims, figsize=(7 * n_dims, 6))
        if n_dims == 1:
            axes = [axes]

        for idx, latent_dim in enumerate(latent_dim_range):
            ax = axes[idx]

            res = next((r for r in subset if r.get("Latent_Dim") == latent_dim), None)

            if res is None or res.get("Confusion Matrix Breed") is None:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=20)
                ax.set_title(f"LatDim={latent_dim}", fontweight="bold")
                continue

            cm = res["Confusion Matrix Breed"]
            class_names = res.get("Class Names Breed", [str(i) for i in range(len(cm))])

            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm_norm = np.nan_to_num(cm_norm)

            sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", ax=ax,
                        xticklabels=class_names, yticklabels=class_names,
                        vmin=0, vmax=1, linewidths=0.3, linecolor="gray",
                        cbar_kws={"label": "Recall per classe"})

            acc = res.get("Acc_Breed (mean)", 0)
            kappa = res.get("Kappa_Breed (mean)", 0)
            ax.set_title(f"LatDim={latent_dim}\nAcc={acc:.3f} | κ={kappa:.3f}", fontweight="bold", fontsize=11)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.tick_params(axis="both", labelsize=7)

        n_classes = subset[0].get("Num Classes Breed", "?")
        fig.suptitle(f"📊 Confusion Matrix — Min Samples = {min_samples} ({n_classes} razze)\n"
                     f"(Normalizzate per riga: recall per classe)",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/cm_grid_samples_{min_samples}.png", dpi=150, bbox_inches="tight")
        plt.show()


def plot_single_cm(single_result: dict, exp_name: str) -> None:
    """Plot the normalised confusion matrix for a single experiment."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = single_result.get("Confusion Matrix Breed")
    if cm is not None:
        class_names = single_result.get("Class Names Breed", [str(i) for i in range(len(cm))])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", ax=ax,
                    xticklabels=class_names, yticklabels=class_names,
                    vmin=0, vmax=1, linewidths=0.3, linecolor="gray",
                    cbar_kws={"label": "Recall per classe"})
        acc = single_result.get("Acc_Breed (mean)", 0)
        kappa = single_result.get("Kappa_Breed (mean)", 0)
        ax.set_title(f"{exp_name}\nAcc={acc:.3f} | κ={kappa:.3f}", fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.tick_params(axis="both", labelsize=7)
        plt.tight_layout()
        plt.show()
