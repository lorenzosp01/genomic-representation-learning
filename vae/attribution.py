"""Feature-selection / attribution helpers for the VMGP model.

Extracted from ``vae/vae_training.ipynb`` (SHAP pre-selection and Random
Forest importance sections). Behaviour is preserved exactly, including the
current non-train-only ranking (a known methodological issue that is
deliberately left unchanged and will be corrected in a later refactor).
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch.nn as nn


class BreedWrapper(nn.Module):
    """SHAP wrapper exposing ``logits_breed`` from a ``VMGP_LightningSystem``.

    This is the actively-used definition from the original notebook (the one
    that wraps the LightningSystem and calls its ``forward``). The other,
    dead notebook definition (wrapping ``VMGP_Network`` directly) was removed.
    """

    def __init__(self, lightning_system):
        super().__init__()
        self.sys = lightning_system

    def forward(self, x):
        outputs = self.sys(x)  # dict con 'mu', 'logits_breed', 'x_recon', ...
        return outputs["logits_breed"]  # (B, num_classes_breed)


def aggregate_shap_to_global(shap_values_obj, n_features: int) -> np.ndarray:
    """Reduce SHAP values to a single global (n_features,) importance vector.

    Handles both the list-of-arrays and the single-ndarray forms returned by
    different SHAP versions. Output is the mean absolute value over all axes
    except the feature axis.
    """
    if isinstance(shap_values_obj, list):
        arr = np.stack([np.asarray(v) for v in shap_values_obj], axis=0)
    else:
        arr = np.asarray(shap_values_obj)

    feature_axes = [i for i, size in enumerate(arr.shape) if size == n_features]
    if not feature_axes:
        raise ValueError(
            f"Impossibile identificare asse feature in shape {arr.shape} "
            f"(n_features={n_features})"
        )

    arr = np.moveaxis(arr, feature_axes[0], -1)
    shap_global_scores = np.abs(arr).mean(axis=tuple(range(arr.ndim - 1)))
    return np.asarray(shap_global_scores, dtype=np.float64)


def run_shap_accuracy_curve(
    shap_global: np.ndarray,
    X_90: np.ndarray,
    y_90: np.ndarray,
    seed: int = 42,
    K_SNPS: Sequence[int] = (47, 233, 465, 930, 2323, 4646),
    out_path: str = "results/shap_vmgp/accuracy_curve.csv",
) -> pd.DataFrame:
    """Accuracy vs top-k SHAP-selected SNP panel (with random baseline)."""
    import os

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    N_SNPS = X_90.shape[1]

    classifiers = {
        "KNN": KNeighborsClassifier(n_neighbors=3),
        "LR": LogisticRegression(max_iter=1000, solver="lbfgs",
                                 multi_class="multinomial", random_state=seed),
        "RF": RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1),
    }
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

    print("\n📊 Baseline full (90% dataset)...")
    X90_s = StandardScaler().fit_transform(X_90)
    for name, clf in classifiers.items():
        s = cross_val_score(clf, X90_s, y_90, cv=skf, scoring="accuracy", n_jobs=-1).mean()
        print(f"  {name}: {s:.4f}")

    results = []
    for k in K_SNPS:
        top_idx = np.argsort(shap_global)[::-1][:k]
        X_sub = StandardScaler().fit_transform(X_90[:, top_idx])
        row = {"k": k, "pct": f"{k / N_SNPS * 100:.1f}%"}

        rand_acc = {n: [] for n in classifiers}
        for seed_i in range(5):
            ri = np.random.RandomState(seed_i).choice(N_SNPS, k, replace=False)
            Xr = StandardScaler().fit_transform(X_90[:, ri])
            for name, clf in classifiers.items():
                rand_acc[name].append(
                    cross_val_score(clf, Xr, y_90, cv=skf, scoring="accuracy", n_jobs=-1).mean()
                )

        for name, clf in classifiers.items():
            s = cross_val_score(clf, X_sub, y_90, cv=skf, scoring="accuracy", n_jobs=-1).mean()
            row[f"acc_{name}_shap"] = s
            row[f"acc_{name}_rand"] = np.mean(rand_acc[name])
            print(f"Top {k:5d} ({row['pct']:>5s}): {name} SHAP={s:.4f} | rand={np.mean(rand_acc[name]):.4f}")
        results.append(row)

    df_shap = pd.DataFrame(results)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df_shap.to_csv(out_path, index=False)
    print(f"\n💾 Salvato: {out_path}")
    return df_shap


def run_rf_selection(
    X_all: np.ndarray,
    y_all: np.ndarray,
    X_ps: np.ndarray,
    y_ps: np.ndarray,
    X_90: np.ndarray,
    y_90: np.ndarray,
    seed: int = 42,
    K_SNPS: Sequence[int] = (47, 233, 465, 930, 2323, 4646),
    out_dir: str = "results/rf_importance",
) -> pd.DataFrame:
    """Random Forest importance on 10% ps and 80% train, with accuracy curves."""
    import os
    import warnings

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    warnings.filterwarnings(
        "ignore",
        message=".*'multi_class' was deprecated.*",
        category=FutureWarning,
    )

    os.makedirs(out_dir, exist_ok=True)
    N_SNPS = X_all.shape[1]

    train80_idx, test20_idx = train_test_split(
        np.arange(len(X_all)),
        test_size=0.20, stratify=y_all, random_state=seed,
    )
    X_train80, y_train80 = X_all[train80_idx], y_all[train80_idx]
    X_test20, y_test20 = X_all[test20_idx], y_all[test20_idx]

    print(f"Training RF su {len(X_ps)} campioni (10% ps)...")
    rf_10 = RandomForestClassifier(n_estimators=500, max_features="sqrt",
                                   random_state=seed, n_jobs=-1)
    rf_10.fit(X_ps, y_ps)
    rf_importance_10 = rf_10.feature_importances_
    np.save(os.path.join(out_dir, "rf_importance_10pct.npy"), rf_importance_10)
    print(f"✅ RF 10%: SNP con imp>0: {(rf_importance_10 > 0).sum()}")

    print(f"Training RF su {len(X_train80)} campioni (80% train)...")
    rf_80 = RandomForestClassifier(n_estimators=500, max_features="sqrt",
                                   random_state=seed, n_jobs=-1)
    rf_80.fit(X_train80, y_train80)
    rf_importance_80 = rf_80.feature_importances_
    np.save(os.path.join(out_dir, "rf_importance_80pct.npy"), rf_importance_80)
    print(f"✅ RF 80%: SNP con imp>0: {(rf_importance_80 > 0).sum()}")

    classifiers_clean = {
        "KNN": KNeighborsClassifier(n_neighbors=3),
        "LR": LogisticRegression(max_iter=1000, solver="lbfgs", random_state=seed),
        "RF": RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1),
    }
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

    configs = [
        ("RF_10pct", rf_importance_10, X_90, y_90),
        ("RF_80pct", rf_importance_80, X_test20, y_test20),
    ]

    results = []
    for label, importance, X_eval, y_eval in configs:
        print(f"\n📊 Curva accuracy — {label} (eval su {len(X_eval)} campioni):")
        for k in K_SNPS:
            top_idx = np.argsort(importance)[::-1][:k]
            X_sub = StandardScaler().fit_transform(X_eval[:, top_idx])
            row = {"method": label, "k": k, "pct": f"{k / N_SNPS * 100:.1f}%"}

            for name, clf in classifiers_clean.items():
                s = cross_val_score(clf, X_sub, y_eval, cv=skf,
                                    scoring="accuracy", n_jobs=-1).mean()
                row[f"acc_{name}"] = s
                print(f"  Top {k:5d} ({row['pct']:>5s}): {name}={s:.4f}")
            results.append(row)

    df_rf = pd.DataFrame(results)
    df_rf.to_csv(os.path.join(out_dir, "accuracy_curve_rf.csv"), index=False)
    print(f"\n💾 Salvato: {out_dir}/accuracy_curve_rf.csv")
    return df_rf
