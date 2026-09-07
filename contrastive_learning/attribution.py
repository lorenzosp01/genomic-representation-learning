"""Feature-attribution (Integrated Gradients) for the contrastive model.

Extracted from the contrastive notebooks. Two distinct attribution paths are
provided and are intentionally kept separate:

* **Probe path** — a supervised MLP probe is trained on the encoder embeddings
  and IG is computed on the target-breed logit.

* **Centroid path** — no probe; IG is computed on the cosine similarity
  between the (L2-normalised) embedding and the breed centroid.

Known methodological issues (full-dataset attribution/ranking, full-data CV,
no locked-test) are preserved exactly and will be handled in a later refactor.
"""

from __future__ import annotations

import glob

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from .model import ContrastiveGeneticModel
from .evaluation import extract_embeddings, compute_metrics
from .augmentation import GeneticAugmentation


class EncoderWithProbeMLP(nn.Module):
    """Probe-path wrapper: one-hot genotype -> encoder -> scaled embedding -> probe logits."""

    def __init__(self, encoder, sc_mean, sc_std, probe):
        super().__init__()
        self.encoder = encoder
        self.probe = probe
        self.register_buffer("sc_mean", torch.FloatTensor(sc_mean))
        self.register_buffer("sc_std", torch.FloatTensor(sc_std))

    def forward(self, x_one_hot):
        z = self.encoder(x_one_hot)
        z_scaled = (z - self.sc_mean) / (self.sc_std + 1e-8)
        return self.probe(z_scaled)


class EncoderToCentroid(nn.Module):
    """Centroid-path wrapper: one-hot genotype -> encoder -> cosine sim to breed centroid.

    ``forward`` returns the dot product between the (L2-normalised) embedding
    and the target breed centroid, which is cosine similarity because both are
    unit vectors. ``target_class`` is expected to be a scalar index (as used by
    the single-sample centroid IG path).
    """

    def __init__(self, encoder, centroids):
        super().__init__()
        self.encoder = encoder
        self.register_buffer("centroids", centroids)

    def forward(self, x_one_hot, target_class):
        z = self.encoder(x_one_hot)  # (B, 3) normalizzato L2
        c = self.centroids[target_class].unsqueeze(0)  # (1, 3)
        return (z * c).sum(dim=1)  # (B,) cosine similarity


def select_best_checkpoint(dm, pattern):
    """Select the checkpoint with the best KNN@3 accuracy over the dataset.

    Mirrors the notebook's selection loop. Returns ``(best_ckpt, best_acc,
    best_fold_idx)`` where ``best_fold_idx`` is derived from the checkpoint
    filename (``fold_N``). This index is required by the probe pipeline, whose
    embeddings are taken from the selected fold's validation partition.
    """
    checkpoints = glob.glob(pattern)
    best_ckpt = None
    best_acc = -1

    for ckpt in checkpoints:
        m = ContrastiveGeneticModel.load_from_checkpoint(ckpt)
        Z = extract_embeddings(m, dm.X_processed)
        acc = compute_metrics(Z, dm.y_processed, k=3)["knn_acc_k3"]
        if acc > best_acc:
            best_acc, best_ckpt = acc, ckpt

    best_fold_idx = int(best_ckpt.split("fold_")[1].split("/")[0]) - 1
    return best_ckpt, best_acc, best_fold_idx


def train_probe_mlp(best_Z, best_y, num_classes, device, *,
                    epochs=300, lr=1e-3, batch=256, hidden=64):
    """Train the probe MLP on 3D embeddings and return it with its scaler.

    Architecture (preserved exactly):
        Linear(3 -> hidden) -> BatchNorm -> ReLU -> Dropout
        Linear(hidden -> hidden) -> BatchNorm -> ReLU -> Dropout
        Linear(hidden -> num_classes)

    Returns ``(probe_mlp, scaler_mlp, best_val_acc)``.
    """
    Z_tr, Z_val_mlp, y_tr, y_val_mlp = train_test_split(
        best_Z, best_y, test_size=0.20, stratify=best_y, random_state=42
    )
    scaler_mlp = StandardScaler().fit(Z_tr)
    Z_tr_s = scaler_mlp.transform(Z_tr).astype(np.float32)
    Z_val_mlp_s = scaler_mlp.transform(Z_val_mlp).astype(np.float32)

    tr_ds = TensorDataset(torch.FloatTensor(Z_tr_s), torch.LongTensor(y_tr))
    val_ds = TensorDataset(torch.FloatTensor(Z_val_mlp_s), torch.LongTensor(y_val_mlp))
    tr_dl = DataLoader(tr_ds, batch_size=batch, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch)

    probe_mlp = nn.Sequential(
        nn.Linear(3, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(hidden, num_classes),
    ).to(device)

    optimizer = optim.Adam(probe_mlp.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        probe_mlp.train()
        for Zb, yb in tr_dl:
            Zb, yb = Zb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(probe_mlp(Zb), yb).backward()
            optimizer.step()
        scheduler.step()

        probe_mlp.eval()
        correct = total = 0
        with torch.no_grad():
            for Zb, yb in val_dl:
                preds = probe_mlp(Zb.to(device)).argmax(dim=1).cpu()
                correct += (preds == yb).sum().item()
                total += len(yb)
        val_acc = correct / total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in probe_mlp.state_dict().items()}

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | val_acc={val_acc:.4f} | best={best_val_acc:.4f}")

    probe_mlp.load_state_dict(best_state)
    probe_mlp.eval()
    return probe_mlp, scaler_mlp, best_val_acc


def compute_ig_batched(model, X_snps, target_classes, device,
                       baseline_val=0.0, n_steps=50, batch_size=32):
    """Probe-path Integrated Gradients (batched).

    ``model`` must return per-sample logits (``EncoderWithProbeMLP``).
    ``target_classes`` is a LongTensor on ``device``. Baseline is a constant
    tensor of ``baseline_val``. Per-SNP attribution is the L1 sum over the four
    genotype channels.
    """
    N, M = X_snps.shape
    ig_all = np.zeros((N, M), dtype=np.float32)
    alphas = torch.linspace(0.0, 1.0, n_steps + 1, device=device)

    model.eval()
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        B = end - start
        x_batch = torch.FloatTensor(X_snps[start:end]).to(device)
        t_batch = target_classes[start:end]

        with torch.no_grad():
            x_oh_b = GeneticAugmentation.one_hot_encode(x_batch).float()
        baseline = torch.full_like(x_oh_b, baseline_val)
        diff = x_oh_b - baseline

        grads_list = []
        for alpha in alphas:
            interp = (baseline + alpha * diff).requires_grad_(True)
            scores = model(interp)
            score = scores[torch.arange(B, device=device), t_batch].sum()
            model.zero_grad()
            score.backward()
            grads_list.append(interp.grad.detach().clone())

        gs = torch.stack(grads_list)
        avg_grads = (gs[0] + gs[-1]) / 2.0 + gs[1:-1].sum(0)
        avg_grads /= n_steps

        ig_map = diff * avg_grads
        ig_all[start:end] = ig_map.abs().sum(dim=-1).cpu().numpy()

        if (end % 256 == 0) or end == N:
            print(f"  {end}/{N} campioni elaborati...")

    return ig_all


def compute_breed_centroids(Z, y, num_classes):
    """Per-breed L2-normalised mean embedding (shape (num_classes, embedding_dim))."""
    embedding_dim = Z.shape[1]
    centroids = np.zeros((num_classes, embedding_dim), dtype=np.float32)
    for cls in range(num_classes):
        mask = y == cls
        if mask.sum() > 0:
            c = Z[mask].mean(axis=0)
            centroids[cls] = c / (np.linalg.norm(c) + 1e-8)
    return centroids


def compute_centroid_ig_single(model, x_one_hot, target_class, device,
                               baseline=None, n_steps=50):
    """Single-sample centroid-path IG.

    ``model`` is an ``EncoderToCentroid`` and ``target_class`` is a scalar.
    Returns ``(ig_per_snp, delta)`` where ``delta`` is the completeness error
    ``|sum(IG) - (F(x) - F(baseline))|``.
    """
    if baseline is None:
        baseline = torch.zeros_like(x_one_hot)

    x_one_hot = x_one_hot.detach().to(device)
    baseline = baseline.detach().to(device)
    diff = x_one_hot - baseline

    alphas = torch.linspace(0.0, 1.0, n_steps + 1, device=device)
    grads_list = []

    model.eval()
    for alpha in alphas:
        interp = (baseline + alpha * diff).requires_grad_(True)
        score = model(interp, target_class).sum()
        model.zero_grad()
        score.backward()
        grads_list.append(interp.grad.detach().clone())

    gs = torch.stack(grads_list)
    avg_grads = (gs[0] + gs[-1]) / 2.0 + gs[1:-1].sum(0)
    avg_grads /= n_steps

    ig_map = diff * avg_grads
    ig_per_snp = ig_map[0].abs().sum(dim=1).cpu().numpy()

    with torch.no_grad():
        f_x = model(x_one_hot, target_class).sum().item()
        f_xb = model(baseline, target_class).sum().item()
    ig_sum = ig_map[0].sum().item()
    delta = abs(ig_sum - (f_x - f_xb))
    status = "OK" if delta < 0.05 else "Aumenta n_steps"
    print(f"   sum(IG)={ig_sum:.5f} | F(x)-F(base)={f_x - f_xb:.5f} | delta={delta:.5f} [{status}]")

    return ig_per_snp, delta


def compute_centroid_ig_batched(model, X_snps, target_classes, device,
                                n_steps=50, batch_size=32):
    """Centroid-path Integrated Gradients (batched).

    ``model`` is an ``EncoderToCentroid``; ``target_classes`` is a LongTensor on
    ``device``. Baseline is zeros. Per-SNP attribution is the L1 sum over the
    four genotype channels.
    """
    N, M = X_snps.shape
    ig_all = np.zeros((N, M), dtype=np.float32)
    alphas = torch.linspace(0.0, 1.0, n_steps + 1, device=device)

    model.eval()
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        B = end - start
        x_batch = torch.FloatTensor(X_snps[start:end]).to(device)
        t_batch = target_classes[start:end]

        with torch.no_grad():
            x_oh_b = GeneticAugmentation.one_hot_encode(x_batch).float()
        baseline = torch.zeros_like(x_oh_b)
        diff = x_oh_b - baseline

        grads_list = []
        for alpha in alphas:
            interp = (baseline + alpha * diff).requires_grad_(True)
            c_batch = model.centroids[t_batch]  # (B, 3)
            z = model.encoder(interp)  # (B, 3)
            score = (z * c_batch).sum(dim=1).sum()
            model.zero_grad()
            score.backward()
            grads_list.append(interp.grad.detach().clone())

        gs = torch.stack(grads_list)
        avg_grads = (gs[0] + gs[-1]) / 2.0 + gs[1:-1].sum(0)
        avg_grads /= n_steps

        ig_map = diff * avg_grads
        ig_all[start:end] = ig_map.abs().sum(dim=-1).cpu().numpy()

        if (end % 256 == 0) or end == N:
            print(f"  {end}/{N} campioni elaborati...")

    return ig_all


def aggregate_attributions(ig_all, y, num_classes):
    """Aggregate per-sample attributions to a global per-SNP vector.

    The aggregation is **per-breed equal-weight**, NOT sample-weighted:

    1. mean attribution within each breed;
    2. mean of those per-breed vectors (equal weight per breed);
    3. ``std`` of the per-breed vectors.

    Returns ``(global_mean, global_std)``.
    """
    ig_per_breed = []
    for cls in range(num_classes):
        mask = y == cls
        if mask.sum() > 0:
            ig_per_breed.append(ig_all[mask].mean(axis=0))

    ig_global = np.mean(ig_per_breed, axis=0)
    ig_global_std = np.std(ig_per_breed, axis=0)
    return ig_global, ig_global_std


def panel_accuracy_curve(ig_scores, X, y, *,
                         percentiles=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
                         cv_folds=3,
                         n_random_seeds=5,
                         classifiers=None):
    """Accuracy vs top-percentile SNP panel selected by ``ig_scores``.

    Preserves the current algorithm exactly: percentile-based top-SNP selection,
    KNN(3) / LogisticRegression / RandomForest classifiers, 3-fold stratified CV
    (shuffle, random_state=42) over the supplied data, and a random-panel
    baseline with ``n_random_seeds`` seeds (0..n-1). Returns
    ``(df_curve, acc_full)``.
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    if classifiers is None:
        classifiers = {
            "KNN": KNeighborsClassifier(n_neighbors=3),
            "LR": LogisticRegression(max_iter=1000, C=1.0,
                                     multi_class="multinomial",
                                     solver="lbfgs", random_state=42),
            "RF": RandomForestClassifier(n_estimators=200,
                                         random_state=42, n_jobs=-1),
        }

    scaler_full = StandardScaler()
    X_full_s = scaler_full.fit_transform(X)
    acc_full = {}
    for name, clf in classifiers.items():
        s = cross_val_score(clf, X_full_s, y, cv=skf,
                            scoring="accuracy", n_jobs=-1).mean()
        acc_full[name] = s

    results = []
    for pct in percentiles:
        threshold = np.percentile(ig_scores, 100 - pct)
        idx = np.where(ig_scores >= threshold)[0]
        n_snp = len(idx)
        X_sub = X[:, idx]
        X_sub_s = StandardScaler().fit_transform(X_sub)

        row = {"percentile": pct, "n_snp": n_snp}

        for name, clf in classifiers.items():
            s = cross_val_score(clf, X_sub_s, y, cv=skf,
                                scoring="accuracy", n_jobs=-1).mean()
            row[f"acc_{name}_ig"] = s

        for name in classifiers:
            rand_accs = []
            for seed in range(n_random_seeds):
                np.random.seed(seed)
                ri = np.random.choice(X.shape[1], n_snp, replace=False)
                Xr = StandardScaler().fit_transform(X[:, ri])
                clf = list(classifiers.values())[list(classifiers.keys()).index(name)]
                s = cross_val_score(clf, Xr, y, cv=skf,
                                    scoring="accuracy", n_jobs=-1).mean()
                rand_accs.append(s)
            row[f"acc_{name}_rand_mean"] = np.mean(rand_accs)
            row[f"acc_{name}_rand_std"] = np.std(rand_accs)

        results.append(row)

    df_curve = pd.DataFrame(results)
    return df_curve, acc_full
