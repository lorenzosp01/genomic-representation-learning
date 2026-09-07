"""Evaluation utilities for the contrastive model.

Extracted verbatim from the contrastive notebooks.
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, silhouette_score, davies_bouldin_score


def equal_earth_projection(xyz: np.ndarray) -> np.ndarray:
    """Equal Earth projection (Šavrič et al. 2019) — usata nel paper."""
    x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]
    lon = np.arctan2(y, x)
    lat = np.arcsin(np.clip(z, -1, 1))
    A1, A2, A3, A4 = 1.340264, -0.081106, 0.000893, 0.003796
    th  = np.arcsin(np.sqrt(3)/2 * np.sin(lat))
    den = 3*(9*A4*th**8 + 7*A3*th**6 + 3*A2*th**2 + A1)
    xp  = 2*np.sqrt(3)*lon*np.cos(th) / (den + 1e-8)
    yp  = A4*th**9 + A3*th**7 + A2*th**3 + A1*th
    return np.stack([xp, yp], axis=1)


@torch.no_grad()
def extract_embeddings(model, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    dl = DataLoader(TensorDataset(torch.FloatTensor(X)), batch_size=batch_size)
    return np.concatenate([model(xb[0].to(device), augment=False).cpu().numpy() for xb in dl])


def compute_metrics(Z: np.ndarray, y: np.ndarray, k: int = 3) -> dict:
    knn = KNeighborsClassifier(n_neighbors=k).fit(Z, y)
    yp  = knn.predict(Z)
    return {
        f'knn_acc_k{k}': round(accuracy_score(y, yp), 4),
        f'knn_f1_k{k}':  round(f1_score(y, yp, average='macro', zero_division=0), 4),
        'silhouette':     round(silhouette_score(Z, y), 4),
        'davies_bouldin': round(davies_bouldin_score(Z, y), 4),
    }
