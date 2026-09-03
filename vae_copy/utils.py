"""
Utility Functions
=================

Funzioni di utilità per:
- Dataset balancing (SMOTE-like)
- Metriche di valutazione (Local Structure, Generalization, Neighbor Overlap)
"""

import numpy as np
from typing import Tuple
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist


def balance_dataset_smote(X: np.ndarray, y: np.ndarray, 
                          target_samples: int = 50, 
                          k_neighbors: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    SMOTE-like interpolation adapted for genomic SNP data.
    
    For minority classes: Generate synthetic samples by interpolating between neighbors.
    For majority classes: Random downsampling.
    
    SNP values are rounded to 0, 1, or 2 after interpolation.
    
    Args:
        X: Feature matrix (n_samples, n_features) with SNP values 0, 1, 2
        y: Labels array
        target_samples: Target samples per class
        k_neighbors: Number of neighbors for SMOTE interpolation
    
    Returns:
        Balanced X and y arrays
    """
    X_bal_list = []
    y_bal_list = []
    unique_classes = np.unique(y)
    
    for cls in unique_classes:
        mask = y == cls
        X_cls = X[mask]
        y_cls = y[mask]
        n_samples = len(X_cls)
        
        if n_samples >= target_samples:
            # Downsample
            indices = np.random.choice(n_samples, target_samples, replace=False)
            X_bal_list.append(X_cls[indices])
            y_bal_list.append(y_cls[indices])
        else:
            # SMOTE-like oversampling
            n_synthetic = target_samples - n_samples
            
            # Use min of k_neighbors and available samples - 1
            k = min(k_neighbors, n_samples - 1)
            if k < 1:
                # Not enough samples for SMOTE, fall back to random
                indices = np.random.choice(n_samples, target_samples, replace=True)
                X_bal_list.append(X_cls[indices])
                y_bal_list.append(y_cls[indices])
                continue
            
            # Fit nearest neighbors
            nn = NearestNeighbors(n_neighbors=k + 1)
            nn.fit(X_cls)
            
            # Generate synthetic samples
            synthetic_samples = []
            for _ in range(n_synthetic):
                # Pick random sample
                idx = np.random.randint(n_samples)
                sample = X_cls[idx]
                
                # Get neighbors
                _, neighbor_indices = nn.kneighbors([sample])
                neighbor_idx = neighbor_indices[0, np.random.randint(1, k + 1)]
                neighbor = X_cls[neighbor_idx]
                
                # Interpolate
                alpha = np.random.random()
                synthetic = sample + alpha * (neighbor - sample)
                
                # Round to valid SNP values (0, 1, 2)
                synthetic = np.clip(np.round(synthetic), 0, 2)
                synthetic_samples.append(synthetic)
            
            # Combine original + synthetic
            X_combined = np.vstack([X_cls, np.array(synthetic_samples)])
            y_combined = np.full(len(X_combined), cls)
            
            X_bal_list.append(X_combined)
            y_bal_list.append(y_combined)
    
    return np.concatenate(X_bal_list), np.concatenate(y_bal_list)


def balance_dataset_multilabel(X: np.ndarray, y_breed: np.ndarray, 
                                y_continent: np.ndarray, y_caseina: np.ndarray,
                                target_samples: int = 50, 
                                y_attitudine: np.ndarray = None,
                                k_neighbors: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    SMOTE-like balancing basato su breed, mantenendo le altre label allineate.
    
    Args:
        X: Feature matrix
        y_breed: Breed labels
        y_continent: Continent labels
        y_caseina: Caseina labels
        target_samples: Target samples per class
        y_attitudine: Attitudine labels (optional)
        k_neighbors: Number of neighbors for SMOTE
    
    Returns:
        Tuple of balanced (X, y_breed, y_continent, y_caseina, y_attitudine)
    """
    if y_attitudine is None:
        y_attitudine = np.full(len(y_breed), -1)
    
    X_bal_list = []
    y_breed_bal_list = []
    y_continent_bal_list = []
    y_caseina_bal_list = []
    y_attitudine_bal_list = []
    
    unique_breeds = np.unique(y_breed)
    
    for breed in unique_breeds:
        mask = y_breed == breed
        X_cls = X[mask]
        y_breed_cls = y_breed[mask]
        y_continent_cls = y_continent[mask]
        y_caseina_cls = y_caseina[mask]
        y_attitudine_cls = y_attitudine[mask]
        n_samples = len(X_cls)
        
        if n_samples >= target_samples:
            indices = np.random.choice(n_samples, target_samples, replace=False)
            X_bal_list.append(X_cls[indices])
            y_breed_bal_list.append(y_breed_cls[indices])
            y_continent_bal_list.append(y_continent_cls[indices])
            y_caseina_bal_list.append(y_caseina_cls[indices])
            y_attitudine_bal_list.append(y_attitudine_cls[indices])
        else:
            n_synthetic = target_samples - n_samples
            k = min(k_neighbors, n_samples - 1)
            
            if k < 1:
                indices = np.random.choice(n_samples, target_samples, replace=True)
                X_bal_list.append(X_cls[indices])
                y_breed_bal_list.append(y_breed_cls[indices])
                y_continent_bal_list.append(y_continent_cls[indices])
                y_caseina_bal_list.append(y_caseina_cls[indices])
                y_attitudine_bal_list.append(y_attitudine_cls[indices])
                continue
            
            nn = NearestNeighbors(n_neighbors=k + 1)
            nn.fit(X_cls)
            
            synthetic_X = []
            synthetic_continent = []
            synthetic_caseina = []
            synthetic_attitudine = []
            
            for _ in range(n_synthetic):
                idx = np.random.randint(n_samples)
                sample = X_cls[idx]
                
                _, neighbor_indices = nn.kneighbors([sample])
                neighbor_idx = neighbor_indices[0, np.random.randint(1, k + 1)]
                neighbor = X_cls[neighbor_idx]
                
                alpha = np.random.random()
                synthetic = sample + alpha * (neighbor - sample)
                synthetic = np.clip(np.round(synthetic), 0, 2)
                
                synthetic_X.append(synthetic)
                synthetic_continent.append(y_continent_cls[idx])
                synthetic_caseina.append(y_caseina_cls[idx])
                synthetic_attitudine.append(y_attitudine_cls[idx])
            
            X_combined = np.vstack([X_cls, np.array(synthetic_X)])
            y_breed_combined = np.concatenate([y_breed_cls, np.full(n_synthetic, breed)])
            y_continent_combined = np.concatenate([y_continent_cls, np.array(synthetic_continent)])
            y_caseina_combined = np.concatenate([y_caseina_cls, np.array(synthetic_caseina)])
            y_attitudine_combined = np.concatenate([y_attitudine_cls, np.array(synthetic_attitudine)])
            
            X_bal_list.append(X_combined)
            y_breed_bal_list.append(y_breed_combined)
            y_continent_bal_list.append(y_continent_combined)
            y_caseina_bal_list.append(y_caseina_combined)
            y_attitudine_bal_list.append(y_attitudine_combined)
    
    return (np.concatenate(X_bal_list), 
            np.concatenate(y_breed_bal_list),
            np.concatenate(y_continent_bal_list),
            np.concatenate(y_caseina_bal_list),
            np.concatenate(y_attitudine_bal_list))


def cap_dataset_multilabel(X: np.ndarray, y_breed: np.ndarray,
                           y_continent: np.ndarray, y_caseina: np.ndarray,
                           target_samples: int = 50,
                           y_attitudine: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Limita (cap/downsample) ogni classe breed a un massimo di target_samples.
    
    A differenza di balance_dataset_multilabel, questa funzione NON genera
    campioni sintetici (NO SMOTE). Le classi con meno di target_samples
    vengono mantenute così come sono, quelle con più vengono sottocampionate.
    
    Args:
        X: Feature matrix
        y_breed: Breed labels
        y_continent: Continent labels
        y_caseina: Caseina labels
        target_samples: Numero massimo di sample per classe
        y_attitudine: Attitudine labels (optional)
    
    Returns:
        Tuple of capped (X, y_breed, y_continent, y_caseina, y_attitudine)
    """
    if y_attitudine is None:
        y_attitudine = np.full(len(y_breed), -1)
    
    X_cap_list = []
    y_breed_cap_list = []
    y_continent_cap_list = []
    y_caseina_cap_list = []
    y_attitudine_cap_list = []
    
    unique_breeds = np.unique(y_breed)
    
    for breed in unique_breeds:
        mask = y_breed == breed
        X_cls = X[mask]
        y_breed_cls = y_breed[mask]
        y_continent_cls = y_continent[mask]
        y_caseina_cls = y_caseina[mask]
        y_attitudine_cls = y_attitudine[mask]
        n_samples = len(X_cls)
        
        if n_samples > target_samples:
            # Downsample a target_samples (solo campioni reali)
            indices = np.random.choice(n_samples, target_samples, replace=False)
            X_cap_list.append(X_cls[indices])
            y_breed_cap_list.append(y_breed_cls[indices])
            y_continent_cap_list.append(y_continent_cls[indices])
            y_caseina_cap_list.append(y_caseina_cls[indices])
            y_attitudine_cap_list.append(y_attitudine_cls[indices])
        else:
            # Mantieni tutti i campioni (nessuna generazione sintetica)
            X_cap_list.append(X_cls)
            y_breed_cap_list.append(y_breed_cls)
            y_continent_cap_list.append(y_continent_cls)
            y_caseina_cap_list.append(y_caseina_cls)
            y_attitudine_cap_list.append(y_attitudine_cls)
    
    return (np.concatenate(X_cap_list),
            np.concatenate(y_breed_cap_list),
            np.concatenate(y_continent_cap_list),
            np.concatenate(y_caseina_cap_list),
            np.concatenate(y_attitudine_cap_list))


def compute_local_structure(embeddings: np.ndarray, 
                            labels: np.ndarray) -> float:
    """
    Local Structure (L) - 3NN leave-one-out accuracy.
    
    Measures how well the latent space preserves local neighborhood structure.
    Uses leave-one-out to avoid data leakage (predicting on training data).
    
    Args:
        embeddings: Embeddings to evaluate
        labels: Corresponding labels
    
    Returns:
        Accuracy score (0-1)
    """
    from sklearn.model_selection import cross_val_score
    
    knn = KNeighborsClassifier(n_neighbors=3)
    
    # Use 5-fold CV instead of predicting on same data (avoids data leakage)
    # For small datasets, use leave-one-out
    n_samples = len(embeddings)
    if n_samples < 50:
        # LOO for very small datasets
        from sklearn.model_selection import LeaveOneOut
        scores = cross_val_score(knn, embeddings, labels, cv=LeaveOneOut())
    else:
        # 5-fold CV for larger datasets
        n_folds = min(5, len(np.unique(labels)))  # Can't have more folds than classes
        scores = cross_val_score(knn, embeddings, labels, cv=n_folds)
    
    return scores.mean()


def compute_generalization(embeddings_train: np.ndarray, 
                           labels_train: np.ndarray,
                           embeddings_val: np.ndarray, 
                           labels_val: np.ndarray) -> float:
    """
    Generalization (GE) - 3NN accuracy on validation set.
    
    Measures how well the learned representation generalizes to unseen data.
    
    Args:
        embeddings_train: Training embeddings
        labels_train: Training labels
        embeddings_val: Validation embeddings
        labels_val: Validation labels
    
    Returns:
        Accuracy score (0-1)
    """
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(embeddings_train, labels_train)
    preds = knn.predict(embeddings_val)
    accuracy = accuracy_score(labels_val, preds)
    return accuracy


def compute_neighbor_overlap(embeddings: np.ndarray, 
                             genotypes: np.ndarray,
                             k_values: list = [3, 10, 30], 
                             max_samples: int = 1000) -> dict:
    """
    Neighbor Overlap - overlap between k-nearest neighbors in genotype space vs embedding space.
    
    Measures how well the embedding preserves genetic similarity relationships.
    
    Args:
        embeddings: Latent space embeddings
        genotypes: Original genotype data
        k_values: List of k values for KNN
        max_samples: Maximum samples for computation (for efficiency)
    
    Returns:
        Dictionary with NO_k{k} scores
    """
    n_samples = len(embeddings)
    if n_samples > max_samples:
        idx = np.random.choice(n_samples, max_samples, replace=False)
        embeddings = embeddings[idx]
        genotypes = genotypes[idx]
    
    dist_geno = cdist(genotypes, genotypes, metric='cityblock') 
    dist_emb = cdist(embeddings, embeddings, metric='euclidean')
    
    overlaps = {}
    for k in k_values:
        overlap_sum = 0
        for i in range(len(embeddings)):
            nn_geno = np.argsort(dist_geno[i])[1:k+1]
            nn_emb = np.argsort(dist_emb[i])[1:k+1]
            
            overlap = len(np.intersect1d(nn_geno, nn_emb)) / k
            overlap_sum += overlap
        
        overlaps[f'NO_k{k}'] = overlap_sum / len(embeddings)
    
    return overlaps
