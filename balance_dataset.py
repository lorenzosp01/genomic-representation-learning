import numpy as np
from typing import Tuple, Literal
from sklearn.neighbors import NearestNeighbors

def balance_dataset_smote(X: np.ndarray, y: np.ndarray, target_samples: int = 50, k_neighbors: int = 5) -> Tuple[np.ndarray, np.ndarray]:
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
