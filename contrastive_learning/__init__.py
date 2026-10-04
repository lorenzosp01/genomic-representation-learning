"""Contrastive representation-learning model (Thor & Nettelblad 2025).

Core implementation of the paper-faithful centroid N-pair model, together
with the Integrated Gradients attribution utilities.
"""

from .augmentation import GeneticAugmentation
from .encoder import GeneticEncoder
from .loss import CentroidNPairLoss
from .model import ContrastiveGeneticModel
from .evaluation import (
    compute_metrics,
    equal_earth_projection,
    evaluate_knn_classification,
    extract_embeddings,
)
from .experiment import run_experiment
from .plotting import plot_confusion_matrix, plot_distribution, plot_embedding

__all__ = [
    "GeneticAugmentation",
    "GeneticEncoder",
    "CentroidNPairLoss",
    "ContrastiveGeneticModel",
    "run_experiment",
    "extract_embeddings",
    "compute_metrics",
    "evaluate_knn_classification",
    "equal_earth_projection",
    "plot_distribution",
    "plot_embedding",
    "plot_confusion_matrix",
]
