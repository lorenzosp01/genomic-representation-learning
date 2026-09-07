"""Contrastive representation-learning model (Thor & Nettelblad 2025).

Core implementation extracted verbatim from the contrastive notebooks. The
Integrated Gradients / attribution code remains in the notebooks for now.
"""

from .data import GenomicDataModule
from .augmentation import GeneticAugmentation
from .encoder import GeneticEncoder
from .loss import CentroidNPairLoss
from .model import ContrastiveGeneticModel
from .evaluation import compute_metrics, equal_earth_projection, extract_embeddings
from .experiment import run_experiment
from .plotting import plot_confusion_matrix, plot_distribution, plot_embedding

__all__ = [
    "GenomicDataModule",
    "GeneticAugmentation",
    "GeneticEncoder",
    "CentroidNPairLoss",
    "ContrastiveGeneticModel",
    "run_experiment",
    "extract_embeddings",
    "compute_metrics",
    "equal_earth_projection",
    "plot_distribution",
    "plot_embedding",
    "plot_confusion_matrix",
]
