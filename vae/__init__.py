"""
VMGP: Variational Auto-Encoder based Multi-task Genomic Prediction
===================================================================

Modulo per la classificazione genomica multi-task usando VAE.
"""

from .network import VMGP_Network
from .lightning_module import VMGP_LightningSystem
from .data_module import VAEFoldData, build_vae_fold
from .utils import (
    balance_dataset_smote,
    balance_dataset_multilabel,
    cap_dataset_multilabel,
    compute_local_structure,
    compute_generalization,
    compute_neighbor_overlap
)
from .experiment import run_experiment, CLASSIFIER_CONFIGS

__all__ = [
    'VMGP_Network',
    'VMGP_LightningSystem',
    'VAEFoldData',
    'build_vae_fold',
    'balance_dataset_smote',
    'balance_dataset_multilabel',
    'cap_dataset_multilabel',
    'compute_local_structure',
    'compute_generalization',
    'compute_neighbor_overlap',
    'run_experiment',
    'CLASSIFIER_CONFIGS',
]
