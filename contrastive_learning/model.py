"""Contrastive genetic model (Thor & Nettelblad 2025).

Extracted verbatim from the contrastive notebooks.
"""

import torch
import pytorch_lightning as pl

from .augmentation import GeneticAugmentation
from .encoder import GeneticEncoder
from .loss import CentroidNPairLoss


class ContrastiveGeneticModel(pl.LightningModule):
    """Modello contrastive learning — completamente fedele al paper."""

    def __init__(self,
                 n_markers:         int,
                 embedding_dim:     int   = 3,
                 flip_max:          float = 0.99,
                 mask_max:          float = 0.99,
                 learning_rate:     float = 0.001,
                 lr_decay_factor:   float = 0.99,
                 lr_decay_interval: int   = 10):
        super().__init__()
        self.save_hyperparameters()
        self.augment   = GeneticAugmentation(flip_max, mask_max)
        self.encoder   = GeneticEncoder(n_markers, embedding_dim)
        self.criterion = CentroidNPairLoss()

    def forward(self, x: torch.Tensor, augment: bool = False) -> torch.Tensor:
        return self.encoder(self.augment(x, training=augment))

    def _build_negatives(self, z: torch.Tensor) -> torch.Tensor:
        """Tutti gli altri campioni come negativi — paper: 'all of the other samples'."""
        B = z.size(0)
        return torch.stack([z[torch.arange(B) != i] for i in range(B)])  # (B, B-1, D)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        z_a = self(x, augment=True)   # anchor
        z_p = self(x, augment=True)   # positive (augmentazione indipendente)
        neg = self._build_negatives(z_p)
        loss = self.criterion(z_a, z_p, neg)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        z_a = self(x, augment=False)  # anchor pulita per monitoring stabile
        z_p = self(x, augment=True)
        neg = self._build_negatives(z_p)
        loss = self.criterion(z_a, z_p, neg)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        # Paper: Adam β₁=0.9, β₂=0.999, lr=0.001, nessun weight decay
        opt = torch.optim.Adam(self.parameters(),
                               lr=self.hparams.learning_rate,
                               betas=(0.9, 0.999))
        # Paper: LR × 0.99 ogni 10 epoche
        sch = torch.optim.lr_scheduler.StepLR(
            opt, step_size=self.hparams.lr_decay_interval,
            gamma=self.hparams.lr_decay_factor)
        return {'optimizer': opt,
                'lr_scheduler': {'scheduler': sch, 'interval': 'epoch', 'frequency': 1}}
