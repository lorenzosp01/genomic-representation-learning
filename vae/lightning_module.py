"""
VMGP Lightning Module
=====================

Gestisce la logica di training con PyTorch Lightning.
Loss Function combinata: L = alpha * L_rec + (1-alpha) * L_pred
"""

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from .network import VMGP_Network


class VMGP_LightningSystem(pl.LightningModule):
    """
    Lightning Module per VMGP con supporto multi-task:
    - Ricostruzione genomica (sempre attiva)
    - Classificazione Breed (opzionale)
    - Classificazione Continent (opzionale)
    - Classificazione Caseina (opzionale)
    - Classificazione Attitudine (opzionale)
    
    Loss totale:
    L = alpha * L_rec + (1-alpha) * (w_breed*L_breed + w_continent*L_continent + w_caseina*L_caseina + w_attitudine*L_attitudine)
    """
    def __init__(self, num_snps, 
                 num_classes_breed=0, num_classes_continent=0, num_classes_caseina=0,
                 num_classes_attitudine=0,
                 alpha=0.5, lr=0.0001,
                 use_breed=True, use_continent=False, use_caseina=False, use_attitudine=False,
                 weight_breed=1.0, weight_continent=1.0, weight_caseina=1.0, weight_attitudine=1.0,
                 latent_dim=96):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = VMGP_Network(
            num_snps=num_snps,
            num_classes_breed=num_classes_breed,
            num_classes_continent=num_classes_continent,
            num_classes_caseina=num_classes_caseina,
            num_classes_attitudine=num_classes_attitudine,
            use_breed=use_breed,
            use_continent=use_continent,
            use_caseina=use_caseina,
            use_attitudine=use_attitudine,
            latent_dim=latent_dim
        )
        
        self.alpha = alpha
        self.use_breed = use_breed
        self.use_continent = use_continent
        self.use_caseina = use_caseina
        self.use_attitudine = use_attitudine
        
        # Pesi per le loss dei classificatori
        self.weight_breed = weight_breed
        self.weight_continent = weight_continent
        self.weight_caseina = weight_caseina
        self.weight_attitudine = weight_attitudine
        
        # Normalizza i pesi
        total_weight = 0
        if use_breed: total_weight += weight_breed
        if use_continent: total_weight += weight_continent
        if use_caseina: total_weight += weight_caseina
        if use_attitudine: total_weight += weight_attitudine
        
        if total_weight > 0:
            self.norm_weight_breed = weight_breed / total_weight if use_breed else 0
            self.norm_weight_continent = weight_continent / total_weight if use_continent else 0
            self.norm_weight_caseina = weight_caseina / total_weight if use_caseina else 0
            self.norm_weight_attitudine = weight_attitudine / total_weight if use_attitudine else 0
        else:
            self.norm_weight_breed = 0
            self.norm_weight_continent = 0
            self.norm_weight_caseina = 0
            self.norm_weight_attitudine = 0

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        # Adam semplice come nella configurazione di riferimento
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def _common_step(self, batch, batch_idx, stage):
        # Unpack batch - supporta formati diversi
        if len(batch) == 2:
            x, labels = batch
            y_breed = labels
            y_continent = None
            y_caseina = None
            y_attitudine = None
        elif len(batch) == 4:
            x, y_breed, y_continent, y_caseina = batch
            y_attitudine = None
        elif len(batch) == 5:
            x, y_breed, y_continent, y_caseina, y_attitudine = batch
        else:
            raise ValueError(f"Unexpected batch format with {len(batch)} elements")
        
        outputs = self(x)
        x_recon = outputs['x_recon']
        mu = outputs['mu']
        logvar = outputs['logvar']
        
        # --- Loss Ricostruzione (MSE + KL) ---
        # Paper VMGP: L_rec = L_dif + L_KL = MSE(X̃, X) + D_KL(N(μ,σ) || N(0,I))
        mse_loss = F.mse_loss(x_recon, x, reduction='mean')
        # KL divergence: -0.5 * Σ(1 + log(σ²) - μ² - σ²)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss /= x.size(0)  # Normalize by batch size
        loss_rec = mse_loss + 0.0001 * kl_loss  # KL weight 0.0001 come nella configurazione di riferimento
        self.log(f'{stage}_kl_loss', kl_loss, prog_bar=False)
        
        # --- Loss Predizione (Multiple Tasks) ---
        loss_pred = torch.tensor(0.0, device=self.device)
        
        # Breed Loss
        if self.use_breed and 'logits_breed' in outputs and y_breed is not None:
            loss_breed = F.cross_entropy(outputs['logits_breed'], y_breed)
            loss_pred = loss_pred + self.norm_weight_breed * loss_breed
            
            preds_breed = torch.argmax(outputs['logits_breed'], dim=1)
            acc_breed = (preds_breed == y_breed).float().mean()
            self.log(f'{stage}_acc_breed', acc_breed, prog_bar=True)
            self.log(f'{stage}_loss_breed', loss_breed, prog_bar=False)
        
        # Continent Loss
        if self.use_continent and 'logits_continent' in outputs and y_continent is not None:
            # Filtra campioni con label valida (>= 0)
            valid_mask = y_continent >= 0
            if valid_mask.sum() > 0:
                loss_continent = F.cross_entropy(
                    outputs['logits_continent'][valid_mask], 
                    y_continent[valid_mask]
                )
                loss_pred = loss_pred + self.norm_weight_continent * loss_continent
                
                preds_continent = torch.argmax(outputs['logits_continent'][valid_mask], dim=1)
                acc_continent = (preds_continent == y_continent[valid_mask]).float().mean()
                self.log(f'{stage}_acc_continent', acc_continent, prog_bar=True)
                self.log(f'{stage}_loss_continent', loss_continent, prog_bar=False)
        
        # Caseina Loss
        if self.use_caseina and 'logits_caseina' in outputs and y_caseina is not None:
            # Filtra campioni con label valida (>= 0)
            valid_mask = y_caseina >= 0
            if valid_mask.sum() > 0:
                loss_caseina = F.cross_entropy(
                    outputs['logits_caseina'][valid_mask], 
                    y_caseina[valid_mask]
                )
                loss_pred = loss_pred + self.norm_weight_caseina * loss_caseina
                
                preds_caseina = torch.argmax(outputs['logits_caseina'][valid_mask], dim=1)
                acc_caseina = (preds_caseina == y_caseina[valid_mask]).float().mean()
                self.log(f'{stage}_acc_caseina', acc_caseina, prog_bar=True)
                self.log(f'{stage}_loss_caseina', loss_caseina, prog_bar=False)
        
        # Attitudine Loss
        if self.use_attitudine and 'logits_attitudine' in outputs and y_attitudine is not None:
            # Filtra campioni con label valida (>= 0)
            valid_mask = y_attitudine >= 0
            if valid_mask.sum() > 0:
                loss_attitudine = F.cross_entropy(
                    outputs['logits_attitudine'][valid_mask], 
                    y_attitudine[valid_mask]
                )
                loss_pred = loss_pred + self.norm_weight_attitudine * loss_attitudine
                
                preds_attitudine = torch.argmax(outputs['logits_attitudine'][valid_mask], dim=1)
                acc_attitudine = (preds_attitudine == y_attitudine[valid_mask]).float().mean()
                self.log(f'{stage}_acc_attitudine', acc_attitudine, prog_bar=True)
                self.log(f'{stage}_loss_attitudine', loss_attitudine, prog_bar=False)
        
        # --- Total Loss ---
        if not (self.use_breed or self.use_continent or self.use_caseina or self.use_attitudine):
            total_loss = loss_rec
        else:
            total_loss = self.alpha * loss_rec + (1 - self.alpha) * loss_pred
        
        # Logging
        self.log(f'{stage}_loss', total_loss, prog_bar=True)
        self.log(f'{stage}_rec_loss', loss_rec, prog_bar=False)
        
        return total_loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'val')
    
    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'test')
