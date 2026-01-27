"""
VMGP Network Architecture
=========================

Implementazione dell'architettura VMGP (Variational Multi-task Genomic Prediction):
- Encoder: Compressione dati genomici
- Sampler: VAE Reparameterization trick
- Decoder: Ricostruzione genomica (Auxiliary Task)
- Predictors: Multi-task classification heads
"""

import torch
import torch.nn as nn


class VMGP_Network(nn.Module):
    """
    VMGP Network con supporto per classificatori multipli:
    - Breed (razza)
    - Continent (continente)
    - Caseina (tipo di caseina)
    - Attitudine (LATTE/FIBRA/CARNE/ALTRO)
    
    Paper Reference: Zhao et al., 2025 - "VMGP: A unified variational auto-encoder 
    based multi-task model for multi-phenotype, multi-environment, and 
    cross-population genomic selection in plants"
    
    Architecture parameters from paper:
    - d1=2048, d2=512, d3=128, latent_dim=128
    """
    def __init__(self, num_snps, num_classes_breed=0, num_classes_continent=0, num_classes_caseina=0,
                 num_classes_attitudine=0,
                 d1=2048, d2=512, d3=128, latent_dim=128,
                 use_breed=True, use_continent=False, use_caseina=False, use_attitudine=False):
        super().__init__()
        
        # Salva configurazione classificatori
        self.use_breed = use_breed
        self.use_continent = use_continent
        self.use_caseina = use_caseina
        self.use_attitudine = use_attitudine
        
        # --- Encoder ---
        self.encoder = nn.Sequential(
            nn.Linear(num_snps, d1),
            nn.BatchNorm1d(d1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(d1, d2),
            nn.BatchNorm1d(d2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1)
        )
        
        # --- Sampler (VAE) ---
        self.fc_mu = nn.Linear(d2, latent_dim)
        self.fc_logvar = nn.Linear(d2, latent_dim)
        
        # --- Decoder (Genomic Reconstruction) ---
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, d2),
            nn.BatchNorm1d(d2),
            nn.LeakyReLU(0.2),
            nn.Linear(d2, d1),
            nn.BatchNorm1d(d1),
            nn.LeakyReLU(0.2),
            nn.Linear(d1, num_snps)
        )
        
        # --- Predictors (Multiple Classification Heads) ---
        
        # Predictor Breed (Razza)
        if self.use_breed and num_classes_breed > 0:
            self.predictor_breed = nn.Sequential(
                nn.Linear(latent_dim, d3),
                nn.BatchNorm1d(d3),
                nn.LeakyReLU(0.2),
                nn.Linear(d3, num_classes_breed)
            )
        else:
            self.predictor_breed = None
            
        # Predictor Continent
        if self.use_continent and num_classes_continent > 0:
            self.predictor_continent = nn.Sequential(
                nn.Linear(latent_dim, d3),
                nn.BatchNorm1d(d3),
                nn.LeakyReLU(0.2),
                nn.Linear(d3, num_classes_continent)
            )
        else:
            self.predictor_continent = None
            
        # Predictor Caseina
        if self.use_caseina and num_classes_caseina > 0:
            self.predictor_caseina = nn.Sequential(
                nn.Linear(latent_dim, d3),
                nn.BatchNorm1d(d3),
                nn.LeakyReLU(0.2),
                nn.Linear(d3, num_classes_caseina)
            )
        else:
            self.predictor_caseina = None
            
        # Predictor Attitudine (LATTE/FIBRA/CARNE/ALTRO)
        if self.use_attitudine and num_classes_attitudine > 0:
            self.predictor_attitudine = nn.Sequential(
                nn.Linear(latent_dim, d3),
                nn.BatchNorm1d(d3),
                nn.LeakyReLU(0.2),
                nn.Linear(d3, num_classes_attitudine)
            )
        else:
            self.predictor_attitudine = None

    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = mu + sigma * epsilon"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x):
        # 1. Encoding
        encoded = self.encoder(x)
        
        # 2. Sampling Latente
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        z = self.reparameterize(mu, logvar)
        
        # 3. Decoder
        x_recon = self.decoder(z)
        
        # 4. Multi-Task Predictions
        outputs = {
            'x_recon': x_recon,
            'mu': mu,
            'logvar': logvar
        }
        
        # Breed prediction
        if self.predictor_breed is not None:
            outputs['logits_breed'] = self.predictor_breed(z)
        
        # Continent prediction
        if self.predictor_continent is not None:
            outputs['logits_continent'] = self.predictor_continent(z)
        
        # Caseina prediction
        if self.predictor_caseina is not None:
            outputs['logits_caseina'] = self.predictor_caseina(z)
        
        # Attitudine prediction
        if self.predictor_attitudine is not None:
            outputs['logits_attitudine'] = self.predictor_attitudine(z)
        
        return outputs
