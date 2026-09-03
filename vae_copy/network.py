"""
VMGP Network Architecture — modificato per breed classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterLoss(nn.Module):
    """Minimizza distanza intra-classe nel latent space (mu)."""
    def __init__(self, num_classes, latent_dim):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, latent_dim))

    def forward(self, mu, labels):
        centers_batch = self.centers[labels]
        return ((mu - centers_batch) ** 2).sum(dim=1).mean()


class FocalLoss(nn.Module):
    """Down-pesa esempi facili, focus sulle razze difficili (gamma=2)."""
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, labels):
        ce = F.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class VMGP_Network(nn.Module):
    """
    VMGP Network — paper architecture invariata.
    Modifiche per breed classification:
    - Dropout(0.3) nel predictor_breed
    - CenterLoss e FocalLoss come attributi
    """
    def __init__(self, num_snps,
                 num_classes_breed=0, num_classes_continent=0,
                 num_classes_caseina=0, num_classes_attitudine=0,
                 d1=2048, d2=512, d3=128, latent_dim=96,
                 use_breed=True, use_continent=False,
                 use_caseina=False, use_attitudine=False):
        super().__init__()

        self.use_breed      = use_breed
        self.use_continent  = use_continent
        self.use_caseina    = use_caseina
        self.use_attitudine = use_attitudine

        # --- Encoder (invariato) ---
        self.encoder = nn.Sequential(
            nn.Linear(num_snps, d1),
            nn.BatchNorm1d(d1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(d1, d2),
            nn.BatchNorm1d(d2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
        )

        self.fc_mu     = nn.Linear(d2, latent_dim)
        self.fc_logvar = nn.Linear(d2, latent_dim)

        # --- Decoder (invariato) ---
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, d2),
            nn.BatchNorm1d(d2),
            nn.LeakyReLU(0.2),
            nn.Linear(d2, d1),
            nn.BatchNorm1d(d1),
            nn.LeakyReLU(0.2),
            nn.Linear(d1, num_snps),
        )

        # --- Predictor Breed — aggiunto Dropout(0.3) ---
        if self.use_breed and num_classes_breed > 0:
            self.predictor_breed = nn.Sequential(
                nn.Linear(latent_dim, d3),
                nn.BatchNorm1d(d3),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),                      # ← NUOVO
                nn.Linear(d3, num_classes_breed),
            )
            self.center_loss_fn = CenterLoss(num_classes_breed, latent_dim)  # ← NUOVO
        else:
            self.predictor_breed  = None
            self.center_loss_fn   = None

        self.focal_loss_fn = FocalLoss(gamma=2.0)     # ← NUOVO

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    def forward(self, x):
        encoded = self.encoder(x)
        mu      = self.fc_mu(encoded)
        logvar  = self.fc_logvar(encoded)
        z       = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)

        outputs = {'x_recon': x_recon, 'mu': mu, 'logvar': logvar}

        if self.predictor_breed is not None:
            outputs['logits_breed'] = self.predictor_breed(mu)  # mu, non z

        return outputs
