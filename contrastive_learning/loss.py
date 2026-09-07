"""Centroid-based N-pair loss (Thor & Nettelblad 2025, Eq. 3–6).

Extracted verbatim from the contrastive notebooks. The sum over negatives is
inside the log, faithful to the paper.
"""

import numpy as np
import torch
import torch.nn as nn


class CentroidNPairLoss(nn.Module):
    """
    Centroid-based N-pair loss — Thor & Nettelblad 2025, Eq. (3)-(6).
    La somma sui negativi è DENTRO il log (fedele al paper).
    """
    _E2 = float(np.exp(-2))   # ≈ 0.1353

    def forward(self,
                anchor:    torch.Tensor,   # (B, D)
                positive:  torch.Tensor,   # (B, D)
                negatives: torch.Tensor,   # (B, N_neg, D)
                ) -> torch.Tensor:

        B, N_neg, D = negatives.shape
        e2 = torch.tensor(self._E2, device=anchor.device)

        # Espandi anchor/positive → (B, N_neg, D)
        z  = anchor.unsqueeze(1).expand(-1, N_neg, -1)
        zp = positive.unsqueeze(1).expand(-1, N_neg, -1)
        zn = negatives

        # Eq. 3 — Centroide: Ci = (z + 2·z⁻ + z⁺) / 4
        C = (z + 2.0 * zn + zp) / 4.0

        z_c  = z  - C
        zp_c = zp - C
        zn_c = zn - C

        # Eq. 4 — Scaling: μi = max(||z_c||², ||zp_c||², ||zn_c||²)
        mu = torch.max(torch.stack([
            (z_c**2).sum(-1), (zp_c**2).sum(-1), (zn_c**2).sum(-1)
        ]), dim=0).values                         # (B, N_neg)

        # Eq. 5 — Normalizzazione: z̃ = z_c / √μ
        d    = torch.sqrt(mu.unsqueeze(-1) + 1e-8)   # (B, N_neg, 1)
        z_t  = z_c  / d
        zp_t = zp_c / d
        zn_t = zn_c / d

        # Dot products
        sim_neg = (z_t * zn_t).sum(-1)   # (B, N_neg)
        sim_pos = (z_t * zp_t).sum(-1)   # (B, N_neg)

        # Eq. 6 — Term per coppia
        term = torch.exp(sim_neg - sim_pos) - e2   # (B, N_neg)

        # Somma sui negativi PRIMA del log  ←  fedele al paper
        inner = term.sum(dim=1)                     # (B,)

        # Clamp numerico (inner ≥ 0 nella configurazione ideale)
        return torch.log(1.0 + inner.clamp(min=0.0)).mean()
