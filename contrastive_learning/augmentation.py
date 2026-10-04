"""Genetic augmentation (Thor & Nettelblad 2025).

Allele flip → mask → 4-channel one-hot encoding.
"""

import torch
import torch.nn as nn


class GeneticAugmentation(nn.Module):
    """Augmentation SNP fedele al paper.
    Paper: p_flip, p_mask ~ U(0.01, 0.99) per ogni campione ad ogni iterazione.
    """

    def __init__(self, flip_max: float = 0.99, mask_max: float = 0.99):
        super().__init__()
        self.flip_max = flip_max
        self.mask_max = mask_max
        # Lower bound = 0.01 come da paper ("U(0.01, 0.99)")
        self.p_min = 0.01

    def forward(self, snps: torch.Tensor, training: bool = True) -> torch.Tensor:
        """snps: (B, M) int {-1,0,1,2}  →  (B, M, 4) one-hot float"""
        if not training:
            return self.one_hot_encode(snps)

        B, M = snps.shape
        aug  = snps.clone()

        # ── 1. Flip allelico ──────────────────────────────────
        p_f = torch.rand(B, 1, device=snps.device) * (self.flip_max - self.p_min) + self.p_min
        flip_mask = torch.rand(B, M, device=snps.device) < p_f

        vals = aug[flip_mask]
        if vals.numel() > 0:
            rnd = torch.rand_like(vals.float())
            flipped = torch.where(vals == 0, torch.ones_like(vals),
                       torch.where(vals == 2, torch.ones_like(vals),
                       torch.where(rnd < 0.5,  torch.zeros_like(vals),
                                               torch.full_like(vals, 2))))
            aug[flip_mask] = flipped

        # ── 2. Masking ────────────────────────────────────────
        p_m = torch.rand(B, 1, device=snps.device) * (self.mask_max - self.p_min) + self.p_min
        aug[torch.rand(B, M, device=snps.device) < p_m] = -1

        # ── 3. One-hot encoding ───────────────────────────────
        return self.one_hot_encode(aug)

    @staticmethod
    def one_hot_encode(snps: torch.Tensor) -> torch.Tensor:
        """0→[1,0,0,0] | 1→[0,1,0,0] | 2→[0,0,1,0] | -1→[0,0,0,1]"""
        B, M = snps.shape
        oh   = torch.zeros(B, M, 4, device=snps.device)
        oh[snps ==  0, 0] = 1
        oh[snps ==  1, 1] = 1
        oh[snps ==  2, 2] = 1
        oh[snps == -1, 3] = 1
        return oh
