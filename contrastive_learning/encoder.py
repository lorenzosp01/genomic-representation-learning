"""Genetic encoder mapping SNP profiles onto a unit 3-sphere (Thor & Nettelblad 2025).

Extracted verbatim from the contrastive notebooks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneticEncoder(nn.Module):
    """Encoder SNP → sfera unitaria 3D (paper Figure 4)."""

    def __init__(self, n_markers: int, embedding_dim: int = 3):
        super().__init__()

        # 2 × Conv1D(filters=5, kernel=3, stride=1, padding=1 → same size)
        self.conv1 = nn.Conv1d(4, 5, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(5, 5, kernel_size=3, stride=1, padding=1)

        # 3 × DenseBlock(256, BatchNorm, SiLU)
        flat = n_markers * 5
        self.fc1, self.bn1 = nn.Linear(flat, 256), nn.BatchNorm1d(256)
        self.fc2, self.bn2 = nn.Linear(256,  256), nn.BatchNorm1d(256)
        self.fc3, self.bn3 = nn.Linear(256,  256), nn.BatchNorm1d(256)

        # Output: Dense(3) + L2-norm
        self.fc_out = nn.Linear(256, embedding_dim)

    @staticmethod
    def silu(x):
        """SiLU / Swish: x * σ(x) — paper eq. in sezione Model Architecture"""
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, M, 4) → z: (B, 3) su sfera unitaria"""
        x = x.transpose(1, 2)                      # (B, 4, M)
        x = self.silu(self.conv1(x))
        x = self.silu(self.conv2(x))
        x = x.flatten(1)                            # (B, M*5)
        x = self.silu(self.bn1(self.fc1(x)))
        x = self.silu(self.bn2(self.fc2(x)))
        x = self.silu(self.bn3(self.fc3(x)))
        return F.normalize(self.fc_out(x), p=2, dim=1)  # L2-norm → sfera
