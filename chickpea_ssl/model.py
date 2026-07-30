from __future__ import annotations

import torch
from torch import nn


class SmallSpectralSpatialEncoder(nn.Module):
    """Compact encoder suited to 15x15 multi-representation patches."""

    def __init__(self, in_channels: int = 16, embedding_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.embedding = nn.Linear(256, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.embedding(self.features(x).flatten(1))
        return nn.functional.normalize(z, dim=1)


class SimCLR(nn.Module):
    def __init__(self, in_channels: int = 16, embedding_dim: int = 128,
                 projection_dim: int = 128):
        super().__init__()
        self.encoder = SmallSpectralSpatialEncoder(in_channels, embedding_dim)
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(x)
        projection = nn.functional.normalize(self.projector(embedding), dim=1)
        return embedding, projection


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    n = z1.shape[0]
    z = torch.cat((z1, z2), dim=0)
    similarity = z @ z.T / temperature
    similarity.fill_diagonal_(float("-inf"))
    targets = torch.arange(n, device=z.device)
    targets = torch.cat((targets + n, targets))
    return nn.functional.cross_entropy(similarity, targets)

