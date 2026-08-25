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


class SupervisedClassifier(nn.Module):
    """Diagnostic three-class head on the shared spectral-spatial encoder."""

    def __init__(self, in_channels: int = 111, embedding_dim: int = 128,
                 classes: int = 3):
        super().__init__()
        self.encoder = SmallSpectralSpatialEncoder(in_channels, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x))



class CenterSpectrumMLP(nn.Module):
    """Classify the authoritative center pixel from its spectrum alone."""

    def __init__(self, in_channels: int = 111, hidden_dim: int = 256,
                 embedding_dim: int = 128, classes: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        center_row, center_column = x.shape[-2] // 2, x.shape[-1] // 2
        return self.network(x[:, :, center_row, center_column])


class CenterContextClassifier(nn.Module):
    """Fuse the center spectrum with a learned spatial-context representation."""

    def __init__(self, in_channels: int = 111, center_dim: int = 128,
                 context_dim: int = 128, fusion_dim: int = 128,
                 classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.center = nn.Sequential(
            nn.Linear(in_channels, center_dim),
            nn.LayerNorm(center_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.context = SmallSpectralSpatialEncoder(in_channels, context_dim)
        self.classifier = nn.Sequential(
            nn.Linear(center_dim + context_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        center_row, center_column = x.shape[-2] // 2, x.shape[-1] // 2
        center = self.center(x[:, :, center_row, center_column])
        context = self.context(x)
        return self.classifier(torch.cat((center, context), dim=1))


def build_supervised_model(
    architecture: str, in_channels: int = 111, classes: int = 3
) -> nn.Module:
    """Construct one declared supervised architecture for controlled ablations."""
    if architecture == "center_spectrum_mlp":
        return CenterSpectrumMLP(in_channels=in_channels, classes=classes)
    if architecture == "spatial_average_cnn":
        return SupervisedClassifier(in_channels=in_channels, classes=classes)
    if architecture == "center_context_fusion":
        return CenterContextClassifier(in_channels=in_channels, classes=classes)
    raise ValueError(f"Unknown supervised architecture: {architecture}")

def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    n = z1.shape[0]
    z = torch.cat((z1, z2), dim=0)
    similarity = z @ z.T / temperature
    similarity.fill_diagonal_(float("-inf"))
    targets = torch.arange(n, device=z.device)
    targets = torch.cat((targets + n, targets))
    return nn.functional.cross_entropy(similarity, targets)
