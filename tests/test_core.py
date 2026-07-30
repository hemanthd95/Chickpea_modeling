import torch

from chickpea_ssl.augment import AlignedTwoView
from chickpea_ssl.model import SimCLR, nt_xent


def test_model_shapes_and_loss():
    model = SimCLR(in_channels=16, embedding_dim=64, projection_dim=32)
    batch = torch.randn(8, 16, 15, 15)
    embedding, projection = model(batch)
    assert embedding.shape == (8, 64)
    assert projection.shape == (8, 32)
    assert torch.isfinite(nt_xent(projection, projection.clone()))


def test_aligned_views_preserve_shape():
    patch = torch.randn(16, 15, 15)
    first, second = AlignedTwoView()(patch)
    assert first.shape == patch.shape
    assert second.shape == patch.shape

