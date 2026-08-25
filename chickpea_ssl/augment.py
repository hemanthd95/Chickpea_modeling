"""Aligned spectral-spatial augmentation.

Every geometric transform is sampled once and applied to the full channel stack.
This fixes the channel misregistration in the original notebook.
"""

from __future__ import annotations

import torch


class AlignedTwoView:
    def __init__(self, noise_std: float = 0.03, gain: float = 0.05):
        self.noise_std = noise_std
        self.gain = gain

    def _view(self, patch: torch.Tensor) -> torch.Tensor:
        x = patch.clone()
        k = int(torch.randint(0, 4, ()).item())
        x = torch.rot90(x, k, dims=(-2, -1))
        if torch.rand(()) < 0.5:
            x = torch.flip(x, dims=(-2,))
        if torch.rand(()) < 0.5:
            x = torch.flip(x, dims=(-1,))
        x = x * (1 + (2 * torch.rand(()) - 1) * self.gain)
        return x + torch.randn_like(x) * self.noise_std

    def __call__(self, patch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._view(patch), self._view(patch)

