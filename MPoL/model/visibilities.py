from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from MPoL.constants import TorchCallable, TorchForwardCompatible


class Visibilities(nn.Module):
    def __init__(
        self,
        starting_image: torch.Tensor,
        fft_strategy: TorchCallable = torch.fft.fftn,
        transforms: TorchForwardCompatible = nn.Identity(),
    ):
        super().__init__()
        self.visibilities = starting_image
        self.fft_strategy = fft_strategy
        self.transforms = transforms

    @property
    def real(self) -> torch.Tensor:
        return self.visibilities.detach().real

    @property
    def imag(self) -> torch.Tensor:
        return self.visibilities.detach().imag

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        self.visibilities = self.transforms(self.fft_strategy(image))
        return self.visibilities
