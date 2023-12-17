from dataclasses import dataclass, field
from typing import Any, Generator

import torch

from MPoL.constants import ARCSEC


@dataclass
class LooseVisibilities:
    u: torch.Tensor
    v: torch.Tensor
    visibilities: torch.Tensor
    weights: torch.Tensor | None = None
    num_channels: int = field(init=False, default=1)

    @property
    def real(self):
        return self.visibilities.real

    @property
    def imag(self):
        return self.visibilities.imag

    def __post_init__(self):
        if self.weights is None:
            self.weights = torch.ones_like(self.visibilities)

        self.num_channels = self.visibilities.size(0)

    def __getitem__(self, index: int) -> "LooseVisibilities":
        assert self.weights is not None
        return LooseVisibilities(
            u=self.u[index],
            v=self.v[index],
            visibilities=self.visibilities[index],
            weights=self.weights[index],
        )

    def __iter__(self) -> Generator["LooseVisibilities", Any, Any]:
        for index in range(self.num_channels):
            yield self[index]

    def max_frequency(self) -> float:
        return torch.max(self.u.abs().max(), self.v.abs().max()).item()

    def max_pixel_size(self) -> float:
        return 1 / (2 * self.max_frequency() * ARCSEC)
