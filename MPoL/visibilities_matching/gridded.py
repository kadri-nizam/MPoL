from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from MPoL.data import LooseVisibilities
    from MPoL.grid import CartesianGrid


@dataclass
class GriddedVisibilities:
    index: torch.Tensor
    visibilities: torch.Tensor
    weights: torch.Tensor | None = None
    num_channels: int = field(init=False, default=1)

    @property
    def real(self) -> torch.Tensor:
        return self.visibilities.detach().real

    @property
    def imag(self) -> torch.Tensor:
        return self.visibilities.detach().imag

    def __post_init__(self):
        if self.weights is None:
            self.weights = torch.ones_like(self.visibilities)

        self.num_channels = self.visibilities.size(0)

    # TODO: Implement index matching and nll loss computation
    def compute_loss(self, model_visibilities: torch.Tensor) -> torch.Tensor:
        ...


# TODO: Implement this
class DataAverager:
    def __init__(self, grid: CartesianGrid):
        ...

    def prepare_data(self, data: LooseVisibilities) -> GriddedVisibilities:
        ...
