from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from MPoL.constants import TorchCallable, TorchForwardCompatible

if TYPE_CHECKING:
    from MPoL.grid import CartesianGrid


class Image(nn.Module):
    def __init__(
        self,
        initial_image: torch.Tensor,
        *,
        pixel_map: TorchCallable = nn.Softplus(),
        transforms: TorchForwardCompatible = nn.Identity(),
    ):
        super().__init__()
        self._base_cube = initial_image
        self.pixel_map = pixel_map
        self.transforms = transforms

    @classmethod
    def default(cls, grid: CartesianGrid) -> Image:
        return cls(initial_image=torch.zeros(grid.num_pixels, grid.num_pixels))

    @property
    def sky_image(self) -> torch.Tensor:
        return self.transforms(self.pixel_map(self._base_cube.detach()))

    def forward(self) -> torch.Tensor:
        return self.transforms(self.pixel_map(self._base_cube))
