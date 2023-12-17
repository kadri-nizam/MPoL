from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import torch

from MPoL.constants import ARCSEC
from MPoL.exceptions import GridIncompatibleError

if TYPE_CHECKING:
    from MPoL.data import LooseVisibilities


@dataclass(frozen=True)
class CartesianGrid:
    num_pixels: int
    pixel_size: float

    def __post_init__(self):
        if self.num_pixels % 2 != 0:
            raise ValueError("num_pixels must be an even integer")

        if self.pixel_size <= 0:
            raise ValueError("pixel_size must be a positive real number")

    @cached_property
    def image_extent(self) -> list[float]:
        max_extent = self.pixel_size * (self.num_pixels // 2 + 0.5)  # arcsecs
        min_extent = -self.pixel_size * (self.num_pixels // 2 - 0.5)
        return [min_extent, max_extent, min_extent, max_extent]

    @cached_property
    def d_image(self) -> float:
        return self.pixel_size * ARCSEC

    @cached_property
    def image_centers(self) -> torch.Tensor:
        N = self.num_pixels // 2
        return torch.linspace(-N + 1, N, self.num_pixels) * self.d_image

    @cached_property
    def d_spatial_freq(self) -> float:
        # units of λ
        return 1 / (self.num_pixels * self.d_image)

    @cached_property
    def spatial_frequency_edges(self) -> torch.Tensor:
        N = self.num_pixels // 2
        return (torch.linspace(-N, N, self.num_pixels + 1) + 0.5) * self.d_spatial_freq

    @cached_property
    def spatial_frequency_centers(self) -> torch.Tensor:
        N = self.num_pixels // 2
        return (torch.linspace(-N + 1, N, self.num_pixels)) * self.d_spatial_freq

    @cached_property
    def uv_extent(self) -> list[float]:
        uv_edges = self.spatial_frequency_edges
        min_extent = uv_edges.min().item()
        max_extent = uv_edges.max().item()

        return [min_extent, max_extent, min_extent, max_extent]

    @cached_property
    def max_spatial_frequency(self) -> float:
        return 1 / (2 * self.d_image)

    @cached_property
    def nyquist_frequency(self) -> float:
        return self.num_pixels / 2 * self.d_spatial_freq

    @staticmethod
    def assert_compatible_with(grid: "CartesianGrid", data: LooseVisibilities) -> None:
        if data.max_frequency() > grid.max_spatial_frequency:
            raise GridIncompatibleError(
                "Dataset contains uv spatial frequency measurements larger than those in the proposed model image. "
                f"Choose pixel_size <= {data.max_pixel_size()} arcsec."
            )
