from __future__ import annotations

import mpol.constants as constants
from mpol.coordinates import GridCoords
import torch
from torch import nn

from scipy.special import j1 as bessel_j1


class AiryPattern(nn.Module):
    def __init__(
        self,
        coords: GridCoords,
        dish_radius: float,
        channel_frequency: torch.Tensor,
        radius_obscured_dish: float = 0,
    ) -> None:
        super().__init__()
        self.dish_radius = dish_radius
        self.channel_frequency = channel_frequency
        self.radius_obscured_dish = radius_obscured_dish

        self.mask = AiryPattern.compute_mask(
            coords=coords,
            dish_radius=dish_radius,
            channel_frequency=channel_frequency,
            radius_obscured_dish=radius_obscured_dish,
        )

    def forward(self, x: torch.Tensor):
        return self.mask * x

    @staticmethod
    def compute_mask(
        coords: GridCoords,
        dish_radius: float,
        channel_frequency: torch.Tensor,
        radius_obscured_dish: float = 0,
    ):
        """Airy pattern mask construction

        The approach follows the definition on:
        https://en.wikipedia.org/wiki/Airy_disk#Obscured_Airy_pattern
        """
        if not 0 <= radius_obscured_dish < dish_radius:
            raise ValueError(
                "The obscured dish radius must be positive and smaller than the radius of the dish."
            )

        # wavelength
        λ = channel_frequency / constants.c_ms

        # wavenumber
        k = 2 * torch.pi / λ
        ε = radius_obscured_dish / dish_radius

        ratio = k * dish_radius
        ratio_cube = torch.tile(ratio, (1, coords.npix, coords.npix))
        r_cube = (
            torch.from_numpy(
                coords.packed_x_centers_2D**2 + coords.packed_y_centers_2D**2
            ).sqrt()
            * constants.arcsec
        )

        x = torch.pi * r_cube * ratio_cube

        A = (1 - ε**2) ** -2
        B = 2 * bessel_j1(x) / x
        C = 2 * ε * bessel_j1(ε * x) / x

        intensity = A * (B - C) ** 2
        return torch.where(r_cube > 0.0, intensity, 1.0)
