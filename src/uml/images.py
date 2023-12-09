from __future__ import annotations
from typing import Callable

import torch
from torch import nn
from mpol.coordinates import GridCoords
from mpol.fourier import FourierCube

from mpol.utils import packed_cube_to_sky_cube


class ImageCube(nn.Module):
    def __init__(
        self,
        starting_cube: torch.Tensor,
        pixel_map: Callable[[torch.Tensor], torch.Tensor] = nn.Softplus(),
    ) -> None:
        super().__init__()

        self.nchan = starting_cube.size(0)
        self._base = starting_cube
        self.pixel_map = pixel_map

    @classmethod
    def default_cube(
        cls,
        coords: GridCoords,
        nchan: int = 1,
        pixel_map: Callable[[torch.Tensor], torch.Tensor] = nn.Softplus(),
    ):
        starting_cube = torch.full((nchan, coords.npix, coords.npix), 0.5)
        return cls(starting_cube, pixel_map)

    def forward(self):
        return self.pixel_map(self._base)

    @property
    def sky_cube(self):
        return packed_cube_to_sky_cube(self().detach())

    @property
    def shape(self):
        return self._base.shape


class ModelImage(nn.Module):
    def __init__(
        self,
        coords: GridCoords,
        *,
        image_cube: ImageCube | None = None,
        transforms: nn.Module | nn.Sequential = nn.Identity(),
        primary_beam_correction: nn.Module | nn.Sequential = nn.Identity(),
        nchan: int = 1,
    ) -> None:
        super().__init__()

        if image_cube is None:
            image_cube = ImageCube.default_cube(coords, nchan)

        self.coords = coords
        self.image_cube = image_cube
        self.transforms = transforms
        self.primary_beam_correction = primary_beam_correction
        self.fourier_cube = FourierCube()

    def forward(self):
        # model_image = self.primary_beam_correction(self.transforms(self.image_cube()))
        # model_vis = self.fourier_cube(model_image)
        print("ModelImage forward pass. Return model_image and model_vis")
        model_image = None
        model_vis = None
        return model_image, model_vis

    @property
    def nchan(self):
        return self.image_cube.nchan

    @property
    def sky_cube(self):
        return self.image_cube.sky_cube

    @property
    def visibilities(self):
        return self.fourier_cube.vis
