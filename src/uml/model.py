from __future__ import annotations

from mpol.coordinates import GridCoords
from mpol.fourier import NuFFT
from mpol.images import HannConvCube
from uml.pb_correction import AiryPattern
from .images import ImageCube, ModelImage
from .regularizer import PSD, Entropy, ModelRegularizer, Sparsity, TSV
from .vis_matching import DirtyImager, FourierStrategy

import torch
from torch import nn


class ModelImageBuilder:
    def __init__(self, coords: GridCoords, nchan: int = 1):
        self._coords = coords
        self._nchan = nchan
        self._starting_cube = torch.full((nchan, coords.npix, coords.npix), 0.5)
        self._pixel_map = nn.Softplus()
        self._image_transforms = nn.Identity()
        self._pb_correction = nn.Identity()

    def from_starting_cube(self, x: torch.Tensor):
        self._starting_cube = x
        return self

    def pixel_map_fn_is(self, x):
        self._pixel_map = x
        return self

    def transformed_by(self, transforms: nn.Module | nn.Sequential):
        self._image_transforms = transforms
        return self

    def use_hann_transform(self, requires_grad: bool = False):
        self._image_transforms = HannConvCube(self._nchan, requires_grad)
        return self

    def use_airy_pb_correction(
        self,
        dish_radius: float,
        obscured_radius: float,
        channel_frequency: torch.Tensor,
    ):
        self._pb_correction = AiryPattern(
            self._coords, dish_radius, channel_frequency, obscured_radius
        )
        return self

    def build(self):
        image_cube = ImageCube(self._starting_cube, self._pixel_map)
        return ModelImage(
            self._coords,
            image_cube=image_cube,
            transforms=self._image_transforms,
            primary_beam_correction=self._pb_correction,
            nchan=self._nchan,
        )


class StrategyBuilder:
    def __init__(
        self,
    ):
        self.strategy: FourierStrategy | None = None

    def use_nufft(self):
        # self.strategy = NuFFT()
        ...

    def build(self):
        ...


class RegularizerBuilder:
    def __init__(self):
        self._image_regularizer: ModelRegularizer = ModelRegularizer()
        self._visibility_regularizer: ModelRegularizer = ModelRegularizer()

    def regularize_image_with(self, *regularizers: Sparsity | Entropy | TSV):
        self._image_regularizer = ModelRegularizer(*regularizers)
        return self

    def regularize_visibilities_with(self, *regularizers: PSD):
        self._visibility_regularizer = ModelRegularizer(*regularizers)
        return self

    def build(self):
        return {
            "image_regularizer": self._image_regularizer,
            "vis_regularizer": self._visibility_regularizer,
        }


class ModelBuilder:
    def __init__(self, coords: GridCoords, nchan: int = 1):
        self._coords = coords
        self._nchan = nchan
        self.image_model = ModelImageBuilder(coords, nchan)
        self.strategy = StrategyBuilder()
        self.regularizers = RegularizerBuilder()

    @classmethod
    def get_basic_model(cls, coords: GridCoords, nchan: int = 1) -> RMLImaging:
        builder = cls(coords=coords, nchan=nchan)
        builder.image_model.transformed_by(HannConvCube(nchan))
        builder.regularizers.regularize_image_with(TSV(1e-4), Entropy(1e-2))

        return builder.get_model()

    @property
    def coords(self):
        return self._coords

    @property
    def nchan(self):
        return self._nchan

    def reset(self):
        self.image_model = ModelImageBuilder(self.coords, self.nchan)
        self.regularizers = RegularizerBuilder()

    def get_model(self):
        return RMLImaging(
            model=self.image_model.build(),
            strategy=DirtyImager,
            **self.regularizers.build(),
        )


class RMLImaging(nn.Module):
    def __init__(
        self,
        model: ModelImage,
        strategy: FourierStrategy,
        image_regularizer: ModelRegularizer,
        vis_regularizer: ModelRegularizer,
    ):
        super().__init__()
        self.model = model
        self.strategy = strategy
        self.image_regularizer = image_regularizer
        self.vis_regularizer = vis_regularizer

    def forward(self):
        print(20 * "-")
        print("RMLImaging forward pass")
        print(20 * "-")

        print("Calling model forward pass...")
        model_image, model_vis = self.model()

        print("Calling indexing strategy; passing in model_image and model_vis")
        # since some algorithms need the pre-fft image while others need post-fft,
        # providing both seems to be the most flexible approach without introducing
        # unnecessary coupling between classes
        # the forward pass of each strategy will decide which to use (implemented by dev)
        self.strategy(model_image, model_vis)
        print("Strategy returned indexed vis")
        # return from strategy will be used for nll computation
        # nll_loss = ...
        print("Compute nll loss")

        print("Compute image regularization loss")
        image_regularization_loss = self.image_regularizer(self.model.sky_cube)

        print("Compute vis regularization loss")
        vis_regularization_loss = self.vis_regularizer(model_vis)

        print("Return nll_loss + image_regularization_loss + vis_regularization_loss")
        # return nll_loss + image_regularization_loss + vis_regularization_loss
