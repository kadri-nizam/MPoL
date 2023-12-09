from __future__ import annotations
from typing import Protocol

from mpol.coordinates import GridCoords
import torch
from torch import nn
from abc import ABC, abstractmethod

from uml.data import VisibilityData  # abstract base class


class ModelImage(Protocol):
    @property
    def image_cube(self):
        ...

    @property
    def visibilities(self):
        ...


class FourierStrategy(nn.Module, ABC):
    # Interface defining any form of gridding/non-gridding strategy
    # Any subclass **must** implement defined @abstractmethod

    # Any subclass that implements the interface can be easily swapped out as the
    # imaging strategy without any changes to the code

    def __init__(self, coords: GridCoords = None):  # type: ignore
        super().__init__()
        self.coords = coords

    @abstractmethod
    def _process_data_vis(self, data):
        ...

    @abstractmethod
    def _match_index(self, cube):
        ...

    # all subclass will have forward already defined from subclassing nn.Module
    # subclasses need only define how _match_index works to get forward working
    def forward(self, cube):
        print("Calling _match_index to index at proper uu and vv")
        return self._match_index(cube)


class GriddedStrategy(FourierStrategy):
    # Additional interface for gridded strategies as this requires a fourier cube
    def __init__(self, coords: GridCoords):
        super().__init__(coords)

        self.vis_gridded: torch.Tensor | None = None
        self.weight_gridded: torch.Tensor | None = None
        self.mask: torch.Tensor | None = None
        self.vis_indexed: torch.Tensor | None = None
        self.weight_indexed: torch.Tensor | None = None

    @abstractmethod
    def fit_observed_visibilities(self, data) -> tuple[torch.Tensor, ...]:
        # for gridded strategy, subclasses must also define how _grid_data_vis works
        # to populate vis_gridded, weight_gridded, mask, vis_indexed, weight_indexed
        # in addition to _match_index
        ...

    @abstractmethod
    def _match_index(self, visibilities):
        ...

    def forward(self, model: ModelImage):
        # Subclasses will also have the additional following variables
        # from _grid_data_vis whose specific implementation is defined in the
        # subclass
        print(
            f"{self.__class__.__name__}: Discarding model_image and forward passing only model_vis..."
        )
        return super().forward(model.visibilities)


class DataAverager(GriddedStrategy):
    # The result is that the constructor requires only coords, data, and strategy-specific
    # parameters from user.
    # Everything else is abstracted away. Only implementation specific method are in each subclass
    def __init__(self, coords: GridCoords = None, **data_averager_params):  # type: ignore
        super().__init__(coords)  # type: ignore
        print("DataAverager constructor is constructing all gridded info from data_vis")
        # all other DataAverager params saved here

    def _match_index(self, visibilities):
        print("Match indexing vis uu and vv from DataAveraged strategy...")

    def _process_data_vis(self) -> tuple[torch.Tensor, ...]:
        print("Making gridded data vis...")
        # Just returning None here to make it work
        vis_gridded = None
        weight_gridded = None
        mask = None
        vis_indexed = None
        weight_indexed = None
        return vis_gridded, weight_gridded, mask, vis_indexed, weight_indexed  # type: ignore


class DirtyImager(GriddedStrategy):
    # The result is that the constructor requires only coords, data, and strategy-specific
    # parameters from user.
    # Everything else is abstracted away. Only implementation specific method are in each subclass
    def __init__(self, coords: GridCoords = None, **dirty_imager_params):  # type: ignore
        super().__init__(coords)  # type: ignore
        print("DirtyImager constructor is constructing all gridded info from data_vis")
        # all other DirtyImager params saved here

    def _match_index(self, visibilities):
        print("Match indexing vis uu and vv from DirtyImager strategy...")

    def _process_data_vis(self) -> tuple[torch.Tensor, ...]:
        print("Making gridded data vis...")
        # Just returning None here to make it work
        vis_gridded = None
        weight_gridded = None
        mask = None
        vis_indexed = None
        weight_indexed = None
        return vis_gridded, weight_gridded, mask, vis_indexed, weight_indexed  # type: ignore


class NuFFT(FourierStrategy):
    # The result is that the constructor requires only coords, data, and strategy-specific
    # parameters from user.
    # Everything else is abstracted away. Only implementation specific method are in each subclass
    def __init__(self, coords: GridCoords, **nufft_params):
        super().__init__(coords)
        print("NuFFT constructor is constructing NuFFT object")
        # make nufft object

    def _match_index(self, image_cube):
        print("Match indexing  vis uu and vv from NuFFT strategy...")

    def forward(self, model: ModelImage):
        print("NuFFT: Discarding model_vis and forward passing only model_image...")
        return super().forward(model.image_cube)
