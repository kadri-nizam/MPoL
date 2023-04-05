from __future__ import annotations

from mpol.coordinates import GridCoords
import torch
from torch import nn
from abc import ABC, abstractmethod  # abstract base class

from .data import VisibilityData


class FourierStrategy(nn.Module, ABC):
    # Interface defining any form of gridding/non-gridding strategy
    # Any subclass **must** implement defined @abstractmethod

    # Any subclass that implements the interface can be easily swapped out as the
    # imaging strategy without any changes to the code

    def __init__(self, coords: GridCoords = None, data: VisibilityData = None):  # type: ignore
        super().__init__()
        self.coords = coords
        self.data = data

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
    def __init__(self, coords: GridCoords, data: VisibilityData):
        super().__init__(coords, data)

        # Subclasses will also have the additional following variables
        # from _grid_data_vis whose specific implementation is defined in the
        # subclass
        (
            self.vis_gridded,
            self.weight_gridded,
            self.mask,
            self.vis_indexed,
            self.weight_indexed,
        ) = self._grid_data_vis()

    @abstractmethod
    def _match_index(self, model_vis):
        ...

    @abstractmethod
    def _grid_data_vis(self) -> tuple[torch.Tensor, ...]:
        # for gridded strategy, subclasses must also define how _grid_data_vis works
        # to populate vis_gridded, weight_gridded, mask, vis_indexed, weight_indexed
        # in addition to _match_index
        ...

    def forward(self, _, model_vis):
        print(
            f"{self.__class__.__name__}: Discarding model_image and forward passing only model_vis..."
        )
        return super().forward(model_vis)


class DataAverager(GriddedStrategy):
    # The result is that the constructor requires only coords, data, and strategy-specific
    # parameters from user.
    # Everything else is abstracted away. Only implementation specific method are in each subclass
    def __init__(self, coords: GridCoords = None, data: VisibilityData = None, **data_averager_params):  # type: ignore
        super().__init__(coords, data)  # type: ignore
        print("DataAverager constructor is constructing all gridded info from data_vis")
        # all other DataAverager params saved here

    def _match_index(self, model_vis):
        print("Match indexing vis uu and vv from DataAveraged strategy...")

    def _grid_data_vis(self) -> tuple[torch.Tensor, ...]:
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
    def __init__(self, coords: GridCoords = None, data: VisibilityData = None, **dirty_imager_params):  # type: ignore
        super().__init__(coords, data)  # type: ignore
        print("DirtyImager constructor is constructing all gridded info from data_vis")
        # all other DirtyImager params saved here

    def _match_index(self, model_vis):
        print("Match indexing vis uu and vv from DirtyImager strategy...")

    def _grid_data_vis(self) -> tuple[torch.Tensor, ...]:
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
    def __init__(self, coords: GridCoords, data: VisibilityData, **nufft_params):
        super().__init__(coords, data)
        print("NuFFT constructor is constructing NuFFT object")
        # make nufft object

    def _match_index(self, model_image):
        print("Match indexing  vis uu and vv from NuFFT strategy...")

    def forward(self, model_image, _):
        print("NuFFT: Discarding model_vis and forward passing only model_image...")
        return super().forward(model_image)
