from __future__ import annotations
from typing import Protocol

import torch
import torch.utils.data as torch_ud
from dataclasses import dataclass


@dataclass
class VisibilityData:
    u: torch.Tensor
    v: torch.Tensor
    visibility: torch.Tensor
    weights: torch.Tensor | None = None

    def __post_init__(self):
        if self.weights is None:
            self.weights = torch.ones_like(self.visibility)

    @property
    def num_channels(self):
        return self.visibility.size(0)

    @property
    def real(self):
        return self.visibility.real

    @property
    def imag(self):
        return self.visibility.imag


class DataPipeline(Protocol):
    def prepare_data(self):
        # Implement download/caching logic here
        ...

    def setup(self) -> list[VisibilityData]:
        # Data preprocessing goes here returning a VisibilityData object
        ...


class ALMAPipeline(DataPipeline):
    def prepare_data(self):
        ...

    def setup(self, directory: str = "./path/to/data"):
        # read data from directory...

        # pretend read in data with 3 channels and 100 u-v samples
        a = torch.randn((3, 100)).tolist()

        # make list of VisibilityData object for every channel
        data = [VisibilityData(x, x, x, x) for x in a]

        return ALMADataset(data)  # type: ignore


class ALMADataset(torch_ud.Dataset):
    def __init__(self, data: list[VisibilityData]) -> None:
        super().__init__()
        self.data = data

    def __getitem__(self, index: int) -> VisibilityData:
        return self.data[index]

    def __len__(self):
        return len(self.data)
