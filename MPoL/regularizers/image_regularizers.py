from __future__ import annotations

import torch

from .interface import RegularizerModule


class TV(RegularizerModule):
    def __init__(self, λ: float, /) -> None:
        super().__init__()
        self.λ = λ

    def __repr__(self):
        return f"TV(λ={self.λ})"

    def forward(self, image: torch.Tensor):
        return self.λ * TV.functional(image)

    @staticmethod
    def functional(image: torch.Tensor) -> torch.Tensor:
        row_diff = torch.diff(image[:, :-1], dim=0).pow(2)
        column_diff = torch.diff(image[:-1, :], dim=1).pow(2)
        return torch.add(row_diff, column_diff).sqrt().sum()


class TSV(RegularizerModule):
    def __init__(self, λ: float, /) -> None:
        super().__init__()
        self.λ = λ

    def __repr__(self):
        return f"TSV(λ={self.λ})"

    def forward(self, image: torch.Tensor):
        return self.λ * TSV.functional(image)

    @staticmethod
    def functional(image: torch.Tensor) -> torch.Tensor:
        row_diff = torch.diff(image[:, :-1], dim=0).pow(2)
        column_diff = torch.diff(image[:-1, :], dim=1).pow(2)
        return torch.add(row_diff, column_diff).sum()
