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
        row_diff = torch.diff(image[:, :-1], dim=0)
        torch.pow(row_diff, 2, out=row_diff)

        column_diff = torch.diff(image[:-1, :], dim=1)
        torch.pow(column_diff, 2, out=column_diff)

        # we'll avoid additional memory allocation by reusing row_diff
        torch.add(row_diff, column_diff, out=row_diff)
        return torch.sqrt(row_diff, out=row_diff).sum()


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
        row_diff = torch.diff(image[:, :-1], dim=0)
        torch.pow(row_diff, 2, out=row_diff)

        column_diff = torch.diff(image[:-1, :], dim=1)
        torch.pow(column_diff, 2, out=column_diff)

        return torch.add(row_diff, column_diff, out=row_diff).sum()
