from __future__ import annotations

import torch
import torch.nn as nn
import mpol.constants as const

_MERGE_DICT = {
    "cat": lambda x: torch.stack(x),
    "sum": lambda x: torch.stack(x).sum(),
    "prod": lambda x: torch.stack(x).prod(),
    "mean": lambda x: torch.stack(x).mean(),
}


class ModelRegularizer(nn.Module):
    def __init__(self, *regularizers: nn.Module, reduction: str = "sum") -> None:
        super().__init__()
        if reduction.lower() not in _MERGE_DICT:
            raise ValueError(f"Reduction method '{reduction}' not available.")

        self.regularizers = nn.ModuleList(regularizers)
        self.reduction = reduction

    def __repr__(self):
        return f"Regularizer {self.regularizers}"

    def __getitem__(self, index: int):
        return self.regularizers[index]

    def forward(self, x: torch.Tensor):
        # reducer = _MERGE_DICT[self.reduction]
        # loss = reducer([regularizer(x) for regularizer in self.regularizers])
        # return loss
        print("Regularizer forward pass")


class Entropy(nn.Module):
    def __init__(self, λ: float, /, prior_intensity: float = 1) -> None:
        super().__init__()
        self.λ = λ
        self.prior_intensity = prior_intensity

    def __repr__(self):
        return f"Entropy(λ={self.λ}, prior_intensity={self.prior_intensity})"

    def forward(self, cube: torch.Tensor):
        return self.λ * Entropy.functional(cube, self.prior_intensity)

    @staticmethod
    def functional(
        cube: torch.Tensor, prior_intensity: float, estimated_total_flux: float = 1e-4
    ):
        return (
            torch.sum(cube * torch.log(cube / prior_intensity)) / estimated_total_flux
        )


class TSV(nn.Module):
    def __init__(self, λ: float, /) -> None:
        super().__init__()
        self.λ = λ

    def __repr__(self):
        return f"TSV(λ={self.λ})"

    def forward(self, cube: torch.Tensor):
        return self.λ * TSV.functional(cube)

    @staticmethod
    def functional(sky_cube: torch.Tensor):
        # diff the cube in ll and remove the last row
        diff_ll = sky_cube[:, 0:-1, 1:] - sky_cube[:, 0:-1, 0:-1]

        # diff the cube in mm and remove the last column
        diff_mm = sky_cube[:, 1:, 0:-1] - sky_cube[:, 0:-1, 0:-1]

        loss = torch.sum(diff_ll**2 + diff_mm**2)

        return loss


class Sparsity(nn.Module):
    def __init__(self, λ: float, /, mask: torch.Tensor | None = None) -> None:
        super().__init__()
        self.λ = λ
        self.mask = mask

    def __repr__(self):
        return f"Sparsity(λ={self.λ}, custom_mask={self.mask is not None})"

    def forward(self, cube: torch.Tensor):
        return self.λ * Sparsity.functional(cube, self.mask)

    @staticmethod
    def functional(cube: torch.Tensor, mask: torch.Tensor | None = None):
        if mask is None:
            return cube.abs().sum()

        return cube.masked_select(mask).abs().sum()


class PSD(nn.Module):
    # pseudo code
    def __init__(self, λ: float, /, *kwargs) -> None:
        super().__init__()
        self.λ = λ

    def __repr__(self):
        return f"PSD(λ={self.λ}, some_params)"

    def forward(self, cube: torch.Tensor):
        return self.λ * PSD.functional(cube)

    @staticmethod
    def functional(some_params):
        return 1
