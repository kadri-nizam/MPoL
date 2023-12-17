from __future__ import annotations

from enum import StrEnum, auto
from typing import TYPE_CHECKING, Protocol

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from .interface import RegularizerModule


class ReducingFunction(Protocol):
    def __call__(self, input: list[torch.Tensor]) -> torch.Tensor:
        ...


class Reduction(StrEnum):
    """Merge type for the regularization."""

    CONCAT = auto()
    SUM = auto()
    MEAN = auto()
    PRODUCT = auto()
    CUSTOM = auto()


_REDUCTIION: dict[Reduction, ReducingFunction] = {
    Reduction.CONCAT: lambda input: torch.stack(input, dim=0),
    Reduction.SUM: lambda input: torch.stack(input, dim=0).sum(),
    Reduction.MEAN: lambda input: torch.stack(input, dim=0).mean(),
    Reduction.PRODUCT: lambda input: torch.stack(input, dim=0).prod(),
}


class ModelRegularizer(nn.Module):
    def __init__(
        self,
        *regularizers: RegularizerModule,
        reduction_type: Reduction = Reduction.SUM,
    ) -> None:
        super().__init__()
        self.regularizers = nn.ModuleList(regularizers) or nn.ModuleList(
            (nn.Identity(),)
        )
        self.reduction_type = reduction_type
        self.reducer: ReducingFunction = _REDUCTIION[reduction_type]

    @classmethod
    def with_custom_reduction(
        cls, *regularizers: RegularizerModule, reducer: ReducingFunction
    ) -> ModelRegularizer:
        _REDUCTIION.update({Reduction.CUSTOM: reducer})
        return cls(*regularizers, reduction_type=Reduction.CUSTOM)

    def __repr__(self):
        modules = "\n".join(f"  ↳ {module}" for module in self.regularizers)

        # fmt: off
        return (
            f"ModelRegularizer(reduction={self.reduction_type}\n"
                f"{modules}\n"
            ")"
        )
        # fmt: on

    def __getitem__(self, index: int) -> nn.Module:
        return self.regularizers[index]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.reducer([r(x) for r in self.regularizers])
