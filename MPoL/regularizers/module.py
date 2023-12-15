from __future__ import annotations

from enum import StrEnum, auto
from typing import TYPE_CHECKING, Callable

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from .interface import RegularizerModule

REDUCING_FUNCTION = Callable[[list[torch.Tensor]], torch.Tensor]


class Reduction(StrEnum):
    """Merge type for the regularization."""

    CONCAT = auto()
    SUM = auto()
    MEAN = auto()
    PRODUCT = auto()
    CUSTOM = auto()


_REDUCTIION = {
    Reduction.CONCAT: lambda x: torch.stack(x, dim=0),
    Reduction.SUM: lambda x: torch.stack(x, dim=0).sum(),
    Reduction.MEAN: lambda x: torch.stack(x, dim=0).mean(),
    Reduction.PRODUCT: lambda x: torch.stack(x, dim=0).prod(),
    Reduction.CUSTOM: lambda _: torch.Tensor([0.0]),
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
        self.reducer: REDUCING_FUNCTION = _REDUCTIION[reduction_type]

    @classmethod
    def with_custom_reduction(
        cls, *regularizers: RegularizerModule, reducer: REDUCING_FUNCTION
    ) -> ModelRegularizer:
        instance = cls(*regularizers, reduction_type=Reduction.CUSTOM)
        instance.reducer = reducer
        return instance

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
