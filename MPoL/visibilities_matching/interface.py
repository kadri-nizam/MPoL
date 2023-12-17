from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import torch

    from MPoL.data import LooseVisibilities


class ProcessedData(Protocol):
    def compute_loss(self, model_visibilities: torch.Tensor) -> torch.Tensor:
        ...


class ProcessingStrategy(Protocol):
    def process_data(self, data: LooseVisibilities) -> ProcessedData:
        ...
