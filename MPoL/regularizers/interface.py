from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class RegularizerModule(ABC, nn.Module):
    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        ...

    @staticmethod
    @abstractmethod
    def functional(*args, **kwargs) -> torch.Tensor:
        ...
