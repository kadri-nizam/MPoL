from __future__ import annotations

import torch
import torch.nn as nn


# TODO: Implement this
class HannConvCube(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
