from __future__ import annotations

from typing import Protocol

import numpy as np
import torch  # type: ignore
from astropy.constants import c as SPEED_OF_LIGHT  # type: ignore
from astropy.constants import k_B as BOLTZMANN  # type: ignore

# ======================
# Constants
# ======================
ARCSEC = np.pi / (180 * 3600)  # radian
DEG = np.pi / 180  # radian

K_B = BOLTZMANN.cgs.value  # erg / K
C_CGS = SPEED_OF_LIGHT.cgs.value  # cm / s
C = SPEED_OF_LIGHT.value  # m / s


# ======================
# Common Protocols
# ======================
class TorchCallable(Protocol):
    def __call__(self, input: torch.Tensor, *args, out=None, **kwargs) -> torch.Tensor:
        ...


class TorchForwardCompatible(Protocol):
    def __call__(self, input: torch.Tensor, *args, out=None, **kwargs) -> torch.Tensor:
        ...

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        ...
