from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import lightning
import torch

from MPoL.grid import CartesianGrid
from MPoL.model import Image
from MPoL.model.visibilities import Visibilities
from MPoL.regularizers import ModelRegularizer

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import OptimizerLRScheduler

    from MPoL.visibilities_matching.interface import ProcessedData


class MPoLModule(ABC, lightning.LightningModule):
    def __init__(
        self,
        image: Image,
        visibilities: Visibilities,
        image_regularizer: ModelRegularizer,
        visibility_regularizer: ModelRegularizer,
    ):
        super().__init__()
        self.image = image
        self.visibility = visibilities
        self.image_regularizer = image_regularizer
        self.visibility_regularizer = visibility_regularizer

    # TODO: Set decent defaults for these
    @classmethod
    def default(
        cls,
        grid: CartesianGrid,
        *,
        image_regularizer: ModelRegularizer = ModelRegularizer(),
        visibility_regularizer: ModelRegularizer = ModelRegularizer(),
    ):
        image = Image.default(grid)
        return cls(
            image=image,
            visibilities=Visibilities(image()),
            image_regularizer=image_regularizer,
            visibility_regularizer=visibility_regularizer,
        )

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        model_image = self.image()
        model_visibility = self.visibility(model_image)

        return model_image, model_visibility

    def training_step(self, batch: ProcessedData, batch_idx: int) -> torch.Tensor:
        model_image = self.image()
        model_visibility = self.visibility(model_image)

        # TODO: add other losses
        return torch.sum(
            batch.compute_loss(model_visibility),
            self.image_regularizer(model_image),
            self.visibility_regularizer(model_visibility),
        )

    @abstractmethod
    def configure_optimizers(self) -> OptimizerLRScheduler:
        ...
