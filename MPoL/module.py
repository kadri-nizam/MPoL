from __future__ import annotations

import lightning
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from MPoL.grid import CartesianGrid
from MPoL.model import Image, Visibilities
from MPoL.regularizers import ModelRegularizer
from MPoL.visibilities_matching.interface import ProcessedData


class MPoLModule(lightning.LightningModule):
    def __init__(
        self,
        image: Image,
        visibilities: Visibilities,
        image_regularizer: ModelRegularizer,
        visibility_regularizer: ModelRegularizer,
        *,
        optimizer: type[torch.optim.Optimizer] = torch.optim.Adam,
        **optimizer_kwargs,
    ):
        super().__init__()
        self.image = image
        self.visibility = visibilities
        self.image_regularizer = image_regularizer
        self.visibility_regularizer = visibility_regularizer

        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs

    # TODO: Set decent defaults for these
    @classmethod
    def default(
        cls,
        grid: CartesianGrid,
        *,
        image_regularizer: ModelRegularizer = ModelRegularizer(),
        visibility_regularizer: ModelRegularizer = ModelRegularizer(),
        optimizer: type[torch.optim.Optimizer] = torch.optim.Adam,
        **optimizer_kwargs,
    ):
        image = Image.default(grid)
        return cls(
            image=image,
            visibilities=Visibilities(image()),
            image_regularizer=image_regularizer,
            visibility_regularizer=visibility_regularizer,
            optimizer=optimizer,
            **optimizer_kwargs,
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

    def __repr__(self) -> str:
        super_repr = super().__repr__()
        optimizer_kwargs = ", ".join(
            f"{key}={value}" for key, value in self.optimizer_kwargs.items()
        )
        return f"{super_repr} with {self.optimizer.__name__}({optimizer_kwargs})"

    def configure_optimizers(self) -> OptimizerLRScheduler:
        parameters = (
            list(self.image.parameters())
            + list(self.visibility.parameters())
            + list(self.image_regularizer.parameters())
            + list(self.visibility_regularizer.parameters())
        )
        return self.optimizer(parameters, **self.optimizer_kwargs)
