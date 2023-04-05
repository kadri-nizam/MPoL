from .images import ModelImage
from .regularizer import ModelRegularizer
from .vis_matching import FourierStrategy

from torch import nn


class RMLImaging(nn.Module):
    def __init__(
        self,
        model: ModelImage,
        strategy: FourierStrategy,
        image_regularizer: ModelRegularizer,
        vis_regularizer: ModelRegularizer,
    ):
        super().__init__()
        self.model = model
        self.strategy = strategy
        self.image_regularizer = image_regularizer
        self.vis_regularizer = vis_regularizer

    def forward(self):
        print(20 * "-")
        print("RMLImaging forward pass")
        print(20 * "-")

        print("Calling model forward pass...")
        model_image, model_vis = self.model()

        print("Calling indexing strategy; passing in model_image and model_vis")
        # since some algorithms need the pre-fft image while others need post-fft,
        # providing both seems to be the most flexible approach without introducing
        # unnecessary coupling between classes
        # the forward pass of each strategy will decide which to use (implemented by dev)
        self.strategy(model_image, model_vis)
        print("Strategy returned indexed vis")
        # return from strategy will be used for nll computation
        # nll_loss = ...
        print("Compute nll loss")

        print("Compute image regularization loss")
        image_regularization_loss = self.image_regularizer(self.model.sky_cube)

        print("Compute vis regularization loss")
        vis_regularization_loss = self.vis_regularizer(model_vis)

        print("Return nll_loss + image_regularization_loss + vis_regularization_loss")
        # return nll_loss + image_regularization_loss + vis_regularization_loss
