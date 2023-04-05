from __future__ import annotations

import torch

from mpol.coordinates import GridCoords
from src.uml.data import ALMAPipeline
from src.uml.pb_correction import AiryPattern
from src.uml.images import ModelImage
from src.uml.regularizer import PSD, Entropy, ModelRegularizer, Sparsity, TSV
from src.uml.vis_matching import DataAverager, DirtyImager, NuFFT
from src.uml.model import RMLImaging
from mpol.images import HannConvCube


def main():
    # setup data and get the first channel
    data = ALMAPipeline().setup()[0]

    # define our grid coordinates
    coords = GridCoords(cell_size=0.004, npix=100)

    # define the pb-correction if applicable
    primary_beam_correction = AiryPattern(
        coords,
        dish_radius=12,
        channel_frequency=torch.tensor([0]),  # some channel frequency
        radius_obscured_dish=3,
    )

    # define the image model with custom base cube
    model_image = ModelImage(
        coords,
        transforms=HannConvCube(nchan=1),
        primary_beam_correction=primary_beam_correction,
    )

    # choose the strategy we want to use match the visibility
    # with polymorphism, we can simply choose a different strategy and the library
    # will call the correct method
    # strategy = DataAverager(coords, data)
    # strategy = NuFFT(coords, data, sparse_matrices=True)
    strategy = DirtyImager(coords, data, weighting="briggs", robust=0.5)

    # include any regularizers
    image_regularizer = ModelRegularizer(Entropy(1e-4, 1e-4), Sparsity(1e-4), TSV(1e-4))
    fourier_regularizer = ModelRegularizer(PSD(1e-4))

    # we are now ready for training
    model = RMLImaging(model_image, strategy, image_regularizer, fourier_regularizer)

    # training epochs for loop, optimizer and all and all goes here
    # forward call on model to get loss
    loss = model()
    # loss.backward()


if __name__ == "__main__":
    main()
