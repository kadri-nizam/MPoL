from __future__ import annotations

import torch
from torch import nn

from mpol.coordinates import GridCoords
from src.uml.data import ALMAPipeline
from src.uml.regularizer import PSD, Entropy, Sparsity, TSV
from src.uml.model import ModelBuilder


def main():
    # setup data and get the first channel
    data = ALMAPipeline().setup()[0]

    # define our grid coordinates
    coords = GridCoords(cell_size=0.004, npix=100)

    builder = ModelBuilder(coords)

    # fmt: off
    # Setup the image model properties
    (
        builder.image_model
        .from_starting_cube(torch.randn((1, 800, 800)))
        .pixel_map_fn_is(torch.exp)
        .use_hann_transform()
        .use_airy_pb_correction(
            dish_radius=12,
            obscured_radius=3,
            channel_frequency=torch.tensor([0, 1, 2]),
        )
        .pixel_map_fn_is(torch.exp)
    )

    # Setup visibility matching strategy

    # Setup the regularizers
    (
        builder.regularizers
        .regularize_image_with(TSV(1), Entropy(1), Sparsity(1))
        .regularize_visibilities_with(PSD(1))
    )
    # fmt: on

    model = builder.get_model()

    # training epochs for loop, optimizer and all and all goes here
    # forward call on model to get loss
    loss = model()
    # loss.backward()


if __name__ == "__main__":
    main()
