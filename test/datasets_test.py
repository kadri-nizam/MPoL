import copy

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from mpol import datasets


def test_dataset_device(dataset):
    # if we have a GPU available, test that we can send a dataset to it

    if torch.cuda.is_available():
        dataset = dataset.to("cuda")
        dataset = dataset.to("cpu")
    else:
        pass


def test_mask_dataset(dataset):
    updated_mask = np.ones_like(dataset.coords.packed_u_centers_2D)
    dataset.add_mask(updated_mask)


def test_dartboard_init(coords):
    datasets.Dartboard(coords=coords)


def test_dartboard_histogram(crossvalidation_products, tmp_path):
    coords, dataset = crossvalidation_products

    # use default bins
    dartboard = datasets.Dartboard(coords=coords)

    # 2D mask for any UV cells that contain visibilities
    # in *any* channel
    stacked_mask = np.any(dataset.mask.detach().numpy(), axis=0)

    # get qs, phis from dataset and turn into 1D lists
    qs = dataset.coords.packed_q_centers_2D[stacked_mask]
    phis = dataset.coords.packed_phi_centers_2D[stacked_mask]

    # use dartboard to calculate histogram
    H = dartboard.get_polar_histogram(qs, phis)

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})

    cmap = copy.copy(matplotlib.colormaps["plasma"])
    cmap.set_under("w")
    norm = matplotlib.colors.LogNorm(vmin=1)

    ax.grid(False)
    im = ax.pcolormesh(
        dartboard.phi_edges,
        dartboard.q_edges,
        H,
        shading="flat",
        norm=norm,
        cmap=cmap,
        zorder=-90,
    )
    plt.colorbar(im, ax=ax)

    ax.scatter(phis, qs, s=1.5, rasterized=True, linewidths=0.0, c="k", alpha=0.3)
    ax.set_ylim(top=2500)

    fig.savefig(tmp_path / "dartboard.png", dpi=300)

    plt.close("all")


def test_dartboard_nonzero(crossvalidation_products, tmp_path):
    coords, dataset = crossvalidation_products

    # use default bins
    dartboard = datasets.Dartboard(coords=coords)

    # 2D mask for any UV cells that contain visibilities
    # in *any* channel
    stacked_mask = np.any(dataset.mask.detach().numpy(), axis=0)

    # get qs, phis from dataset and turn into 1D lists
    qs = dataset.coords.packed_q_centers_2D[stacked_mask]
    phis = dataset.coords.packed_phi_centers_2D[stacked_mask]

    # use dartboard to calculate nonzero cells
    indices = dartboard.get_nonzero_cell_indices(qs, phis)

    fig, ax = plt.subplots(nrows=1)

    ax.scatter(*indices.T, s=1.5, rasterized=True, linewidths=0.0, c="k")
    ax.set_xlabel("q index")
    ax.set_ylabel("phi index")

    fig.savefig(tmp_path / "indices.png", dpi=300)

    plt.close("all")


def test_dartboard_mask(crossvalidation_products, tmp_path):
    coords, dataset = crossvalidation_products

    # use default bins
    dartboard = datasets.Dartboard(coords=coords)

    # 2D mask for any UV cells that contain visibilities
    # in *any* channel
    stacked_mask = np.any(dataset.mask.detach().numpy(), axis=0)

    # get qs, phis from dataset and turn into 1D lists
    qs = dataset.coords.packed_q_centers_2D[stacked_mask]
    phis = dataset.coords.packed_phi_centers_2D[stacked_mask]

    # use dartboard to calculate nonzero cells
    indices = dartboard.get_nonzero_cell_indices(qs, phis)
    print(indices)

    # get boolean mask from cell indices
    mask = np.fft.fftshift(dartboard.build_grid_mask_from_cells(indices))

    fig, ax = plt.subplots(nrows=1)

    ax.imshow(mask, origin="lower", interpolation="none")
    fig.savefig(tmp_path / "mask.png", dpi=300)

    plt.close("all")


def test_hermitian_mask_full(crossvalidation_products, tmp_path):
    coords, dataset = crossvalidation_products

    dartboard = datasets.Dartboard(coords=coords)

    chan = 4

    # do the indexing of individual points
    # plot up as function of q, phi, each point should have two dots (opacity layer).
    # make the

    # 2D mask for any UV cells that contain visibilities
    # in *any* channel
    mask = dataset.mask[chan].detach().numpy()

    # get qs, phis from dataset and turn into 1D lists
    qs = dataset.coords.packed_q_centers_2D[mask]
    phis = dataset.coords.packed_phi_centers_2D[mask]

    # use dartboard to calculate nonzero cells between 0 and pi
    indices = dartboard.get_nonzero_cell_indices(qs, phis)

    # create a new mask from this
    dartboard_mask = dartboard.build_grid_mask_from_cells(indices)

    # use this mask to index the dataset
    masked_dataset = copy.deepcopy(dataset)
    masked_dataset.add_mask(dartboard_mask)

    # get updated q and phi values
    qs = masked_dataset.coords.packed_q_centers_2D[masked_dataset.mask[chan]]
    phis = masked_dataset.coords.packed_phi_centers_2D[masked_dataset.mask[chan]]

    ind = phis <= np.pi

    fig, ax = plt.subplots(nrows=1)

    ax.plot(qs[ind], phis[ind], "o", ms=3)
    ax.plot(qs[~ind], phis[~ind] - np.pi, "o", ms=1)
    fig.savefig(tmp_path / "hermitian.png", dpi=300)
