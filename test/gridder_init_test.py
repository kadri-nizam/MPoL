import pytest

from mpol import coordinates, gridding
from mpol.constants import *
from mpol.exceptions import CellSizeError, DataError


def test_hermitian_pairs(mock_visibility_data):
    # Test to see whether our routine checking whether Hermitian pairs
    # exist in the dataset works correctly in the False and True cases

    uu, vv, weight, data_re, data_im = mock_visibility_data

    # should *NOT* contain Hermitian pairs
    gridding.verify_no_hermitian_pairs(uu, vv, data_re + 1.0j * data_im)

    # expand the vectors to include complex conjugates
    uu = np.concatenate([uu, -uu], axis=1)
    vv = np.concatenate([vv, -vv], axis=1)
    data_re = np.concatenate([data_re, data_re], axis=1)
    data_im = np.concatenate([data_im, -data_im], axis=1)

    # should contain Hermitian pairs
    with pytest.raises(
        DataError,
        match="Hermitian pairs were found in the data. Please provide data without Hermitian pairs.",
    ):
        gridding.verify_no_hermitian_pairs(uu, vv, data_re + 1.0j * data_im)


def test_averager_instantiate_cell_npix(mock_visibility_data):
    uu, vv, weight, data_re, data_im = mock_visibility_data

    gridding.DataAverager.from_image_properties(
        cell_size=0.005,
        npix=800,
        uu=uu,
        vv=vv,
        weight=weight,
        data_re=data_re,
        data_im=data_im,
    )


def test_averager_instantiate_gridCoord(mock_visibility_data):
    uu, vv, weight, data_re, data_im = mock_visibility_data

    mycoords = coordinates.GridCoords(cell_size=0.005, npix=800)

    gridding.DataAverager(
        coords=mycoords,
        uu=uu,
        vv=vv,
        weight=weight,
        data_re=data_re,
        data_im=data_im,
    )


def test_averager_instantiate_bounds_fail(mock_visibility_data):
    uu, vv, weight, data_re, data_im = mock_visibility_data

    mycoords = coordinates.GridCoords(cell_size=0.05, npix=800)

    with pytest.raises(CellSizeError):
        gridding.DataAverager(
            coords=mycoords,
            uu=uu,
            vv=vv,
            weight=weight,
            data_re=data_re,
            data_im=data_im,
        )
