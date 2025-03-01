import numpy as np
from astropy.io import fits

from ranch import Cube


def test_flatten():
    nz, ny, nx = 10, 20, 30
    header = fits.PrimaryHDU(np.zeros((nz, ny, nx))).header
    cube = Cube.ones(header)

    assert cube.flatten().shape == (nx * ny * nz,)
