from abc import ABC, abstractmethod
from warnings import warn

import numpy as np

from astropy.io import fits

from .core import header as hdr

__all__ = ['Cube', 'Map', 'Profile', 'Struct']

class Struct(ABC) :
    """ Astro structure """

    def __init__(self, data : np.ndarray, header : fits.Header) :
        """ TODO """
        if data.dtype is np.dtype('bool') :
            data = data.astype('float')
        elif data.dtype is np.dtype('int') :
            data = data.astype('float')
        if np.isinf(data).any() :
            data[np.isinf(data)] = float('nan') # Avoid inf in FITS header
        self.data : np.ndarray = data
        self.header : fits.Header = hdr.update_header(data, header)

    from .core.util import copy, from_fits, from_numpy, zeros, ones
    from_fits = classmethod(from_fits)
    from_numpy = classmethod(from_numpy)
    zeros = classmethod(zeros)
    ones = classmethod(ones)

    @abstractmethod
    def __str__(self) -> str:
        """ Returns a printable version of the structure. Can be called with str(self) """
        pass

    @property
    def shape(self : 'Struct') -> tuple[int] :
        """ Shape of the data """
        return self.data.shape

    @property
    def size(self : 'Struct') -> int :
        """ Number of scalars in the cube. """
        return self.data.size

    # Unary float operators

    from .core._op import __neg__, __abs__, __round__, __floor__, __ceil__

    # Binary float operators

    from .core._op import (__add__, __radd__, __sub__, __rsub__, __mul__, __rmul__,
        __truediv__, __rtruediv__, __floordiv__, __mod__, __pow__, __rpow__)

    # Unary boolean operators

    from .core._op import __invert__

    # Binary boolean operators

    from .core._op import __or__, __ror__, __and__, __rand__, __xor__, __rxor__

    # Comparison operators

    from .core._op import __eq__, __ne__, __ge__, __gt__, __le__, __lt__
    
    # Getitem operator

    from .core._op import __getitem__

    # Others

    from .io.io import _save_fits as save_fits
    from .io.io import plot_hist, plot_hist2d, save_hist, save_hist2d, show_hist, show_hist2d
    from .core.util import apply_element_wise, is_logical, to_numpy
    from .core.util import isnan, isfinite
    from .core.util import zeros_like, ones_like, where, clip
    from .core.util import astype
    from .reduction.stats import (any, all, sum, min, max, ptp, argmin, argmax, mean, std, var, moment, rms, median, quantile, percentile)
    from .core.math import (abs, sqrt, cbrt, exp, log, cos, sin, tan, arccos, arcsin, arctan,
        cosh, sinh, tanh, arccosh, arcsinh, arctanh)
    from .filtering.morphology import dilation, erosion, closing, opening, gradient, laplacian
    from .models.distribution import kde
    from .learning.preprocessing import standardize, unstandardize, normalize, unnormalize, scale, unscale
    from .models.noise import additive_noise, multiplicative_noise


class Cube(Struct) :
    """ Astronomical data cube """

    def __init__(self, data : np.ndarray, header : fits.Header) :
        # Check data and header number of axis
        if data.ndim != 3 :
            raise ValueError(f"data must have 3 dimensions, not {data.ndim}")
        if header['NAXIS'] != 3 :
            raise ValueError(f"header must have 3 axes, not {header['NAXIS']}")
        # Check axes compatibility
        dims = (header['NAXIS3'], header['NAXIS2'], header['NAXIS1'])
        if data.shape == (header['NAXIS3'], header['NAXIS2'], header['NAXIS1']) :
            pass
        elif data.shape == (header['NAXIS3'], header['NAXIS1'], header['NAXIS2']) :
            warn(f'Axis of cube data swapped to match shape {dims}')
            data = np.moveaxis(data, (0, 2, 1), (0, 1, 2))
        elif data.shape == (header['NAXIS2'], header['NAXIS1'], header['NAXIS3']) :
            warn(f'Axis of cube data swapped to match shape {dims}')
            data = np.moveaxis(data, (2, 0, 1), (0, 1, 2))
        elif data.shape == (header['NAXIS1'], header['NAXIS2'], header['NAXIS3']) :
            warn(f'Axis of cube data swapped to match shape {dims}')
            data = np.moveaxis(data, (2, 1, 0), (0, 1, 2))
        else :
            raise ValueError(f'Shape of data {data.shape} cannot match {dims}, even by swapping axes')
        super().__init__(data, header)

    from .core.util import cube_from_maps as from_maps, cube_from_profiles as from_profiles
    from_maps = staticmethod(from_maps)
    from_profiles = staticmethod(from_profiles)

    @property
    def nx(self : 'Struct') -> int :
        """ Length of cube x axis """
        return self.data.shape[2]

    @property
    def ny(self : 'Struct') -> int :
        """ Length of cube y axis """
        return self.data.shape[1]

    @property
    def nz(self : 'Struct') -> int :
        """ Length of cube z axis """
        return self.data.shape[0]
    
    def __str__(self) -> str:
        """ Returns a printable version of the cube. Can be called with str(self) """
        return f"Cube (nx: {self.nx}, ny: {self.ny}, nz: {self.nz})"

    from .core.util import map_from, profile_from
    from .core.util import x_axis, y_axis, z_axis, change_axes_order
    from .reduction.getters import (get_channel, get_pixel, get_channels, get_pixels,
                                 integral, spectrum, noise_map, reduce_spectral, reduce_spatial)
    from .filtering.filters import filter_pixels, filter_channels, filter_cube as filter
    from .io.io import (plot_channel, show_channel, save_channel_plot,
                        plot_pixel, show_pixel, save_pixel_plot)


class Map(Struct) :
    """ Astronomical data map """

    def __init__(self, data : np.ndarray, header : fits.Header) :
        # Check data and header number of axis
        if data.ndim != 2 :
            raise ValueError(f"data must have 2 dimensions, not {data.ndim}")
        if header['NAXIS'] != 2 :
            raise ValueError(f"header must have 2 axes, not {header['NAXIS']}")
        # Check axes compatibility
        dims = (header['NAXIS2'], header['NAXIS1'])
        if data.shape == (header['NAXIS2'], header['NAXIS1']) :
            pass
        elif data.shape == (header['NAXIS1'], header['NAXIS2']) :
            warn(f'Axis of map data swapped to match shape {dims}')
            data = data.T
        else :
            raise ValueError(f'Shape of data {data.shape} cannot match {dims}, even by swapping axes')
        super().__init__(data, header)

    @property
    def nx(self : 'Struct') -> int :
        """ Length of cube x axis """
        return self.data.shape[1]

    @property
    def ny(self : 'Struct') -> int :
        """ Length of cube y axis """
        return self.data.shape[0]

    def __str__(self) -> str:
        """ Returns a printable version of the map. Can be called with str(self) """
        return f"Map (nx: {self.nx}, ny: {self.ny})"

    from .reduction.astro import reduce_spatial
    from .core.util import x_axis, y_axis, change_axes_order
    from .filtering.filters import filter_map as filter
    from .io.io import plot_map as plot, save_map_plot as save_plot, show_map as show


class Profile(Struct) :
    """ Astronomical data profile """

    def __init__(self, data : np.ndarray, header : fits.Header) :
        # Check data and header number of axis
        if data.ndim != 1 :
            raise ValueError(f"data must have 1 dimension, not {data.ndim}")
        if header['NAXIS'] != 1 :
            raise ValueError(f"header must have 1 axis, not {header['NAXIS']}")
        super().__init__(data.flatten(), header)

    @property
    def nz(self : 'Struct') -> int :
        """ Length of cube z axis """
        return self.data.shape[0]

    def __str__(self) -> str:
        """ Returns a printable version of the profile. Can be called with str(self) """
        return f"Profile (nz: {self.nz})"

    from .reduction.astro import reduce_spectral
    from .core.util import z_axis
    from .filtering.filters import filter_profile as filter
    from .io.io import plot_profile as plot, save_profile_plot as save_plot, show_profile as show
