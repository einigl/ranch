from abc import ABC, abstractmethod
from warnings import warn

import numpy as np
from astropy.io import fits

from .core import header as hdr

__all__ = ["Cube", "Map", "Profile", "Struct"]


class Struct(ABC):
    """Radio astronomy multidimensional data"""

    def __init__(self, data: np.ndarray, header: fits.Header):
        """
        Initializer.

        Parameters
        ----------
        data : np.ndarray
            Data array.
        header : fits.Header
            FITS header.
        """
        if data.dtype is np.dtype("bool"):
            data = data.astype("float")
        elif data.dtype is np.dtype("int"):
            data = data.astype("float")
        if np.isinf(data).any():
            data[np.isinf(data)] = float("nan")  # Avoid inf in FITS header
        self.data: np.ndarray = data  #: Data array
        self.header: fits.Header = hdr.update_header(data, header)  #: FITS header

    from .core.util import copy, from_fits, from_numpy, ones, zeros

    from_fits = classmethod(from_fits)
    from_numpy = classmethod(from_numpy)
    zeros = classmethod(zeros)
    ones = classmethod(ones)

    @abstractmethod
    def __str__(self) -> str:
        """Returns a printable version of the structure. Can be called with str(self)"""
        pass

    @property
    def shape(self: "Struct") -> tuple[int]:
        """Shape of the data"""
        return self.data.shape

    @property
    def size(self: "Struct") -> int:
        """Number of scalars in the cube."""
        return self.data.size

    # Unary float operators

    # Getitem operator
    # Comparison operators
    # Binary boolean operators
    # Unary boolean operators
    # Binary float operators
    from .core._op import (
        __abs__,
        __add__,
        __and__,
        __ceil__,
        __eq__,
        __floor__,
        __floordiv__,
        __ge__,
        __getitem__,
        __gt__,
        __invert__,
        __le__,
        __lt__,
        __mod__,
        __mul__,
        __ne__,
        __neg__,
        __or__,
        __pow__,
        __radd__,
        __rand__,
        __rmul__,
        __ror__,
        __round__,
        __rpow__,
        __rsub__,
        __rtruediv__,
        __rxor__,
        __sub__,
        __truediv__,
        __xor__,
    )
    from .core.math import (
        abs,
        arccos,
        arccosh,
        arcsin,
        arcsinh,
        arctan,
        arctanh,
        cbrt,
        cos,
        cosh,
        exp,
        log,
        sin,
        sinh,
        sqrt,
        tan,
        tanh,
    )
    from .core.util import (
        apply_element_wise,
        astype,
        clip,
        is_logical,
        isfinite,
        isnan,
        ones_like,
        to_numpy,
        where,
        zeros_like,
    )
    from .filtering.morphology import (
        closing,
        dilation,
        erosion,
        gradient,
        laplacian,
        opening,
    )

    # Others
    from .io.io import _save_fits as save_fits
    from .io.io import (
        plot_hist,
        plot_hist2d,
        save_hist,
        save_hist2d,
        show_hist,
        show_hist2d,
    )
    from .learning.preprocessing import (
        normalize,
        scale,
        standardize,
        unnormalize,
        unscale,
        unstandardize,
    )
    from .models.distribution import kde
    from .models.noise import additive_noise, multiplicative_noise
    from .reduction.stats import (
        all,
        any,
        argmax,
        argmin,
        max,
        mean,
        median,
        min,
        moment,
        percentile,
        ptp,
        quantile,
        rms,
        std,
        sum,
        var,
    )


class Cube(Struct):
    """Radio astronomy data cube"""

    def __init__(self, data: np.ndarray, header: fits.Header):
        # Check data and header number of axis
        if data.ndim != 3:
            raise ValueError(f"data must have 3 dimensions, not {data.ndim}")
        if header["NAXIS"] != 3:
            raise ValueError(f"header must have 3 axes, not {header['NAXIS']}")
        # Check axes compatibility
        dims = (header["NAXIS3"], header["NAXIS2"], header["NAXIS1"])
        if data.shape == (header["NAXIS3"], header["NAXIS2"], header["NAXIS1"]):
            pass
        elif data.shape == (header["NAXIS3"], header["NAXIS1"], header["NAXIS2"]):
            warn(f"Axis of cube data swapped to match shape {dims}")
            data = np.moveaxis(data, (0, 2, 1), (0, 1, 2))
        elif data.shape == (header["NAXIS2"], header["NAXIS1"], header["NAXIS3"]):
            warn(f"Axis of cube data swapped to match shape {dims}")
            data = np.moveaxis(data, (2, 0, 1), (0, 1, 2))
        elif data.shape == (header["NAXIS1"], header["NAXIS2"], header["NAXIS3"]):
            warn(f"Axis of cube data swapped to match shape {dims}")
            data = np.moveaxis(data, (2, 1, 0), (0, 1, 2))
        else:
            raise ValueError(
                f"Shape of data {data.shape} cannot match {dims}, even by swapping axes"
            )
        super().__init__(data, header)

    from .core.util import cube_from_maps as from_maps
    from .core.util import cube_from_profiles as from_profiles

    from_maps = staticmethod(from_maps)
    from_profiles = staticmethod(from_profiles)

    @property
    def nx(self: "Struct") -> int:
        """Length of cube x axis"""
        return self.data.shape[2]

    @property
    def ny(self: "Struct") -> int:
        """Length of cube y axis"""
        return self.data.shape[1]

    @property
    def nz(self: "Struct") -> int:
        """Length of cube z axis"""
        return self.data.shape[0]

    def __str__(self) -> str:
        """Returns a printable version of the cube. Can be called with str(self)"""
        return f"Cube (nx: {self.nx}, ny: {self.ny}, nz: {self.nz})"

    from .core.util import (
        change_axes_order,
        map_from,
        profile_from,
        x_axis,
        y_axis,
        z_axis,
    )
    from .filtering.filters import filter_channels
    from .filtering.filters import filter_cube as filter
    from .filtering.filters import filter_pixels
    from .io.io import (
        plot_channel,
        plot_pixel,
        save_channel_plot,
        save_pixel_plot,
        show_channel,
        show_pixel,
    )
    from .reduction.astro import (
        integral,
        noise_map,
        reduce_spatial,
        reduce_spectral,
        spectrum,
    )
    from .reduction.getters import get_channel, get_channels, get_pixel, get_pixels


class Map(Struct):
    """Radio astronomy data map"""

    def __init__(self, data: np.ndarray, header: fits.Header):
        # Check data and header number of axis
        if data.ndim != 2:
            raise ValueError(f"data must have 2 dimensions, not {data.ndim}")
        if header["NAXIS"] != 2:
            raise ValueError(f"header must have 2 axes, not {header['NAXIS']}")
        # Check axes compatibility
        dims = (header["NAXIS2"], header["NAXIS1"])
        if data.shape == (header["NAXIS2"], header["NAXIS1"]):
            pass
        elif data.shape == (header["NAXIS1"], header["NAXIS2"]):
            warn(f"Axis of map data swapped to match shape {dims}")
            data = data.T
        else:
            raise ValueError(
                f"Shape of data {data.shape} cannot match {dims}, even by swapping axes"
            )
        super().__init__(data, header)

    @property
    def nx(self: "Struct") -> int:
        """Length of cube x axis"""
        return self.data.shape[1]

    @property
    def ny(self: "Struct") -> int:
        """Length of cube y axis"""
        return self.data.shape[0]

    def __str__(self) -> str:
        """Returns a printable version of the map. Can be called with str(self)"""
        return f"Map (nx: {self.nx}, ny: {self.ny})"

    from .core.util import change_axes_order, x_axis, y_axis
    from .filtering.filters import filter_map as filter
    from .io.io import plot_map as plot
    from .io.io import save_map_plot as save_plot
    from .io.io import show_map as show
    from .reduction.astro import reduce_spatial


class Profile(Struct):
    """Radio astronomy data profile"""

    def __init__(self, data: np.ndarray, header: fits.Header):
        # Check data and header number of axis
        if data.ndim != 1:
            raise ValueError(f"data must have 1 dimension, not {data.ndim}")
        if header["NAXIS"] != 1:
            raise ValueError(f"header must have 1 axis, not {header['NAXIS']}")
        super().__init__(data.flatten(), header)

    @property
    def nz(self: "Struct") -> int:
        """Length of cube z axis"""
        return self.data.shape[0]

    def __str__(self) -> str:
        """Returns a printable version of the profile. Can be called with str(self)"""
        return f"Profile (nz: {self.nz})"

    from .core.util import z_axis
    from .filtering.filters import filter_profile as filter
    from .io.io import plot_profile as plot
    from .io.io import save_profile_plot as save_plot
    from .io.io import show_profile as show
    from .reduction.astro import reduce_spectral
