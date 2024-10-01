import warnings
from typing import List, Optional, Union, overload

import numpy as np

from .. import structures as struct
from ..core import header as hdr

__all__ = ["integral", "spectrum", "noise_map", "reduce_spectral", "reduce_spatial"]


def integral(
    cube: "struct.Cube",
    ppv_mask: Optional["struct.Cube"] = None,
    ignore_nans: bool = True,
) -> "struct.Map":
    """
    Returns the integral of `cube` over it spectral axis. Output unit is [cube unit] * km/s.

    Parameters
    ----------

    cube : `Cube`
        Input line cube.

    Returns
    -------

    integ : `Map`
        Output map such that `integ.nx == cube.nx` and `integ.ny == cube.ny`.

    """
    if ppv_mask is not None:
        cube = ppv_mask * cube
    return (
        cube.sum(struct.Map, ignore_nans=ignore_nans) * cube.header["CDELT3"] / 1e3
    )  # Convert to km/s


def spectrum(cube: "struct.Cube") -> "struct.Profile":
    """
    Returns the mean spectrum of `cube` ,i.e., the channel-wise average.

    Parameters
    ----------

    cube : `Cube`
        Input line cube.

    Returns
    -------

    spectrum : `Profile`
        Output spectrum such that `spectrum.nz == cube.nz`.

    """
    new_data = np.nanmean(cube.data, axis=(1, 2))
    new_header = hdr.remove_header_axis(cube.header, "spatial")
    return struct.Profile(new_data, new_header)


# TODO: noise_map -> permettre aussi d'utiliser un masque 3D
def noise_map(
    cube: "struct.Cube", signal_mask: Union[slice, List[slice]], unit="index"
) -> "struct.Map":
    """
    Returns the noise map (pixel-wise standard deviation of noise) of `cube` by hiding
    the velocity channels containing signal. If every channel contains signal,
    the noise map will be filled with `NaNs`.

    Parameters
    ----------

    cube : `Cube`
        Input line cube.
    signal_mask : `slice | list[slice]`
        Intervals of channels to hide. Only step of 1 is supported.
    unit : `str`, optional
        Describe how to read the bound values of signal_mask. Must be `'index'`, `'velocity'` or `'frequency'`.
        If `unit == 'index'`, the values are numpy indexes (starting from 0).
        If `unit == 'velocity'`, the values are in km/s.
        If `unit == 'frequency'`, the values are in GHz.

    Returns
    -------

    noise_map : `Map`
        Noise map of `cube`.
    """
    unit = unit.lower()
    if isinstance(signal_mask, slice):
        signal_mask = [signal_mask]

    new_data = cube.data.copy()
    for mask in signal_mask:
        if mask.step is not None and mask.step != 1:
            raise ValueError("Slice step different of 1 is not supported")
        if unit == "index":
            index_mask = mask
        elif unit == "velocity":
            index_mask = slice(
                hdr.velocity_to_index(cube.header, mask.start),
                hdr.velocity_to_index(cube.header, mask.stop),
            )
        elif unit == "frequency":  # Order is reversed for frequencies
            index_mask = slice(
                hdr.frequency_to_index(cube.header, mask.stop) - 1,
                hdr.frequency_to_index(cube.header, mask.start) + 1,
            )
        else:
            raise ValueError(
                f"unit must be 'index', 'velocity' or 'frequency', not '{unit}'"
            )
        new_data[index_mask] *= np.nan

    with warnings.catch_warnings():  # Suppress warning when a profile is full of NaNs
        warnings.simplefilter("ignore")
        new_data = np.nanstd(new_data, axis=0)
    new_header = hdr.remove_header_axis(cube.header, "spectral")
    return struct.Map(new_data, new_header)


@overload
def reduce_spectral(
    input: "struct.Cube", z_interv: Optional[slice] = None
) -> "struct.Cube":
    ...


@overload
def reduce_spectral(
    input: "struct.Profile", z_interv: Optional[slice] = None
) -> "struct.Profile":
    ...


def reduce_spectral(
    input: Union["struct.Cube", "struct.Profile"],
    z_interv: Optional[slice] = None,
    unit: str = "index",
) -> Union["struct.Cube", "struct.Profile"]:
    """
    z_interv must be a slice (indices begin to zero)
    unit must be 'index', 'velocity' or 'frequency'
    """
    unit = unit.lower()
    if unit not in ["index", "velocity", "frequency"]:
        raise ValueError(
            f"unit must be 'index', 'velocity' or 'frequency', not '{unit}'"
        )

    # Handle Nones
    if z_interv is None:
        return input.copy()

    # Handle unit
    if unit == "index":
        z = [z_interv.start or 0, (z_interv.stop or input.nz) - 1]
    elif unit == "velocity":
        z = [
            hdr.velocity_to_index(z_interv.start) if z_interv.start is not None else 0,
            (
                hdr.velocity_to_index(z_interv.stop)
                if z_interv.stop is not None
                else input.nz
            )
            - 1,
        ]
    elif unit == "frequency":  # Order is reversed for 'frequency' unit
        z = [
            (
                hdr.frequency_to_index(z_interv.stop)
                if z_interv.stop is not None
                else input.nz
            )
            - 1,
            hdr.frequency_to_index(z_interv.start) if z_interv.start is not None else 0,
        ]

    new_data = input.data[z_interv]
    new_header = hdr.change_coordinates(input.header, z=z)
    return type(input)(new_data, new_header)


@overload
def reduce_spatial(
    input: "struct.Cube",
    x_interv: Optional[slice] = None,
    y_interv: Optional[slice] = None,
) -> "struct.Cube":
    ...


@overload
def reduce_spatial(
    input: "struct.Map",
    x_interv: Optional[slice] = None,
    y_interv: Optional[slice] = None,
) -> "struct.Map":
    ...


def reduce_spatial(
    input: Union["struct.Cube", "struct.Map"],
    x_interv: Optional[slice] = None,
    y_interv: Optional[slice] = None,
    unit: str = "index",
) -> Union["struct.Cube", "struct.Map"]:
    """
    x_interv must be a slice (indices begin to zero)
    y_interv must be a slice (indices begin to zero)
    unit must be 'index' or 'angle'
    """
    unit = unit.lower()
    if unit not in ["index", "angle"]:
        raise ValueError(f"unit must be 'index', or 'angle', not '{unit}'")

    # Handle Nones
    if x_interv is None:
        x_interv = slice(None, None)
    if y_interv is None:
        y_interv = slice(None, None)

    # Handle unit (TODO: gérer les None -> pour le moment incorrect)
    if unit == "index":
        x = [x_interv.start, x_interv.stop - 1]
        y = [y_interv.start, y_interv.stop - 1]
    elif unit == "angle":
        x_start, y_start = hdr.coordinates_to_indices(
            input.header, x_interv.start, y_interv.start
        )
        x_stop, y_stop = hdr.coordinates_to_indices(
            input.header, x_interv.stop, y_interv.stop
        )
        x = [x_start, x_stop - 1]
        y = [y_start, y_stop - 1]

    new_data = input.data[..., y_interv, x_interv]
    new_header = hdr.change_coordinates(input.header, x=x, y=y)
    return type(input)(new_data, new_header)
