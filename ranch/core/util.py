from typing import Callable, Literal, Optional, Sequence, Type, Union, overload

import numpy as np
from astropy.io import fits

from .. import structures as struct
from ..io import io
from . import header as hdr

__all__ = [
    "apply_element_wise",
    "astype",
    "change_axes_order",
    "clip",
    "copy",
    "cube_from",
    "cube_from_maps",
    "cube_from_profiles",
    "from_fits",
    "from_numpy",
    "is_logical",
    "isfinite",
    "isnan",
    "map_from",
    "nan_to_num",
    "ones",
    "ones_like",
    "profile_from",
    "stack_numpy",
    "to_numpy",
    "where",
    "x_axis",
    "y_axis",
    "z_axis",
    "zeros",
    "zeros_like",
]

# Axes handler


@overload
def change_axes_order(
    input: "struct.Cube", order: Literal["xyz", "yxz", "zxy", "zyx"]
) -> "struct.Cube":
    ...


@overload
def change_axes_order(input: "struct.Map", order: Literal["xy", "yx"]) -> "struct.Map":
    ...


def change_axes_order(
    input: Union["struct.Cube", "struct.Map"], order: str
) -> Union["struct.Cube", "struct.Map"]:
    """TODO"""
    header = input.header
    data = input.data

    new_header = hdr.move_header_axes(header, order)

    axes_sources = {
        2: (0, 1),
        3: (0, 1, 2),
    }

    axes_destinations = {
        "xyz": (0, 1, 2),
        "yxz": (0, 2, 1),
        "zxy": (1, 2, 0),
        "zyx": (2, 1, 0),
        "xy": (0, 1),
        "yx": (1, 0),
    }

    new_data = np.moveaxis(
        data, axes_sources[len(order)], axes_destinations[order.strip().lower()]
    )

    return type(input)(new_data, new_header)


# Element wise application


@overload
def apply_element_wise(
    input: "struct.Cube", fun: Callable[[np.ndarray], np.ndarray]
) -> "struct.Cube":
    ...


@overload
def apply_element_wise(
    input: "struct.Map", fun: Callable[[np.ndarray], np.ndarray]
) -> "struct.Map":
    ...


@overload
def apply_element_wise(
    input: "struct.Profile", fun: Callable[[np.ndarray], np.ndarray]
) -> "struct.Profile":
    ...


def apply_element_wise(
    input: "struct.Struct", fun: Callable[[np.ndarray], np.ndarray]
) -> "struct.Struct":
    """
    Apply the element-wise operator `fun` on the `input` structure.

    Parameters
    ----------
    input : `Cube | Map | Profile`
        Input structure.
    fun : `Callable[[np.ndarray], ndarray]`
        Element-wise operator.

    Returns
    -------
    out : `Cube | Map | Profile`
        Output structure.
    """
    new_data = fun(input.data)
    return type(input)(new_data, input.header)


def is_logical(input: "struct.Struct") -> bool:
    """TODO"""
    return ((input.data == 0) | (input.data == 1)).all()


# Floating type conversion


@overload
def astype(input: "struct.Cube", dtype: Literal["float", "double"]) -> "struct.Cube":
    ...


@overload
def astype(input: "struct.Map", dtype: Literal["float", "double"]) -> "struct.Map":
    ...


@overload
def astype(
    input: "struct.Profile", dtype: Literal["float", "double"]
) -> "struct.Profile":
    ...


def astype(
    input: "struct.Struct", dtype: Literal["float", "double"]
) -> "struct.Struct":
    """
    Return the input structure with floating type 'float' or 'double'.
    Note that there is no function as_int or as_float because structures are always of type float.

    Parameters
    ----------
    input : `Cube | Map | Profile`
        Input structure.
    dtype : str
        Floating type of output data. Must be 'float' or 'double'.

    Returns
    -------
    out : `Cube | Map | Profile`
        Output structure.
    """
    if dtype.lower() == "float":
        return type(input)(input.data.astype(np.single), input.header)
    if dtype.lower() == "double":
        return type(input)(input.data.astype(np.double), input.header)
    raise ValueError("dtype must be 'float' or 'double', not {dtype}")


# Element wise test for nan values


@overload
def isnan(input: "struct.Cube") -> "struct.Cube":
    ...


@overload
def isnan(input: "struct.Map") -> "struct.Map":
    ...


@overload
def isnan(input: "struct.Profile") -> "struct.Profile":
    ...


def isnan(input: "struct.Struct") -> "struct.Struct":
    """
    Return a structure similar to `input` where a sample is 1 if input same sample is `nan` and 0 if is not.

    Parameters
    ----------
    input : `Cube | Map | Profile`
        Input structure.

    Returns
    -------
    out : `Cube | Map | Profile`
        Output structure.
    """
    return type(input)(np.isnan(input.data), input.header)


@overload
def isfinite(input: "struct.Cube") -> "struct.Cube":
    ...


@overload
def isfinite(input: "struct.Map") -> "struct.Map":
    ...


@overload
def isfinite(input: "struct.Profile") -> "struct.Profile":
    ...


def isfinite(input: "struct.Struct") -> "struct.Profile":
    """
    Return a structure similar to `input` where a sample is 1 if input same sample is finite and 0 if is not.
    A finite element is a value different of `nan`, `inf` or `neginf`.
    This function is the opposite to isnan because `inf` and `neginf` are automatically casted to `nan`
    in structures constructors. So in practice : isnan(input) == ~isfinite(input).

    Parameters
    ----------
    input : `Cube | Map | Profile`
        Input structure.

    Returns
    -------
    out : `Cube | Map | Profile`
        Output structure.
    """
    return type(input)(np.isfinite(input.data), input.header)


# Nan functions


@overload
def nan_to_num(input: "struct.Cube", value: float = 0.0) -> "struct.Cube":
    ...


@overload
def nan_to_num(input: "struct.Map", value: float = 0.0) -> "struct.Map":
    ...


@overload
def nan_to_num(input: "struct.Profile", value: float = 0.0) -> "struct.Profile":
    ...


def nan_to_num(input: "struct.Struct", value: float = 0.0) -> "struct.Struct":
    """
    Returns a copy of `input` where the nans have been replaced by `value`.

    Parameters
    ----------
    input : `Cube | Map | Profile`
        Input structure.
    value : float, optional
        Value to replace `nans` elements. Default : 0.

    Returns
    -------
    out : `Cube | Map | Profile`
        Output structure.
    """
    return type(input)(np.nan_to_num(input.data, nan=value), input.header)


# Initializers


@overload
def copy(input: "struct.Cube") -> "struct.Cube":
    ...


@overload
def copy(input: "struct.Map") -> "struct.Map":
    ...


@overload
def copy(input: "struct.Profile") -> "struct.Profile":
    ...


def copy(input: "struct.Struct") -> "struct.Struct":
    """
    Returns a copy of `input`.

    Parameters
    ----------
    input : `Cube | Map | Profile`
        Input structure.

    Returns
    -------
    out : `Cube | Map | Profile`
        Output structure.
    """
    return type(input)(input.data.copy(), input.header.copy())


@overload
def from_numpy(
    cls: Type["struct.Cube"],
    array: np.ndarray,
    header: fits.Header,
    axes: Optional[Literal["xyz", "yxz", "zxy", "zyx"]] = None,
) -> "struct.Cube":
    ...


@overload
def from_numpy(
    cls: Type["struct.Map"],
    array: np.ndarray,
    header: fits.Header,
    axes: Optional[Literal["xy", "yx"]] = None,
) -> "struct.Map":
    ...


@overload
def from_numpy(
    cls: Type["struct.Profile"],
    array: np.ndarray,
    header: fits.Header,
    axes: None = None,
) -> "struct.Profile":
    ...


def from_numpy(
    cls: Type["struct.Struct"],
    array: np.ndarray,
    header: fits.Header,
    axes: Optional[str] = None,
) -> "struct.Struct":
    """
    Create an inputect of type `cls` from numpy array `array` and astropy header `header`.

    Parameters
    ----------
    cls : `Type[Cube] | Type[Map] | Type[Profile]`
        Type of structure to create.
    array : np.ndarray
        Data of structure to create. Must be 1D if cls is `Profile`, 2D if cls is `Map` and
        3D if cls is `Cube`.
    header : fits.Header
        Astropy fits header. Can be taken from another cube or created by hand.
    axes : `str`, optional
        Order of axes. Must be 'xyz', 'yxz', 'zxy', 'zyx', 'xy', 'yx' or None.

    Returns
    -------
    out : `Cube | Map | Profile`
        Created structure.
    """
    if axes is not None:
        header = hdr.move_header_axes(header, axes)
    return cls(array, header)


@overload
def from_fits(
    cls: Type["struct.Cube"],
    filename: str,
    path: str = None,
    axes=Optional[Literal["xyz", "yxz", "zxy", "zyx"]],
) -> "struct.Cube":
    ...


@overload
def from_fits(
    cls: Type["struct.Map"],
    filename: str,
    path: str = None,
    axes: Optional[Literal["xy", "yx"]] = None,
) -> "struct.Map":
    ...


@overload
def from_fits(
    cls: Type["struct.Profile"], filename: str, path: str = None, axes: None = None
) -> "struct.Profile":
    ...


def from_fits(
    cls: Type["struct.Struct"],
    filename: str,
    path: str = None,
    axes: Optional[str] = None,
) -> "struct.Struct":
    """
    Load an inputect of type `cls` from file `filename/path`.

    Parameters
    ----------
    cls : `Type[Cube] | Type[Map] | Type[Profile]`
        Type of structure to create.
    filename : `str`
        Filename of FITS file to load. Extension can be ommited. Handle both .fits or .fits.gz files,
        but notice that .gz files take longer to load.
    path : `str`, optional
        Path to the FITS file.
    axes : `str`, optional
        Order of axes. Must be 'xyz', 'yxz', 'zxy', 'zyx', 'xy', 'yx' or None.

    Returns
    -------
    out : `Cube | Map | Profile`
        Output structure.
    """
    data, header = io._load_fits(filename, path)
    if axes is not None:
        header = hdr.move_header_axes(header, axes)
    return cls(data.astype(np.float32), header)


@overload
def zeros(cls: Type["struct.Cube"], header: fits.Header) -> "struct.Cube":
    ...


@overload
def zeros(cls: Type["struct.Map"], header: fits.Header) -> "struct.Map":
    ...


@overload
def zeros(cls: Type["struct.Profile"], header: fits.Header) -> "struct.Profile":
    ...


def zeros(cls: Type["struct.Struct"], header: fits.Header) -> "struct.Struct":
    """
    Create an inputect of type `cls` fill with zeros from astropy header `header`.

    Parameters
    ----------
    cls : `Type[Cube] | Type[Map] | Type[Profile]`
        Type of structure to create.
    header : fits.Header
        Astropy fits header. Can be taken from another cube or created by hand.

    Returns
    -------
    out : `Cube | Map | Profile`
        Output structure.
    """
    shape = [header[f"NAXIS{k}"] for k in range(header["NAXIS"], 0, -1)]
    return from_numpy(cls, np.zeros(shape), header)


@overload
def ones(cls: Type["struct.Cube"], header: fits.Header) -> "struct.Cube":
    ...


@overload
def ones(cls: Type["struct.Map"], header: fits.Header) -> "struct.Map":
    ...


@overload
def ones(cls: Type["struct.Profile"], header: fits.Header) -> "struct.Profile":
    ...


def ones(cls: Type["struct.Struct"], header: fits.Header) -> "struct.Struct":
    """
    Create an inputect of type `cls` fill with ones from astropy header `header`.

    Parameters
    ----------
    cls : `Type[Cube] | Type[Map] | Type[Profile]`
        Type of structure to create.
    header : fits.Header
        Astropy fits header. Can be taken from another cube or created by hand.

    Returns
    -------
    out : `Cube | Map | Profile`
        Output structure.
    """
    shape = [header[f"NAXIS{k}"] for k in range(header["NAXIS"], 0, -1)]
    print(shape)
    return from_numpy(cls, np.ones(shape), header)


@overload
def zeros_like(input: "struct.Cube") -> "struct.Cube":
    ...


@overload
def zeros_like(input: "struct.Map") -> "struct.Map":
    ...


@overload
def zeros_like(input: "struct.Profile") -> "struct.Profile":
    ...


def zeros_like(input: "struct.Struct"):
    """
    Returns a structure with the same shape than input filled with zeros.

    Parameters
    ----------
    input : Cube | Map | Profile
        Input structure

    Returns
    -------
    out : Cube | Map | Profile
        Output structure
    """
    return type(input)(np.zeros_like(input.data), input.header)


@overload
def ones_like(input: "struct.Cube") -> "struct.Cube":
    ...


@overload
def ones_like(input: "struct.Map") -> "struct.Map":
    ...


@overload
def ones_like(input: "struct.Profile") -> "struct.Profile":
    ...


def ones_like(input: "struct.Struct") -> "struct.Struct":
    """
    Returns a structure with the same shape than input filled with ones.

    Parameters
    ----------
    input : Cube | Map | Profile
        Input structure

    Returns
    -------
    out : Cube | Map | Profile
        Output structure
    """
    return type(input)(np.ones_like(input.data), input.header)


# Derived initializers


def map_from(cube: "struct.Cube", value: float = 0.0):
    """
    Returns a map with the same x and y axis than `cube` filled with `value`.

    Parameters
    ----------
    cube : Cube
        Input cube.

    Returns
    -------
    out : Map
        Output map.
    """
    new_header = hdr.remove_header_axis(cube.header, axis="spectral")
    new_data = np.zeros((cube.ny, cube.nx)) + value
    return struct.Map(new_data, new_header)


def profile_from(cube: "struct.Cube", value: float = 0.0):
    """
    Returns a profile with the same z axis than `cube` filled with `value`.

    Parameters
    ----------
    cube : Cube
        Input cube.

    Returns
    -------
    out : Profile
        Output profile.
    """
    new_header = hdr.remove_header_axis(cube.header, axis="spatial")
    new_data = np.zeros(cube.nz) + value
    return struct.Profile(new_data, new_header)


def cube_from(map: "struct.Map", profile: "struct.Profile", value: float = 0.0):
    """
    Returns a cube with the same x and y axis than `map` and the same z axis
    than profile filled with `value`.

    Parameters
    ----------
    map : Map
        Input map.
    profile : Profile
        Input profile.

    Returns
    -------
    cube : Cube
        Output cube.
    """
    new_header = hdr.merge_headers(map.header, profile.header)
    new_data = np.zeros(profile.shape + map.shape) + value
    return struct.Cube(new_data, new_header)


def cube_from_maps(maps: Sequence["struct.Map"]) -> "struct.Cube":
    """
    Returns a cube builded by concatening maps in `maps` sequences.

    Parameters
    ----------
    maps : Sequence[Map]
        Sequence of maps of same shape. Must not be empty.

    Returns
    -------
    cube : Cube
        Output cube with same spatial shape than elements of maps.
    """
    if len(maps) == 0:
        raise ValueError("maps must not be empty")
    nz, ny, nx = len(maps), maps[0].ny, maps[0].nx
    new_data = np.zeros((nz, ny, nx))
    new_header = hdr.create_header("cube", maps[0].header, nz=nz)
    for k in range(nz):
        new_data[k, :, :] = maps[k].data
    return struct.Cube(new_data, new_header)


def cube_from_profiles(profiles: Sequence[Sequence["struct.Profile"]]) -> "struct.Cube":
    """
    Returns a cube builded by concatening maps in `maps` sequences.

    Parameters
    ----------
    profiles : Sequence[Sequence[Profile]]
        Sequence of sequence of profiles of same shape.
        The sequence must not be empty and each sub-sequence must also not be empty.
        profiles[i][j] is the pixel of the i-th row and the j-th column.

    Returns
    -------
    cube : Cube
        Output cube with same spectral shape than elements of profiles.
    """
    if len(profiles) == 0:
        raise ValueError("profiles must not be empty")
    for _, e in enumerate(profiles):
        if len(e) == 0:
            raise ValueError("Each element of profiles must be a non-empty sequence")
    ny, nx, nz = len(profiles), len(profiles[0]), profiles[0][0].nz
    new_data = np.zeros((nz, ny, nx))
    new_header = hdr.create_header("cube", profiles[0][0].header, nx=nx, ny=ny)
    for i in range(ny):
        for j in range(nx):
            new_data[:, j, i] = profiles[j][i].data
    return struct.Cube(new_data, new_header)


# Other functions


@overload
def where(
    bool_input: "struct.Cube",
    input_1: Union["struct.Cube", float],
    input_2: Union["struct.Cube", float],
) -> "struct.Cube":
    ...


@overload
def where(
    bool_input: "struct.Map",
    input_1: Union["struct.Map", float],
    input_2: Union["struct.Map", float],
) -> "struct.Map":
    ...


@overload
def where(
    bool_input: "struct.Profile",
    input_1: Union["struct.Profile", float],
    input_2: Union["struct.Profile", float],
) -> "struct.Profile":
    ...


def where(
    logical_input: "struct.Struct",
    input_1: Union["struct.Struct", float],
    input_2: Union["struct.Struct", float],
) -> "struct.Struct":
    """TODO"""
    if not isinstance(logical_input, struct.Struct):
        raise TypeError(
            f"logical_input must be an instance of Struct (Cube, Map or Profile), not {type(logical_input)}"
        )
    if not is_logical(logical_input):
        raise TypeError(
            f"logical_input must be logical i.e. contains only 0 and 1 samples"
        )
    a = input_1.data if isinstance(input_1, struct.Struct) else input_1
    b = input_2.data if isinstance(input_2, struct.Struct) else input_2
    new_data = np.where(logical_input.data == 1, a, b)
    return type(logical_input)(new_data, logical_input.header)


@overload
def clip(
    input: "struct.Cube", vmin: Optional[float], vmax: Optional[float]
) -> "struct.Cube":
    ...


@overload
def clip(
    input: "struct.Map", vmin: Optional[float], vmax: Optional[float]
) -> "struct.Map":
    ...


@overload
def clip(
    input: "struct.Profile", vmin: Optional[float], vmax: Optional[float]
) -> "struct.Profile":
    ...


def clip(
    input: "struct.Struct", vmin: Optional[float], vmax: Optional[float]
) -> "struct.Struct":
    """TODO"""
    return type(input)(np.clip(input.data, vmin, vmax), input.header)


# To numpy


def to_numpy(input: "struct.Struct", item: str) -> np.ndarray:
    """TODO"""
    if item.strip().lower() not in ["pixel", "map"]:
        raise ValueError(f"item must be 'pixel' or 'map', not {item}")
    if isinstance(input, struct.Cube):
        if item == "pixel":
            return input.data.reshape(-1, input.data.shape[1] * input.data.shape[2]).T
        else:
            return input.data
    elif isinstance(input, struct.Map):
        if item == "pixel":
            return input.data.flatten()
        else:
            raise ValueError("item = 'map' is not compatible with a Map inputect")
    elif isinstance(input, struct.Profile):
        if item == "pixel":
            raise ValueError(
                "item = 'profile' is not compatible with a Profile inputect"
            )
        else:
            return input.data
    else:
        raise TypeError(
            f"input must be an instance of Cube, Map or Profile, not {type(input)}"
        )


def stack_numpy(arrays: Sequence[np.ndarray]) -> np.ndarray:
    """TODO"""
    return np.concatenate(arrays, axis=0)


# Axes


def x_axis(
    input: Union["struct.Map", "struct.Cube"], unit: Literal["index", "angle"] = "index"
) -> np.ndarray:
    """TODO"""
    if not isinstance(input, (struct.Map, struct.Cube)):
        raise TypeError(f"input must be an instance of Map or Cube, not {type(input)}")
    unit = unit.strip().lower()
    if unit not in ["index", "angle"]:
        raise ValueError(f"unit must be 'index' or 'angle', not '{unit}'")

    axis = np.arange(input.nx)
    if unit == "angle":
        axis = hdr.indices_to_coordinates(input.header, axis, 0)[0]
    return axis


def y_axis(
    input: Union["struct.Map", "struct.Cube"], unit: Literal["index", "angle"] = "index"
) -> np.ndarray:
    """TODO"""
    if not isinstance(input, (struct.Map, struct.Cube)):
        raise TypeError(f"input must be an instance of Map or Cube, not {type(input)}")
    unit = unit.strip().lower()
    if unit not in ["index", "angle"]:
        raise ValueError(f"unit must be 'index' or 'angle', not '{unit}'")

    axis = np.arange(input.ny)
    if unit == "angle":
        axis = hdr.indices_to_coordinates(input.header, axis, 0)[1]
    return axis


def z_axis(
    input: Union["struct.Profile", "struct.Cube"],
    unit: Literal["index", "velocity", "frequency"] = "index",
) -> np.ndarray:
    """TODO"""
    if not isinstance(input, (struct.Profile, struct.Cube)):
        raise TypeError(
            f"input must be an instance of Profile or Cube, not {type(input)}"
        )
    unit = unit.strip().lower()
    if unit not in ["index", "velocity", "frequency"]:
        raise ValueError(
            f"unit must be 'index', 'velocity' or 'frequency', not '{unit}'"
        )

    axis = np.arange(input.nz)
    if unit == "velocity":
        axis = hdr.index_to_velocity(input.header, axis)
    elif unit == "frequency":
        axis = hdr.index_to_frequency(input.header, axis)
    return axis
