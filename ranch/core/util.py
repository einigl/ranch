from typing import Literal, Optional, Union, Callable, Sequence, Type, overload

import numpy as np
from astropy.io import fits

from . import header as hdr

from .. import structures as struct
from ..io import io

__all__ = [
    'apply_element_wise', 'astype', 'change_axes_order', 'clip', 'copy', 'cube_from', 'cube_from_maps', 'cube_from_profiles', 'from_fits', 'from_numpy', 'is_logical', 'isfinite', 'isnan', 'map_from', 'nan_to_num', 'ones', 'ones_like', 'profile_from', 'stack_numpy', 'to_numpy', 'where', 'x_axis', 'y_axis', 'z_axis', 'zeros', 'zeros_like'
]

# Axes handler

@overload
def change_axes_order(obj : 'struct.Cube', order: Literal['xyz', 'yxz', 'zxy', 'zyx']) -> 'struct.Cube' : ...
@overload
def change_axes_order(obj : 'struct.Map', order: Literal['xy', 'yx']) -> 'struct.Map' : ...

def change_axes_order(obj: Union['struct.Cube', 'struct.Map'], order: str) -> Union['struct.Cube', 'struct.Map']:
    """ TODO """
    header = obj.header
    data = obj.data

    new_header = hdr.move_header_axes(header, order)

    axes_sources = {
        2: (0,1),
        3: (0,1,2),
    }

    axes_destinations = {
        'xyz' : (0,1,2),
        'yxz' : (0,2,1),
        'zxy' : (1,2,0),
        'zyx' : (2,1,0),
        'xy' : (0,1),
        'yx' : (1,0),
    }

    new_data = np.moveaxis(data, axes_sources[len(order)], axes_destinations[order.strip().lower()])

    return type(obj)(new_data, new_header)

# Element wise application

@overload
def apply_element_wise(obj : 'struct.Cube', fun : Callable[[np.ndarray], np.ndarray]) -> 'struct.Cube' : ...
@overload
def apply_element_wise(obj : 'struct.Map', fun : Callable[[np.ndarray], np.ndarray]) -> 'struct.Map' : ...
@overload
def apply_element_wise(obj : 'struct.Profile', fun : Callable[[np.ndarray], np.ndarray]) -> 'struct.Profile' : ...

def apply_element_wise(obj : 'struct.Struct', fun : Callable[[np.ndarray], np.ndarray])\
    -> 'struct.Struct' :
    """
    Apply the element-wise operator `fun` on the `obj` structure.

    Parameters
    ----------
    obj : `Cube | Map | Profile`
        Input structure.
    fun : `Callable[[np.ndarray], ndarray]`
        Element-wise operator.

    Returns
    -------
    out : `Cube | Map | Profile`
        Output structure.
    """
    new_data = fun(obj.data)
    return type(obj)(new_data, obj.header)

def is_logical(obj : 'struct.Struct') -> bool :
    """ TODO """
    return ((obj.data == 0) | (obj.data == 1)).all()

# Floating type conversion

@overload
def astype(obj : 'struct.Cube', dtype : Literal['float', 'double']) -> 'struct.Cube' : ...
@overload
def astype(obj : 'struct.Map', dtype : Literal['float', 'double']) -> 'struct.Map' : ...
@overload
def astype(obj : 'struct.Profile', dtype : Literal['float', 'double']) -> 'struct.Profile' : ...

def astype(obj : 'struct.Struct', dtype : Literal['float', 'double']) -> 'struct.Struct' :
    """
    Return the obj structure with floating type 'float' or 'double'.
    Note that there is no function as_int or as_float because structures are always of type float.

    Parameters
    ----------
    obj : `Cube | Map | Profile`
        Input structure.
    dtype : str
        Floating type of output data. Must be 'float' or 'double'.

    Returns
    -------
    out : `Cube | Map | Profile`
        Output structure.
    """
    if dtype.lower() == 'float' :
        return type(obj)(obj.data.astype(np.single), obj.header)
    if dtype.lower() == 'double' :
        return type(obj)(obj.data.astype(np.double), obj.header)
    raise ValueError("dtype must be 'float' or 'double', not {dtype}")

# Element wise test for nan values

@overload
def isnan(obj : 'struct.Cube') -> 'struct.Cube' : ...
@overload
def isnan(obj : 'struct.Map') -> 'struct.Map' : ...
@overload
def isnan(obj : 'struct.Profile') -> 'struct.Profile' : ...

def isnan(obj : 'struct.Struct') -> 'struct.Struct' :
    """
    Return a structure similar to `obj` where a sample is 1 if obj same sample is `nan` and 0 if is not.

    Parameters
    ----------
    obj : `Cube | Map | Profile`
        Input structure.

    Returns
    -------
    out : `Cube | Map | Profile`
        Output structure.
    """
    return type(obj)(np.isnan(obj.data), obj.header)

@overload
def isfinite(obj : 'struct.Cube') -> 'struct.Cube' : ...
@overload
def isfinite(obj : 'struct.Map') -> 'struct.Map' : ...
@overload
def isfinite(obj : 'struct.Profile') -> 'struct.Profile' : ...

def isfinite(obj : 'struct.Struct') -> 'struct.Profile' :
    """
    Return a structure similar to `obj` where a sample is 1 if obj same sample is finite and 0 if is not.
    A finite element is a value different of `nan`, `inf` or `neginf`.
    This function is the opposite to isnan because `inf` and `neginf` are automatically casted to `nan`
    in structures constructors. So in practice : isnan(obj) == ~isfinite(obj).
    
    Parameters
    ----------
    obj : `Cube | Map | Profile`
        Input structure.

    Returns
    -------
    out : `Cube | Map | Profile`
        Output structure.
    """
    return type(obj)(np.isfinite(obj.data), obj.header)

# Nan functions

@overload
def nan_to_num(obj : 'struct.Cube', value : float = 0.) -> 'struct.Cube' : ...
@overload
def nan_to_num(obj : 'struct.Map', value : float = 0.) -> 'struct.Map' : ...
@overload
def nan_to_num(obj : 'struct.Profile', value : float = 0.) -> 'struct.Profile' : ...

def nan_to_num(obj : 'struct.Struct', value : float = 0.) -> 'struct.Struct' :
    """
    Returns a copy of `obj` where the nans have been replaced by `value`.
    
    Parameters
    ----------
    obj : `Cube | Map | Profile`
        Input structure.
    value : float, optional
        Value to replace `nans` elements. Default : 0.

    Returns
    -------
    out : `Cube | Map | Profile`
        Output structure.
    """
    return type(obj)(np.nan_to_num(obj.data, nan = value), obj.header)

# Initializers

@overload
def copy(obj : 'struct.Cube') -> 'struct.Cube' : ...
@overload
def copy(obj : 'struct.Map') -> 'struct.Map' : ...
@overload
def copy(obj : 'struct.Profile') -> 'struct.Profile' : ...

def copy(obj : 'struct.Struct') -> 'struct.Struct' :
    """
    Returns a copy of `obj`.
    
    Parameters
    ----------
    obj : `Cube | Map | Profile`
        Input structure.

    Returns
    -------
    out : `Cube | Map | Profile`
        Output structure.
    """
    return type(obj)(obj.data.copy(), obj.header.copy())

@overload
def from_numpy(cls : Type['struct.Cube'], array : np.ndarray, header : fits.Header,
    axes : Optional[Literal['xyz', 'yxz', 'zxy', 'zyx']] = None) -> 'struct.Cube' : ...
@overload
def from_numpy(cls : Type['struct.Map'], array : np.ndarray, header : fits.Header,
    axes : Optional[Literal['xy', 'yx']] = None) -> 'struct.Map' : ...
@overload
def from_numpy(cls : Type['struct.Profile'], array : np.ndarray, header : fits.Header,
    axes : None = None) -> 'struct.Profile' : ...

def from_numpy(cls : Type['struct.Struct'], array : np.ndarray, header : fits.Header,
    axes : Optional[str] = None) -> 'struct.Struct' :
    """
    Create an object of type `cls` from numpy array `array` and astropy header `header`.
    
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
    if axes is not None :
        header = hdr.move_header_axes(header, axes)
    return cls(array, header)

@overload
def from_fits(cls : Type['struct.Cube'], filename : str, path : str = None,
    axes = Optional[Literal['xyz', 'yxz', 'zxy', 'zyx']]) -> 'struct.Cube' : ...
@overload
def from_fits(cls : Type['struct.Map'], filename : str, path : str = None,
    axes : Optional[Literal['xy', 'yx']] = None) -> 'struct.Map' : ...
@overload
def from_fits(cls : Type['struct.Profile'], filename : str, path : str = None,
    axes : None = None) -> 'struct.Profile' : ...

def from_fits(cls : Type['struct.Struct'], filename : str, path : str = None,
    axes : Optional[str] = None) -> 'struct.Struct' :
    """
    Load an object of type `cls` from file `filename/path`.
    
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
    if axes is not None :
        header = hdr.move_header_axes(header, axes)
    return cls(data.astype(np.float32), header)

@overload
def zeros(cls : Type['struct.Cube'], header : fits.Header) -> 'struct.Cube' : ...
@overload
def zeros(cls : Type['struct.Map'], header : fits.Header) -> 'struct.Map' : ...
@overload
def zeros(cls : Type['struct.Profile'], header : fits.Header) -> 'struct.Profile' : ...

def zeros(cls : Type['struct.Struct'], header : fits.Header) -> 'struct.Struct' :
    """
    Create an object of type `cls` fill with zeros from astropy header `header`.
    
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
    shape = [header[f'NAXIS{k}'] for k in range(header['NAXIS'], 0, -1)]
    return from_numpy(cls, np.zeros(shape), header)

@overload
def ones(cls : Type['struct.Cube'], header : fits.Header) -> 'struct.Cube' : ...
@overload
def ones(cls : Type['struct.Map'], header : fits.Header) -> 'struct.Map' : ...
@overload
def ones(cls : Type['struct.Profile'], header : fits.Header) -> 'struct.Profile' : ...

def ones(cls : Type['struct.Struct'], header : fits.Header) -> 'struct.Struct' :
    """
    Create an object of type `cls` fill with ones from astropy header `header`.
    
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
    shape = [header[f'NAXIS{k}'] for k in range(header['NAXIS'], 0, -1)]
    print(shape)
    return from_numpy(cls, np.ones(shape), header)

@overload
def zeros_like(obj : 'struct.Cube') -> 'struct.Cube' : ...
@overload
def zeros_like(obj : 'struct.Map') -> 'struct.Map' : ...
@overload
def zeros_like(obj : 'struct.Profile') -> 'struct.Profile' : ...

def zeros_like(obj : 'struct.Struct') :
    """
    Returns a structure with the same shape than obj filled with zeros.
    
    Parameters
    ----------
    obj : Cube | Map | Profile
        Input structure

    Returns
    -------
    out : Cube | Map | Profile
        Output structure
    """
    return type(obj)(np.zeros_like(obj.data), obj.header)

@overload
def ones_like(obj : 'struct.Cube') -> 'struct.Cube' : ...
@overload
def ones_like(obj : 'struct.Map') -> 'struct.Map' : ...
@overload
def ones_like(obj : 'struct.Profile') -> 'struct.Profile' : ...

def ones_like(obj : 'struct.Struct') -> 'struct.Struct' :
    """
    Returns a structure with the same shape than obj filled with ones.
    
    Parameters
    ----------
    obj : Cube | Map | Profile
        Input structure

    Returns
    -------
    out : Cube | Map | Profile
        Output structure
    """
    return type(obj)(np.ones_like(obj.data), obj.header)

# Derived initializers

def map_from(cube : 'struct.Cube', value : float = 0.) :
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
    new_header = hdr.remove_header_axis(cube.header, axis = 'spectral')
    new_data = np.zeros((cube.ny, cube.nx)) + value
    return struct.Map(new_data, new_header)

def profile_from(cube : 'struct.Cube', value : float = 0.) :
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
    new_header = hdr.remove_header_axis(cube.header, axis = 'spatial')
    new_data = np.zeros(cube.nz) + value
    return struct.Profile(new_data, new_header)

def cube_from(map : 'struct.Map', profile : 'struct.Profile', value : float = 0.) :
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

def cube_from_maps(maps : Sequence['struct.Map']) -> 'struct.Cube' :
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
    if len(maps) == 0 :
        raise ValueError('maps must not be empty')
    nz, ny, nx = len(maps), maps[0].ny, maps[0].nx
    new_data = np.zeros( (nz, ny, nx) )
    new_header = hdr.create_header('cube', maps[0].header, nz = nz)
    for k in range(nz) :
        new_data[k, :, :] = maps[k].data
    return struct.Cube(new_data, new_header)

def cube_from_profiles(profiles : Sequence[Sequence['struct.Profile']]) -> 'struct.Cube' :
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
    if len(profiles) == 0 :
        raise ValueError('profiles must not be empty')
    for _, e in enumerate(profiles) :
        if len(e) == 0 :
            raise ValueError('Each element of profiles must be a non-empty sequence')
    ny, nx, nz = len(profiles), len(profiles[0]), profiles[0][0].nz
    new_data = np.zeros( (nz, ny, nx) )
    new_header = hdr.create_header('cube', profiles[0][0].header, nx = nx, ny = ny)
    for i in range(ny) :
        for j in range(nx) :
            new_data[:, j, i] = profiles[j][i].data
    return struct.Cube(new_data, new_header)


# Other functions

@overload
def where(bool_obj : 'struct.Cube', obj_1 : Union['struct.Cube', float],
    obj_2 : Union['struct.Cube', float]) -> 'struct.Cube' : ...
@overload
def where(bool_obj : 'struct.Map', obj_1 : Union['struct.Map', float],
    obj_2 : Union['struct.Map', float]) -> 'struct.Map' : ...
@overload
def where(bool_obj : 'struct.Profile', obj_1 : Union['struct.Profile', float],
    obj_2 : Union['struct.Profile', float]) -> 'struct.Profile' : ...

def where(logical_obj : 'struct.Struct', obj_1 : Union['struct.Struct', float],
    obj_2 : Union['struct.Struct', float]) -> 'struct.Struct' :
    """ TODO """
    if not isinstance(logical_obj, struct.Struct) :
        raise TypeError(f"logical_obj must be an instance of Struct (Cube, Map or Profile), not {type(logical_obj)}")
    if not is_logical(logical_obj) :
        raise TypeError(f"logical_obj must be logical i.e. contains only 0 and 1 samples")
    a = obj_1.data if isinstance(obj_1, struct.Struct) else obj_1
    b = obj_2.data if isinstance(obj_2, struct.Struct) else obj_2
    new_data = np.where(logical_obj.data == 1, a, b)
    return type(logical_obj)(new_data, logical_obj.header)

@overload
def clip(obj : 'struct.Cube', vmin : Optional[float], vmax : Optional[float]) -> 'struct.Cube' : ...
@overload
def clip(obj : 'struct.Map', vmin : Optional[float], vmax : Optional[float]) -> 'struct.Map' : ...
@overload
def clip(obj : 'struct.Profile', vmin : Optional[float], vmax : Optional[float]) -> 'struct.Profile' : ...

def clip(obj : 'struct.Struct', vmin : Optional[float], vmax : Optional[float]) -> 'struct.Struct':
    """ TODO """
    return type(obj)(np.clip(obj.data, vmin, vmax), obj.header)


# To numpy

def to_numpy(obj : 'struct.Struct', item : str) -> np.ndarray :
    """ TODO """
    if item.strip().lower() not in ['pixel', 'map'] :
        raise ValueError(f"item must be 'pixel' or 'map', not {item}")
    if isinstance(obj, struct.Cube) :
        if item == 'pixel' :
            return obj.data.reshape(-1, obj.data.shape[1]*obj.data.shape[2]).T
        else :
            return obj.data
    elif isinstance(obj, struct.Map) :
        if item == 'pixel' :
            return obj.data.flatten()
        else :
            raise ValueError("item = 'map' is not compatible with a Map object")
    elif isinstance(obj, struct.Profile) :
        if item == 'pixel' :
            raise ValueError("item = 'profile' is not compatible with a Profile object")
        else :
            return obj.data
    else :
        raise TypeError(f"obj must be an instance of Cube, Map or Profile, not {type(obj)}")

def stack_numpy(arrays : Sequence[np.ndarray]) -> np.ndarray :
    """ TODO """
    return np.concatenate(arrays, axis = 0)


# Axes

def x_axis(obj : Union['struct.Map', 'struct.Cube'],
    unit : Literal['index', 'angle'] = 'index') -> np.ndarray :
    """ TODO """
    if not isinstance(obj, (struct.Map, struct.Cube)) :
        raise TypeError(f'obj must be an instance of Map or Cube, not {type(obj)}')
    unit = unit.strip().lower()
    if unit not in ['index', 'angle'] :
        raise ValueError(f"unit must be 'index' or 'angle', not '{unit}'")
    
    axis = np.arange(obj.nx)
    if unit == 'angle' :
        axis = hdr.indices_to_coordinates(obj.header, axis, 0)[0]
    return axis

def y_axis(obj : Union['struct.Map', 'struct.Cube'],
    unit : Literal['index', 'angle'] = 'index') -> np.ndarray :
    """ TODO """
    if not isinstance(obj, (struct.Map, struct.Cube)) :
        raise TypeError(f'obj must be an instance of Map or Cube, not {type(obj)}')
    unit = unit.strip().lower()
    if unit not in ['index', 'angle'] :
        raise ValueError(f"unit must be 'index' or 'angle', not '{unit}'")

    axis = np.arange(obj.ny)
    if unit == 'angle' :
        axis = hdr.indices_to_coordinates(obj.header, axis, 0)[1]
    return axis

def z_axis(obj : Union['struct.Profile', 'struct.Cube'],
    unit : Literal['index', 'velocity', 'frequency'] = 'index') -> np.ndarray :
    """ TODO """
    if not isinstance(obj, (struct.Profile, struct.Cube)) :
        raise TypeError(f'obj must be an instance of Profile or Cube, not {type(obj)}')
    unit = unit.strip().lower()
    if unit not in ['index', 'velocity', 'frequency'] :
        raise ValueError(f"unit must be 'index', 'velocity' or 'frequency', not '{unit}'")

    axis = np.arange(obj.nz)
    if unit == 'velocity' :
        axis = hdr.index_to_velocity(obj.header, axis)
    elif unit == 'frequency' :
        axis = hdr.index_to_frequency(obj.header, axis)
    return axis
