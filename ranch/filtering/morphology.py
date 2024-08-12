from typing import overload

import numpy as np
from scipy import ndimage

from .. import structures as struct

from . import filters

__all__ = [
    'StructuringElements', 'closing', 'dilation', 'erosion', 'filters', 'gradient', 'laplacian', 'opening'
]

class StructuringElements :
    """ Helpers to instantiate structuring elements """

    @staticmethod
    def segment(lg : int, filled : bool = True) -> np.ndarray :
        """ TODO """
        se = np.ones(lg, dtype = np.bool)
        if not filled :
            se[1:-1] = False
        return se

    @staticmethod
    def disk(r : int, filled : bool = True) -> np.ndarray :
        """ TODO """
        x = np.arange(2*r+1) - r
        X, Y = np.meshgrid(x, x)
        if filled :
            return (X**2 + Y**2) <= r**2
        return (X**2 + Y**2) == r**2

    @staticmethod
    def ellipse(a : int, b : int, theta : float, filled : bool = True) -> np.ndarray :
        """ TODO """
        raise NotImplementedError()

    @staticmethod
    def square(c : int, filled : bool = True) -> np.ndarray :
        """ TODO """
        return StructuringElements.rectangle(c, c)

    @staticmethod
    def rectangle(a : int, b : int, filled : bool = True) -> np.ndarray :
        """ TODO """
        se = np.ones((a, b), dtype = np.bool)
        if filled :
            se[1:-1, 1:-1] = False
        return se

    @staticmethod
    def ball(r : int, filled : bool = True) -> np.ndarray :
        """ TODO """
        x = np.arange(2*r+1) - r
        X, Y, Z = np.meshgrid(x, x, x)
        if filled :
            return (X**2 + Y**2 + Z**2) <= r**2
        return (X**2 + Y**2 + Z**2) == r**2

    @staticmethod
    def ellipsoid(rx : int, ry : int, rz : int, theta : float = 0., phi : float = 0.,
        filled : bool = True) -> np.ndarray : # TODO
        """ TODO """
        r = max(rx, ry, rz)
        x = np.arange(2*r+1) - r
        Z, Y, X = np.meshgrid(x, x, x, indexing = 'ij')
        if filled :
            return (X**2/rx**2 + Y**2/ry**2 + Z**2/rz**2) <= 1
        return (X**2/rx**2 + Y**2/ry**2 + Z**2/rz**2) == 1

    @staticmethod
    def cube(c : int, filled = True) -> np.ndarray :
        """ TODO """
        return StructuringElements.cuboid(c, c, c, filled = filled)

    @staticmethod
    def cuboid(a : int, b : int, c : int, filled : bool = True) -> np.ndarray :
        """ TODO """
        se = np.ones(a, b, c, dtype = np.bool)
        if not filled :
            se[1:-1, 1:-1, 1:-1] = False
        return se

@overload
def dilation(obj : 'struct.Cube', se : np.ndarray) -> 'struct.Cube' : ...
@overload
def dilation(obj : 'struct.Map', se : np.ndarray) -> 'struct.Map' : ...
@overload
def dilation(obj : 'struct.Profile', se : np.ndarray) -> 'struct.Profile' : ...

def dilation(obj : 'struct.Struct', se : np.ndarray) -> 'struct.Struct' :
    """ TODO """
    if isinstance(obj, struct.Cube) :
        if se.ndim == 1 :
            return obj.filter_pixels(se, filtering_mode = 'max', padding_mode = 'reflect')
        if se.ndim == 2 :
            return obj.filter_channels(se, filtering_mode = 'max', padding_mode = 'reflect')
        if se.ndim == 3 :
            return obj.filter(se, filtering_mode = 'max', padding_mode = 'reflect')
        raise ValueError(f'Structuring element se must have 1, 2 or 3 dimensions, not {se.ndim}, for obj of type Cube')
    elif isinstance(obj, struct.Map) :
        if se.ndim == 2 :
            return obj.filter(se, filtering_mode = 'max', padding_mode = 'reflect')
        raise ValueError(f'Structuring element se must have 2 dimensions, not {se.ndim}, for obj of type Map')
    elif isinstance(obj, struct.Profile) :
        if se.ndim == 1 :
            return obj.filter(se, filtering_mode = 'max', padding_mode = 'reflect')
        raise ValueError(f'Structuring element se must have 1 dimension, not {se.ndim}, for obj of type Profile')
    else :
        raise ValueError(f'obj must an instance of Cube, Map or Profile, not {type(obj)}')

@overload
def erosion(obj : 'struct.Cube', se : np.ndarray) -> 'struct.Cube' : ...
@overload
def erosion(obj : 'struct.Map', se : np.ndarray) -> 'struct.Map' : ...
@overload
def erosion(obj : 'struct.Profile', se : np.ndarray) -> 'struct.Profile' : ...

def erosion(obj : 'struct.Struct', se : np.ndarray) -> 'struct.Struct' :
    """ TODO """
    if isinstance(obj, struct.Cube) :
        if se.ndim == 1 :
            return obj.filter_pixels(se, filtering_mode = 'min', padding_mode = 'reflect')
        if se.ndim == 2 :
            return obj.filter_channels(se, filtering_mode = 'min', padding_mode = 'reflect')
        if se.ndim == 3 :
            return obj.filter(se, 'min')
        raise ValueError(f'Structuring element se must have 1, 2 or 3 dimensions, not {se.ndim}, for obj of type Cube')
    elif isinstance(obj, struct.Map) :
        if se.ndim == 2 :
            return obj.filter(se, filtering_mode = 'min', padding_mode = 'reflect')
        raise ValueError(f'Structuring element se must have 2 dimensions, not {se.ndim}, for obj of type Map')
    elif isinstance(obj, struct.Profile) :
        if se.ndim == 1 :
            return obj.filter(se, filtering_mode = 'min', padding_mode = 'reflect')
        raise ValueError(f'Structuring element se must have 1 dimension, not {se.ndim}, for obj of type Profile')
    else :
        raise ValueError(f'obj must an instance of Cube, Map or Profile, not {type(obj)}')

@overload
def closing(obj : 'struct.Cube', se : np.ndarray) -> 'struct.Cube' : ...
@overload
def closing(obj : 'struct.Map', se : np.ndarray) -> 'struct.Map' : ...
@overload
def closing(obj : 'struct.Profile', se : np.ndarray) -> 'struct.Profile' : ...

def closing(obj : 'struct.Struct', se : np.ndarray) -> 'struct.Struct' :
    """ TODO """
    return erosion(dilation(obj, se), se)

@overload
def opening(obj : 'struct.Cube', se : np.ndarray) -> 'struct.Cube' : ...
@overload
def opening(obj : 'struct.Map', se : np.ndarray) -> 'struct.Map' : ...
@overload
def opening(obj : 'struct.Profile', se : np.ndarray) -> 'struct.Profile' : ...

def opening(obj : 'struct.Struct', se : np.ndarray) -> 'struct.Struct' :
    """ TODO """
    return dilation(erosion(obj, se), se)

@overload
def gradient(obj : 'struct.Cube', se : np.ndarray) -> 'struct.Cube' : ...
@overload
def gradient(obj : 'struct.Map', se : np.ndarray) -> 'struct.Map' : ...
@overload
def gradient(obj : 'struct.Profile', se : np.ndarray) -> 'struct.Profile' : ...

def gradient(obj : 'struct.Struct', se : np.ndarray) -> 'struct.Struct' :
    """ TODO """
    return dilation(obj, se) - erosion(obj, se)

@overload
def laplacian(obj : 'struct.Cube', se : np.ndarray) -> 'struct.Cube' : ...
@overload
def laplacian(obj : 'struct.Map', se : np.ndarray) -> 'struct.Map' : ...
@overload
def laplacian(obj : 'struct.Profile', se : np.ndarray) -> 'struct.Profile' : ...

def laplacian(obj : 'struct.Struct', se : np.ndarray) -> 'struct.Struct' :
    """ TODO """
    return dilation(obj, se) + erosion(obj, se) - 2*obj
