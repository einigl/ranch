from math import sqrt
from typing import Literal, Union, overload
from numbers import Number

import numpy as np

from .. import structures as struct

__all__ = [
    'additive_noise', 'multiplicative_noise'
]

def _std_broadcasting(obj : 'struct.Struct', std : Union['struct.Struct', float] = 1.)\
    -> np.ndarray :
    std = std if std is not None else 1
    if isinstance(obj, struct.Cube) :
        if isinstance(std, struct.Cube) :
            std_ = std.data
        elif isinstance(std, struct.Map) :
            std_ = std.data[np.newaxis, :, :] * np.ones((obj.nz, 1, 1))
        elif isinstance(std, struct.Profile) :
            std_ = std.data[:, np.newaxis, np.newaxis] * np.ones((1, obj.ny, obj.nx))
        elif isinstance(std, Number) :
            std_ = std * np.ones((obj.nz, obj.ny, obj.nx))
        else :
            raise ValueError(f"std is of type {type(std)} which is incompatible for obj of type {type(obj)}")
    elif isinstance(obj, struct.Map) :
        if isinstance(std, struct.Map) :
            std_ = std.data
        elif isinstance(std, Number) :
            std_ = std * np.ones((obj.ny, obj.nx))
        else :
            raise ValueError(f"std is of type {type(std)} which is incompatible for obj of type {type(obj)}")
    elif isinstance(obj, struct.Profile) :
        if isinstance(std, struct.Profile) :
            std_ = std.data
        elif isinstance(std, Number) :
            std_ = std * np.ones(obj.nz)
        else :
            raise ValueError(f"std is of type {type(std)} which is incompatible for obj of type {type(obj)}")
    else :
        raise ValueError(f"obj must be of type Cube, Map or Profile, not {type(obj)}")
    return std_

@overload
def additive_noise(obj : 'struct.Cube', noise_type : Literal['gaussian', 'uniform'],
    std : Union['struct.Cube', 'struct.Map', 'struct.Profile', float]) -> 'struct.Cube' : ...
@overload
def additive_noise(obj : 'struct.Map', noise_type : Literal['gaussian', 'uniform'],
    std : Union['struct.Map', float]) -> 'struct.Map' : ...
@overload
def additive_noise(obj : 'struct.Profile', noise_type : Literal['gaussian', 'uniform'],
    std : Union['struct.Profile', float]) -> 'struct.Profile' : ...

def additive_noise(obj : 'struct.Struct', noise_type : Literal['gaussian', 'uniform'],
                   std : Union['struct.Struct', float] = 1.) -> 'struct.Struct' :
    """
    Return the input structure degraded with an additive noise of type `noise_type`.
    
    Parameters
    ----------
    obj : Cube | Map | Profile
        Input structure.
    noise_type : str
        Type of noise. Must be 'gaussian' or 'uniform'.
    std : Cube | Map | Profile
        Standard deviation of noise.

    Returns
    -------

    res : Cube | Map | Profile
        Noisy structure.
    """
    noise_type = noise_type.lower().strip()
    std_ = _std_broadcasting(obj, std)

    if noise_type == 'gaussian' :
        data = obj.data + np.random.normal(0, std_)
    elif noise_type == 'uniform' :
        data = obj.data + np.random.uniform(-sqrt(3)*std_, sqrt(3)*std_)
    else :
        raise ValueError(f"noise_type must be gaussian or uniform, not {noise_type}")
    
    return type(obj)(data, obj.header)

@overload
def multiplicative_noise(obj : 'struct.Cube', noise_type : Literal['gaussian', 'uniform'],
    std : Union['struct.Cube', 'struct.Map', 'struct.Profile', float]) -> 'struct.Cube' : ...
@overload
def multiplicative_noise(obj : 'struct.Map', noise_type : Literal['gaussian', 'uniform'],
    std : Union['struct.Map', float]) -> 'struct.Map' : ...
@overload
def multiplicative_noise(obj : 'struct.Profile', noise_type : Literal['gaussian', 'uniform'],
    std : Union['struct.Profile', float]) -> 'struct.Profile' : ...

def multiplicative_noise(obj : 'struct.Struct', noise_type : Literal['gaussian', 'uniform'],
    std : Union['struct.Struct', float] = 1.) -> 'struct.Struct' :
    """
    Return the input structure degraded with a multiplicative noise of type `noise_type`.
    
    Parameters
    ----------
    obj : Cube | Map | Profile
        Input structure.
    noise_type : str
        Type of noise. Must be 'gaussian' or 'uniform'.
    std : Cube | Map | Profile
        Standard deviation of noise.

    Returns
    -------

    res : Cube | Map | Profile
        Noisy structure.
    """
    noise_type = noise_type.lower().strip()
    std_ = _std_broadcasting(obj, std)

    if noise_type == 'gaussian' :
        data = obj.data * (1 + np.random.normal(0, std_))
    elif noise_type == 'uniform' :
        data = obj.data * (1 + np.random.uniform(-sqrt(3)*std_, sqrt(3)*std_))
    else :
        raise ValueError(f"noise_type must be gaussian or uniform, not {noise_type}")
    
    return type(obj)(data, obj.header)