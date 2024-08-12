from warnings import warn
from typing import Literal, Union, Optional, Tuple, overload
from numbers import Number

from .. import structures as struct

__all__ = [
    'normalize', 'scale', 'standardize', 'unnormalize', 'unscale', 'unstandardize'
]

def _axis_parsing(input : 'struct.Struct', axis : Literal['spectral', 'spatial', 'all']) :
    """ Check if data is constant and if axis is valid. """
    # Check if axis is valid
    axis = axis.lower()
    if axis == 'all' :
        output_type = float
    elif axis == 'spectral' :
        output_type = struct.Profile
    elif axis == 'spatial' :
        output_type = struct.Map
    else :
        raise ValueError("axis must be 'all', 'spectral' or 'spatial', not {axis}")

    # Return parameters type
    return output_type

@overload
def standardize(input : 'struct.Cube', axis : Literal['spatial'],
    training_subset : Optional[Union['struct.Cube', 'struct.Map', 'struct.Profile']] = None,
    alternative_mode : bool = False) -> Tuple['struct.Cube', 'struct.Map', 'struct.Map'] : ...
@overload
def standardize(input : 'struct.Cube', axis : Literal['spectral'],
    training_subset : Optional[Union['struct.Cube', 'struct.Map', 'struct.Profile']] = None,
    alternative_mode : bool = False) -> Tuple['struct.Cube', 'struct.Profile', 'struct.Profile'] : ...
@overload
def standardize(input : 'struct.Cube', axis : Literal['all'] = 'all',
    training_subset : Optional[Union['struct.Cube', 'struct.Map', 'struct.Profile']] = None,
    alternative_mode : bool = False) -> Tuple['struct.Cube', float, float] : ...
@overload
def standardize(input : 'struct.Map', axis : Literal['all'] = 'all',
    training_subset : Optional['struct.Map'] = None,
    alternative_mode : bool = False) -> Tuple['struct.Map', float, float] : ...
@overload
def standardize(input : 'struct.Profile', axis : Literal['all'] = 'all',
    training_subset : Optional['struct.Profile'] = None,
    alternative_mode : bool = False) -> Tuple['struct.Profile', float, float] : ...

def standardize(input : 'struct.Struct', axis : Literal['spectral', 'spatial', 'all'] = 'all',
    training_subset : Optional['struct.Struct'] = None,
    alternative_mode : bool = False) -> Tuple['struct.Struct', ...] :
    """
    If alternative_mode is False, return a modified version of `input` with mean 0 and standart deviation 1.
    If alternative_mode is True, return a modified version of `input` with mean 0 but same standart deviation.
    """
    output_type = _axis_parsing(input, axis)
    
    if training_subset is None :
        training_subset = input.ones_like()

    mu = training_subset.where(input, float('nan')).mean(output_type)
    sigma = training_subset.where(input, float('nan')).std(output_type)
    
    if alternative_mode and isinstance(sigma, Number) :
        if sigma == 0 :
            warn('Input data are constant and cannot be standardized')
            sigma_ = 1
        else :
            sigma_ = sigma
    elif alternative_mode :
        if (sigma == 0).any() :
            warn('Some features of input data are constant and cannot be standardized')
        sigma_ = (sigma == 0).where(1, sigma)

    if alternative_mode :
        out = input - mu
    else :
        out = (input - mu) / sigma_
    
    return out, mu, sigma

@overload
def unstandardize(input : 'struct.Cube', mu : Union['struct.Map', 'struct.Profile', float],
    sigma : Union['struct.Map', 'struct.Profile', float], alternative_mode : bool = False) -> 'struct.Cube' : ...
@overload
def unstandardize(input : 'struct.Map', mu : float,
    sigma : float, alternative_mode : bool = False) -> 'struct.Map' : ...
@overload
def unstandardize(input : 'struct.Profile', mu : float,
    sigma : float, alternative_mode : bool = False) -> 'struct.Profile' : ...

def unstandardize(input : 'struct.Struct', mu : Union['struct.Struct', float],
    sigma : Union['struct.Struct', float], alternative_mode : bool = False) -> 'struct.Struct' :
    """ TODO """
    if alternative_mode :
        return input + mu
    return sigma * input + mu

@overload
def normalize(input : 'struct.Cube', axis : Literal['spatial'],
    training_subset : Optional[Union['struct.Cube', 'struct.Map', 'struct.Profile']] = None,
    alternative_mode : bool = False) -> Tuple['struct.Cube', 'struct.Map', 'struct.Map'] : ...
@overload
def normalize(input : 'struct.Cube', axis : Literal['spectral'],
    training_subset : Optional[Union['struct.Cube', 'struct.Map', 'struct.Profile']] = None,
    alternative_mode : bool = False) -> Tuple['struct.Cube', 'struct.Profile', 'struct.Profile'] : ...
@overload
def normalize(input : 'struct.Cube', axis : Literal['all'] = 'all',
    training_subset : Optional[Union['struct.Cube', 'struct.Map', 'struct.Profile']] = None,
    alternative_mode : bool = False) -> Tuple['struct.Cube', float, float] : ...
@overload
def normalize(input : 'struct.Map', axis : Literal['all'] = 'all',
    training_subset : Optional['struct.Map'] = None,
    alternative_mode : bool = False) -> Tuple['struct.Map', float, float] : ...
@overload
def normalize(input : 'struct.Profile', axis : Literal['all'] = 'all',
    training_subset : Optional['struct.Profile'] = None,
    alternative_mode : bool = False) -> Tuple['struct.Profile', float, float] : ...

def normalize(input : 'struct.Struct', axis : Literal['spectral', 'spatial', 'all'] = 'all',
    training_subset : Optional['struct.Struct'] = None,
    alternative_mode : bool = False) -> Tuple[Union['struct.Struct', float], ...] :
    """
    If alternative_mode is False, return a modified version of `input` with every values between 0 and 1.
    If alternative_mode is True, return a modified version of `input` with every values between -1 and 1.
    """
    output_type = _axis_parsing(input, axis)
    
    if training_subset is None :
        training_subset = input.ones_like()

    a = training_subset.where(input, float('nan')).min(output_type)
    b = training_subset.where(input, float('nan')).max(output_type)

    if isinstance(a, Number) :
        if a == b :
            warn('Input data are constant and cannot be standardized')
            scale_ = 1
        else :
            scale_ = b - a
    else :
        if (a == b).any() :
            warn('Some features of input data are constant and cannot be standardized')
        scale_ = (a == b).where(1, b - a)

    if alternative_mode :
        out = 2 * (input - a) / scale_ - 1 + (a==b).where(1, 0)
    else :
        out = (input - a) / scale_
    
    return out, a, b

@overload
def unnormalize(input : 'struct.Cube', a : Union['struct.Map', 'struct.Profile', float],
    b : Union['struct.Map', 'struct.Profile', float]) -> 'struct.Cube': ...
@overload
def unnormalize(input : 'struct.Map', a : float, b : float) -> 'struct.Map' : ...
@overload
def unnormalize(input : 'struct.Profile', a : float, b : float) -> 'struct.Profile' : ...

def unnormalize(input : 'struct.Struct', a : Union['struct.Map', 'struct.Profile', float],
    b : Union['struct.Struct', float], alternative_mode : bool = False) -> 'struct.Struct' :
    """ TODO """
    return (b-a) * input + a

@overload
def scale(input : 'struct.Cube', axis : Literal['spatial'],
    training_subset : Optional[Union['struct.Cube', 'struct.Map', 'struct.Profile']] = None,
    alternative_mode : bool = False) -> Tuple['struct.Cube', 'struct.Map'] : ...
@overload
def scale(input : 'struct.Cube', axis : Literal['spectral'],
    training_subset : Optional[Union['struct.Cube', 'struct.Map', 'struct.Profile']] = None,
    alternative_mode : bool = False) -> Tuple['struct.Cube', 'struct.Profile'] : ...
@overload
def scale(input : 'struct.Cube', axis : Literal['all'] = 'all',
    training_subset : Optional[Union['struct.Cube', 'struct.Map', 'struct.Profile']] = None,
    alternative_mode : bool = False) -> Tuple['struct.Cube', float] : ...
@overload
def scale(input : 'struct.Map', axis : Literal['all'] = 'all',
    training_subset : Optional['struct.Map'] = None,
    alternative_mode : bool = False) -> Tuple['struct.Map', float] : ...
@overload
def scale(input : 'struct.Profile', axis : Literal['all'] = 'all',
    training_subset : Optional['struct.Map'] = None,
    alternative_mode : bool = False) -> Tuple['struct.Profile', float] : ...

def scale(input : 'struct.Struct', axis : Literal['spectral', 'spatial', 'all'] = 'all',
    training_subset : Optional['struct.Struct'] = None,
    alternative_mode : bool = False) -> Tuple[Union['struct.Map', 'struct.Profile', float], ...] :
    """
    If alternative_mode is False, return a modified version of `input` where every value is between -1 and 1
    If alternative_mode is True, raise a NotImplementedError
    """
    output_type = _axis_parsing(input, axis)
    
    if training_subset is None :
        training_subset = input.ones_like()

    s = training_subset.where(input, float('nan')).abs().max(output_type)

    if isinstance(s, Number) :
        if s == 0 :
            warn('Input data are constant and cannot be standardized')
            s_ = 1
        else :
            s_ = s
    else :
        if (s == 0).any() :
            warn('Some features of input data are constant and cannot be standardized')
        s_ = (s == 0).where(1, s)

    if alternative_mode :
        raise NotImplementedError()
    else :
        pass
    out = input / s

    return out, s

@overload
def unscale(input : 'struct.Cube', s : Union['struct.Map', 'struct.Profile', float]) -> 'struct.Cube': ...
@overload
def unscale(input : 'struct.Map', s : float) -> 'struct.Map' : ...
@overload
def unscale(input : 'struct.Profile', s : float) -> 'struct.Profile' : ...

def unscale(input : 'struct.Struct', s : Union['struct.Struct', float], alternative_mode : bool = False) :
    """ TODO """
    if alternative_mode :
        raise NotImplementedError()
    else :
        return s * input