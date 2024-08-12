from typing import Optional, overload

import numpy as np

from .. import structures as struct

__all__ = [
    'abs', 'arccos', 'arccosh', 'arcsin', 'arcsinh', 'arctan', 'arctanh', 'cbrt', 'cos', 'cosh', 'exp', 'log', 'sin', 'sinh', 'sqrt', 'struct', 'tan', 'tanh'
]

@overload
def abs(obj : 'struct.Cube') -> 'struct.Cube' : ...
@overload
def abs(obj : 'struct.Map') -> 'struct.Map' : ...
@overload
def abs(obj : 'struct.Profile') -> 'struct.Profile' : ...

def abs(obj : 'struct.Struct') -> 'struct.Struct' :
    """
    Element-wise absolute value operator.
    
    Parameters
    ----------
    obj : `Cube | Map | Profile`
        Input structure.

    Returns
    -------
    out : `Cube | Map | Profile`
        Output structure.
    """
    return type(obj)(np.abs(obj.data), obj.header)

@overload
def sqrt(obj : 'struct.Cube') -> 'struct.Cube' : ...
@overload
def sqrt(obj : 'struct.Map') -> 'struct.Map' : ...
@overload
def sqrt(obj : 'struct.Profile') -> 'struct.Profile' : ...

def sqrt(obj : 'struct.Struct') -> 'struct.Struct' :
    """
    Element-wise square root operator.
    
    Parameters
    ----------
    obj : `Cube | Map | Profile`
        Input structure.

    Returns
    -------
    out : `Cube | Map | Profile`
        Output structure.
    """
    return type(obj)(np.sqrt(obj.data), obj.header)

@overload
def cbrt(obj : 'struct.Cube') -> 'struct.Cube' : ...
@overload
def cbrt(obj : 'struct.Map') -> 'struct.Map' : ...
@overload
def cbrt(obj : 'struct.Profile') -> 'struct.Profile' : ...

def cbrt(obj : 'struct.Struct') -> 'struct.Struct' :
    """
    Element-wise cube root operator.
    
    Parameters
    ----------
    obj : `Cube | Map | Profile`
        Input structure.

    Returns
    -------
    out : `Cube | Map | Profile`
        Output structure.
    """
    return type(obj)(np.cbrt(obj.data), obj.header)

@overload
def exp(obj : 'struct.Cube') -> 'struct.Cube' : ...
@overload
def exp(obj : 'struct.Map') -> 'struct.Map' : ...
@overload
def exp(obj : 'struct.Profile') -> 'struct.Profile' : ...

def exp(obj : 'struct.Struct') :
    """
    Element-wise exponential operator.
    
    Parameters
    ----------
    obj : `Cube | Map | Profile`
        Input structure.

    Returns
    -------
    out : `Cube | Map | Profile`
        Output structure.
    """
    return type(obj)(np.exp(obj.data), obj.header)

@overload
def log(obj : 'struct.Cube', base : Optional[float] = None) -> 'struct.Cube' : ...
@overload
def log(obj : 'struct.Map', base : Optional[float] = None) -> 'struct.Map' : ...
@overload
def log(obj : 'struct.Profile', base : Optional[float] = None) -> 'struct.Profile' : ...

def log(obj : 'struct.Struct', base : Optional[float] = None) :
    """
    Element-wise logarithm operator.

    Parameters
    ----------
    obj : `Cube | Map | Profile`
        Input structure.
    base : `float | None`, optional
        Base of the logarithm (by default natural logarithm). Must a positive number.

    Returns
    -------
    out : `Cube | Map | Profile`
        Output structure.
    """
    if base is None :
        return type(obj)(np.log(obj.data), obj.header)
    return type(obj)(np.log(obj.data) / np.log(base), obj.header)

@overload
def cos(obj : 'struct.Cube') -> 'struct.Cube' : ...
@overload
def cos(obj : 'struct.Map') -> 'struct.Map' : ...
@overload
def cos(obj : 'struct.Profile') -> 'struct.Profile' : ...

def cos(obj : 'struct.Struct') :
    """
    Element-wise cosine operator.
    
    Parameters
    ----------
    obj : `Cube | Map | Profile`
        Input structure.

    Returns
    -------
    out : `Cube | Map | Profile`
        Output structure.
    """
    return type(obj)(np.cos(obj.data), obj.header)

@overload
def sin(obj : 'struct.Cube') -> 'struct.Cube' : ...
@overload
def sin(obj : 'struct.Map') -> 'struct.Map' : ...
@overload
def sin(obj : 'struct.Profile') -> 'struct.Profile' : ...

def sin(obj : 'struct.Struct') :
    """
    Element-wise sine operator.
    
    Parameters
    ----------
    obj : `Cube | Map | Profile`
        Input structure.

    Returns
    -------
    out : `Cube | Map | Profile`
        Output structure.
    """
    return type(obj)(np.sin(obj.data), obj.header)

@overload
def tan(obj : 'struct.Cube') -> 'struct.Cube' : ...
@overload
def tan(obj : 'struct.Map') -> 'struct.Map' : ...
@overload
def tan(obj : 'struct.Profile') -> 'struct.Profile' : ...

def tan(obj : 'struct.Struct') :
    """
    Element-wise tangent operator.
    
    Parameters
    ----------
    obj : `Cube | Map | Profile`
        Input structure.

    Returns
    -------
    out : `Cube | Map | Profile`
        Output structure.
    """
    return type(obj)(np.tan(obj.data), obj.header)

@overload
def arccos(obj : 'struct.Cube') -> 'struct.Cube' : ...
@overload
def arccos(obj : 'struct.Map') -> 'struct.Map' : ...
@overload
def arccos(obj : 'struct.Profile') -> 'struct.Profile' : ...

def arccos(obj : 'struct.Struct') :
    """
    Element-wise inverse cosine operator.
    
    Parameters
    ----------
    obj : `Cube | Map | Profile`
        Input structure.

    Returns
    -------
    out : `Cube | Map | Profile`
        Output structure.
    """
    return type(obj)(np.arccos(obj.data), obj.header)

@overload
def arcsin(obj : 'struct.Cube') -> 'struct.Cube' : ...
@overload
def arcsin(obj : 'struct.Map') -> 'struct.Map' : ...
@overload
def arcsin(obj : 'struct.Profile') -> 'struct.Profile' : ...

def arcsin(obj : 'struct.Struct') :
    """
    Element-wise inverse sine operator.
    
    Parameters
    ----------
    obj : `Cube | Map | Profile`
        Input structure.

    Returns
    -------
    out : `Cube | Map | Profile`
        Output structure.
    """
    return type(obj)(np.arcsin(obj.data), obj.header)

@overload
def arctan(obj : 'struct.Cube') -> 'struct.Cube' : ...
@overload
def arctan(obj : 'struct.Map') -> 'struct.Map' : ...
@overload
def arctan(obj : 'struct.Profile') -> 'struct.Profile' : ...

def arctan(obj : 'struct.Struct') :
    """
    Element-wise inverse tangent operator.
    
    Parameters
    ----------
    obj : `Cube | Map | Profile`
        Input structure.

    Returns
    -------
    out : `Cube | Map | Profile`
        Output structure.
    """
    return type(obj)(np.arctan(obj.data), obj.header)

@overload
def cosh(obj : 'struct.Cube') -> 'struct.Cube' : ...
@overload
def cosh(obj : 'struct.Map') -> 'struct.Map' : ...
@overload
def cosh(obj : 'struct.Profile') -> 'struct.Profile' : ...

def cosh(obj : 'struct.Struct') :
    """
    Element-wise hyperbolic cosine operator.
    
    Parameters
    ----------
    obj : `Cube | Map | Profile`
        Input structure.

    Returns
    -------
    out : `Cube | Map | Profile`
        Output structure.
    """
    return type(obj)(np.cosh(obj.data), obj.header)

@overload
def sinh(obj : 'struct.Cube') -> 'struct.Cube' : ...
@overload
def sinh(obj : 'struct.Map') -> 'struct.Map' : ...
@overload
def sinh(obj : 'struct.Profile') -> 'struct.Profile' : ...

def sinh(obj : 'struct.Struct') :
    """
    Element-wise hyperbolic sine operator.
    
    Parameters
    ----------
    obj : `Cube | Map | Profile`
        Input structure.

    Returns
    -------
    out : `Cube | Map | Profile`
        Output structure.
    """
    return type(obj)(np.sinh(obj.data), obj.header)

@overload
def tanh(obj : 'struct.Cube') -> 'struct.Cube' : ...
@overload
def tanh(obj : 'struct.Map') -> 'struct.Map' : ...
@overload
def tanh(obj : 'struct.Profile') -> 'struct.Profile' : ...

def tanh(obj : 'struct.Struct') :
    """
    Element-wise hyperbolic tangent operator.
    
    Parameters
    ----------
    obj : `Cube | Map | Profile`
        Input structure.

    Returns
    -------
    out : `Cube | Map | Profile`
        Output structure.
    """
    return type(obj)(np.tanh(obj.data), obj.header)

@overload
def arccosh(obj : 'struct.Cube') -> 'struct.Cube' : ...
@overload
def arccosh(obj : 'struct.Map') -> 'struct.Map' : ...
@overload
def arccosh(obj : 'struct.Profile') -> 'struct.Profile' : ...

def arccosh(obj : 'struct.Struct') :
    """
    Element-wise inverse hyperbolic cosine operator.
    
    Parameters
    ----------
    obj : `Cube | Map | Profile`
        Input structure.

    Returns
    -------
    out : `Cube | Map | Profile`
        Output structure.
    """
    return type(obj)(np.arccosh(obj.data), obj.header)

@overload
def arcsinh(obj : 'struct.Cube') -> 'struct.Cube' : ...
@overload
def arcsinh(obj : 'struct.Map') -> 'struct.Map' : ...
@overload
def arcsinh(obj : 'struct.Profile') -> 'struct.Profile' : ...

def arcsinh(obj : 'struct.Struct') :
    """
    Element-wise inverse hyperbolic sine operator.
    
    Parameters
    ----------
    obj : `Cube | Map | Profile`
        Input structure.

    Returns
    -------
    out : `Cube | Map | Profile`
        Output structure.
    """
    return type(obj)(np.arcsinh(obj.data), obj.header)

@overload
def arctanh(obj : 'struct.Cube') -> 'struct.Cube' : ...
@overload
def arctanh(obj : 'struct.Map') -> 'struct.Map' : ...
@overload
def arctanh(obj : 'struct.Profile') -> 'struct.Profile' : ...

def arctanh(obj : 'struct.Struct') :
    """
    Element-wise inverse hyperbolic tangent operator.
    
    Parameters
    ----------
    obj : `Cube | Map | Profile`
        Input structure.

    Returns
    -------
    out : `Cube | Map | Profile`
        Output structure.
    """
    return type(obj)(np.arctanh(obj.data), obj.header)
