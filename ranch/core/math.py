from typing import Optional, overload

import numpy as np

from .. import structures as struct

__all__ = [
    "abs",
    "arccos",
    "arccosh",
    "arcsin",
    "arcsinh",
    "arctan",
    "arctanh",
    "cbrt",
    "cos",
    "cosh",
    "exp",
    "log",
    "sin",
    "sinh",
    "sqrt",
    "struct",
    "tan",
    "tanh",
]


@overload
def abs(input: "struct.Cube") -> "struct.Cube":
    ...


@overload
def abs(input: "struct.Map") -> "struct.Map":
    ...


@overload
def abs(input: "struct.Profile") -> "struct.Profile":
    ...


def abs(input: "struct.Struct") -> "struct.Struct":
    """
    Element-wise absolute value operator.

    Parameters
    ----------
    input : `Cube | Map | Profile`
        Input structure.

    Returns
    -------
    out : `Cube | Map | Profile`
        Output structure.
    """
    return type(input)(np.abs(input.data), input.header)


@overload
def sqrt(input: "struct.Cube") -> "struct.Cube":
    ...


@overload
def sqrt(input: "struct.Map") -> "struct.Map":
    ...


@overload
def sqrt(input: "struct.Profile") -> "struct.Profile":
    ...


def sqrt(input: "struct.Struct") -> "struct.Struct":
    """
    Element-wise square root operator.

    Parameters
    ----------
    input : `Cube | Map | Profile`
        Input structure.

    Returns
    -------
    out : `Cube | Map | Profile`
        Output structure.
    """
    return type(input)(np.sqrt(input.data), input.header)


@overload
def cbrt(input: "struct.Cube") -> "struct.Cube":
    ...


@overload
def cbrt(input: "struct.Map") -> "struct.Map":
    ...


@overload
def cbrt(input: "struct.Profile") -> "struct.Profile":
    ...


def cbrt(input: "struct.Struct") -> "struct.Struct":
    """
    Element-wise cube root operator.

    Parameters
    ----------
    input : `Cube | Map | Profile`
        Input structure.

    Returns
    -------
    out : `Cube | Map | Profile`
        Output structure.
    """
    return type(input)(np.cbrt(input.data), input.header)


@overload
def exp(input: "struct.Cube") -> "struct.Cube":
    ...


@overload
def exp(input: "struct.Map") -> "struct.Map":
    ...


@overload
def exp(input: "struct.Profile") -> "struct.Profile":
    ...


def exp(input: "struct.Struct"):
    """
    Element-wise exponential operator.

    Parameters
    ----------
    input : `Cube | Map | Profile`
        Input structure.

    Returns
    -------
    out : `Cube | Map | Profile`
        Output structure.
    """
    return type(input)(np.exp(input.data), input.header)


@overload
def log(input: "struct.Cube", base: Optional[float] = None) -> "struct.Cube":
    ...


@overload
def log(input: "struct.Map", base: Optional[float] = None) -> "struct.Map":
    ...


@overload
def log(input: "struct.Profile", base: Optional[float] = None) -> "struct.Profile":
    ...


def log(input: "struct.Struct", base: Optional[float] = None):
    """
    Element-wise logarithm operator.

    Parameters
    ----------
    input : `Cube | Map | Profile`
        Input structure.
    base : `float | None`, optional
        Base of the logarithm (by default natural logarithm). Must a positive number.

    Returns
    -------
    out : `Cube | Map | Profile`
        Output structure.
    """
    if base is None:
        return type(input)(np.log(input.data), input.header)
    return type(input)(np.log(input.data) / np.log(base), input.header)


@overload
def cos(input: "struct.Cube") -> "struct.Cube":
    ...


@overload
def cos(input: "struct.Map") -> "struct.Map":
    ...


@overload
def cos(input: "struct.Profile") -> "struct.Profile":
    ...


def cos(input: "struct.Struct"):
    """
    Element-wise cosine operator.

    Parameters
    ----------
    input : `Cube | Map | Profile`
        Input structure.

    Returns
    -------
    out : `Cube | Map | Profile`
        Output structure.
    """
    return type(input)(np.cos(input.data), input.header)


@overload
def sin(input: "struct.Cube") -> "struct.Cube":
    ...


@overload
def sin(input: "struct.Map") -> "struct.Map":
    ...


@overload
def sin(input: "struct.Profile") -> "struct.Profile":
    ...


def sin(input: "struct.Struct"):
    """
    Element-wise sine operator.

    Parameters
    ----------
    input : `Cube | Map | Profile`
        Input structure.

    Returns
    -------
    out : `Cube | Map | Profile`
        Output structure.
    """
    return type(input)(np.sin(input.data), input.header)


@overload
def tan(input: "struct.Cube") -> "struct.Cube":
    ...


@overload
def tan(input: "struct.Map") -> "struct.Map":
    ...


@overload
def tan(input: "struct.Profile") -> "struct.Profile":
    ...


def tan(input: "struct.Struct"):
    """
    Element-wise tangent operator.

    Parameters
    ----------
    input : `Cube | Map | Profile`
        Input structure.

    Returns
    -------
    out : `Cube | Map | Profile`
        Output structure.
    """
    return type(input)(np.tan(input.data), input.header)


@overload
def arccos(input: "struct.Cube") -> "struct.Cube":
    ...


@overload
def arccos(input: "struct.Map") -> "struct.Map":
    ...


@overload
def arccos(input: "struct.Profile") -> "struct.Profile":
    ...


def arccos(input: "struct.Struct"):
    """
    Element-wise inverse cosine operator.

    Parameters
    ----------
    input : `Cube | Map | Profile`
        Input structure.

    Returns
    -------
    out : `Cube | Map | Profile`
        Output structure.
    """
    return type(input)(np.arccos(input.data), input.header)


@overload
def arcsin(input: "struct.Cube") -> "struct.Cube":
    ...


@overload
def arcsin(input: "struct.Map") -> "struct.Map":
    ...


@overload
def arcsin(input: "struct.Profile") -> "struct.Profile":
    ...


def arcsin(input: "struct.Struct"):
    """
    Element-wise inverse sine operator.

    Parameters
    ----------
    input : `Cube | Map | Profile`
        Input structure.

    Returns
    -------
    out : `Cube | Map | Profile`
        Output structure.
    """
    return type(input)(np.arcsin(input.data), input.header)


@overload
def arctan(input: "struct.Cube") -> "struct.Cube":
    ...


@overload
def arctan(input: "struct.Map") -> "struct.Map":
    ...


@overload
def arctan(input: "struct.Profile") -> "struct.Profile":
    ...


def arctan(input: "struct.Struct"):
    """
    Element-wise inverse tangent operator.

    Parameters
    ----------
    input : `Cube | Map | Profile`
        Input structure.

    Returns
    -------
    out : `Cube | Map | Profile`
        Output structure.
    """
    return type(input)(np.arctan(input.data), input.header)


@overload
def cosh(input: "struct.Cube") -> "struct.Cube":
    ...


@overload
def cosh(input: "struct.Map") -> "struct.Map":
    ...


@overload
def cosh(input: "struct.Profile") -> "struct.Profile":
    ...


def cosh(input: "struct.Struct"):
    """
    Element-wise hyperbolic cosine operator.

    Parameters
    ----------
    input : `Cube | Map | Profile`
        Input structure.

    Returns
    -------
    out : `Cube | Map | Profile`
        Output structure.
    """
    return type(input)(np.cosh(input.data), input.header)


@overload
def sinh(input: "struct.Cube") -> "struct.Cube":
    ...


@overload
def sinh(input: "struct.Map") -> "struct.Map":
    ...


@overload
def sinh(input: "struct.Profile") -> "struct.Profile":
    ...


def sinh(input: "struct.Struct"):
    """
    Element-wise hyperbolic sine operator.

    Parameters
    ----------
    input : `Cube | Map | Profile`
        Input structure.

    Returns
    -------
    out : `Cube | Map | Profile`
        Output structure.
    """
    return type(input)(np.sinh(input.data), input.header)


@overload
def tanh(input: "struct.Cube") -> "struct.Cube":
    ...


@overload
def tanh(input: "struct.Map") -> "struct.Map":
    ...


@overload
def tanh(input: "struct.Profile") -> "struct.Profile":
    ...


def tanh(input: "struct.Struct"):
    """
    Element-wise hyperbolic tangent operator.

    Parameters
    ----------
    input : `Cube | Map | Profile`
        Input structure.

    Returns
    -------
    out : `Cube | Map | Profile`
        Output structure.
    """
    return type(input)(np.tanh(input.data), input.header)


@overload
def arccosh(input: "struct.Cube") -> "struct.Cube":
    ...


@overload
def arccosh(input: "struct.Map") -> "struct.Map":
    ...


@overload
def arccosh(input: "struct.Profile") -> "struct.Profile":
    ...


def arccosh(input: "struct.Struct"):
    """
    Element-wise inverse hyperbolic cosine operator.

    Parameters
    ----------
    input : `Cube | Map | Profile`
        Input structure.

    Returns
    -------
    out : `Cube | Map | Profile`
        Output structure.
    """
    return type(input)(np.arccosh(input.data), input.header)


@overload
def arcsinh(input: "struct.Cube") -> "struct.Cube":
    ...


@overload
def arcsinh(input: "struct.Map") -> "struct.Map":
    ...


@overload
def arcsinh(input: "struct.Profile") -> "struct.Profile":
    ...


def arcsinh(input: "struct.Struct"):
    """
    Element-wise inverse hyperbolic sine operator.

    Parameters
    ----------
    input : `Cube | Map | Profile`
        Input structure.

    Returns
    -------
    out : `Cube | Map | Profile`
        Output structure.
    """
    return type(input)(np.arcsinh(input.data), input.header)


@overload
def arctanh(input: "struct.Cube") -> "struct.Cube":
    ...


@overload
def arctanh(input: "struct.Map") -> "struct.Map":
    ...


@overload
def arctanh(input: "struct.Profile") -> "struct.Profile":
    ...


def arctanh(input: "struct.Struct"):
    """
    Element-wise inverse hyperbolic tangent operator.

    Parameters
    ----------
    input : `Cube | Map | Profile`
        Input structure.

    Returns
    -------
    out : `Cube | Map | Profile`
        Output structure.
    """
    return type(input)(np.arctanh(input.data), input.header)
