from typing import Callable, Tuple, Type, Union, overload

import numpy as np

from .. import structures as struct
from ..core import header as hdr

__all__ = [
    "all",
    "any",
    "argmax",
    "argmin",
    "max",
    "mean",
    "median",
    "min",
    "moment",
    "percentile",
    "ptp",
    "quantile",
    "std",
    "sum",
    "var",
    "rms",
]


def _reduce(
    input: "struct.Struct",
    output_type: Type[Union["struct.Map", "struct.Profile", float]],
    operator: Callable,
) -> Union["struct.Map", "struct.Profile", float]:
    """
    Convenience function to implement reducers.

    Parameters
    ----------
    input: struct.Struct
        Input multidimensional data.
    output_type: Type[Map] | Type[Profile] | Type[float]
        Type of output data. Determine the axis over which the operator has to be applied.
    operator:
        Operation on numpy array that reduce the number of dimensions.

    Returns
    -------
    Map | Profile | float
        Resulting data of type `output_type`.
    """
    if not isinstance(input, struct.Cube) and output_type is not float:
        raise TypeError(
            f"input type {type(input)} is not compatible with output type {output_type}"
        )
    if output_type is float:
        return operator(input.data)
    if isinstance(input, struct.Cube) and output_type is struct.Map:
        new_header = hdr.remove_header_axis(input.header, axis="spectral")
        new_data = operator(input.data, axis=0)
    elif isinstance(input, struct.Cube) and output_type is struct.Profile:
        new_header = hdr.remove_header_axis(input.header, axis="spatial")
        new_data = operator(input.data, axis=(1, 2))
    else:
        raise ValueError("Should never been here")
    return output_type(new_data, new_header)


def _arg(
    input: "struct.Struct",
    output_type: Type[Union["struct.Map", "struct.Profile", float]],
    operator: Callable,
) -> Union["struct.Map", "struct.Profile", tuple, int]:
    """
    Convenience function to implement argmin and argmax functions.

    Parameters
    ----------
    input: struct.Struct
        Input multidimensional data.
    output_type: Type[Map] | Type[Profile] | Type[float]
        Type of output data. Determine the axis over which the operator has to be applied.
    operator:
        Operation on numpy array that reduce the number of dimensions.

    Returns
    -------
    Map | Profile | float
        Resulting data of type `output_type`.
    """
    if not isinstance(input, struct.Cube) and output_type is not float:
        raise TypeError(
            f"input type {type(input)} is not compatible with output type {output_type}"
        )
    if output_type is float:
        index = operator(input.data)
        indices = np.unravel_index(index, input.data.shape)[::-1]
        if isinstance(input, struct.Profile):
            return indices[0]  # z
        return indices  # (x,y) or (x,y,z)
    if isinstance(input, struct.Cube) and output_type is struct.Map:
        new_header = hdr.remove_header_axis(input.header, axis="spectral")
        indices = operator(input.data, axis=0)
        indices = np.unravel_index(index, input.shape)
    elif isinstance(input, struct.Cube) and output_type is struct.Profile:
        new_header = hdr.remove_header_axis(input.header, axis="spatial")
        indices = operator(input.data, axis=(1, 2))
    else:
        raise ValueError("Should never been here")
    return output_type(indices, new_header)


@overload
def all(
    input: "struct.Struct", output_type: Type[float] = float, ignore_nans: bool = True
) -> float:
    ...


@overload
def all(
    input: "struct.Struct",
    output_type: Type["struct.Map"] = float,
    ignore_nans: bool = True,
) -> "struct.Map":
    ...


@overload
def all(
    input: "struct.Struct",
    output_type: Type["struct.Profile"] = float,
    ignore_nans: bool = True,
) -> "struct.Profile":
    ...


def all(
    input: "struct.Struct",
    output_type: Type[Union["struct.Map", "struct.Profile", float]] = float,
) -> Union["struct.Map", "struct.Profile", float]:
    """
    Returns True if input.data contains only non-zero elements over the considered axis.

    Parameters
    ----------
    input: struct.Struct
        Input multidimensional data.
    output_type: Type[Map] | Type[Profile] | Type[float]
        Type of output data. Determine the axis over which the operator has to be applied.
    operator:
        Operation on numpy array that reduce the number of dimensions.

    Returns
    -------
    Map | Profile | float
        Resulting data of type `output_type`.
    """
    return _reduce(input, output_type, np.all)


@overload
def any(
    input: "struct.Struct", output_type: Type[float] = float, ignore_nans: bool = True
) -> float:
    ...


@overload
def any(
    input: "struct.Struct",
    output_type: Type["struct.Map"] = float,
    ignore_nans: bool = True,
) -> "struct.Map":
    ...


@overload
def any(
    input: "struct.Struct",
    output_type: Type["struct.Profile"] = float,
    ignore_nans: bool = True,
) -> "struct.Profile":
    ...


def any(
    input: "struct.Struct",
    output_type: Type[Union["struct.Map", "struct.Profile", float]] = float,
) -> bool:
    """
    Returns True if input.data contains at least one non-zero elements over the considered axis.

    Parameters
    ----------
    input: struct.Struct
        Input multidimensional data.
    output_type: Type[Map] | Type[Profile] | Type[float]
        Type of output data. Determine the axis over which the operator has to be applied.
    operator:
        Operation on numpy array that reduce the number of dimensions.

    Returns
    -------
    Map | Profile | float
        Resulting data of type `output_type`.
    """
    return _reduce(input, output_type, np.any)


@overload
def sum(
    input: "struct.Struct", output_type: Type[float] = float, ignore_nans: bool = True
) -> float:
    ...


@overload
def sum(
    input: "struct.Struct",
    output_type: Type["struct.Map"] = float,
    ignore_nans: bool = True,
) -> "struct.Map":
    ...


@overload
def sum(
    input: "struct.Struct",
    output_type: Type["struct.Profile"] = float,
    ignore_nans: bool = True,
) -> "struct.Profile":
    ...


def sum(
    input: "struct.Struct",
    output_type: Type[Union["struct.Map", "struct.Profile", float]] = float,
    ignore_nans: bool = True,
) -> Union["struct.Struct", float]:
    """
    Computes a sum over the needed axes to obtain a data of type `output_type`.

    Parameters
    ----------
    input: struct.Struct
        Input multidimensional data.
    output_type: Type[Map] | Type[Profile] | Type[float]
        Type of output data. Determine the axis over which the operator has to be applied.
    operator:
        Operation on numpy array that reduce the number of dimensions.

    Returns
    -------
    Map | Profile | float
        Resulting data of type `output_type`.
    """
    if ignore_nans:
        return _reduce(input, output_type, np.nansum)
    return _reduce(input, output_type, np.sum)


@overload
def min(
    input: "struct.Struct", output_type: Type[float] = float, ignore_nans: bool = True
) -> float:
    ...


@overload
def min(
    input: "struct.Struct",
    output_type: Type["struct.Map"] = float,
    ignore_nans: bool = True,
) -> "struct.Map":
    ...


@overload
def min(
    input: "struct.Struct",
    output_type: Type["struct.Profile"] = float,
    ignore_nans: bool = True,
) -> "struct.Profile":
    ...


def min(
    input: "struct.Struct",
    output_type: Type[Union["struct.Map", "struct.Profile", float]] = float,
    ignore_nans: bool = True,
) -> Union["struct.Struct", float]:
    """
    Computes the minimum value over the needed axes to obtain a data of type `output_type`.

    Parameters
    ----------
    input: struct.Struct
        Input multidimensional data.
    output_type: Type[Map] | Type[Profile] | Type[float]
        Type of output data. Determine the axis over which the operator has to be applied.
    operator:
        Operation on numpy array that reduce the number of dimensions.

    Returns
    -------
    Map | Profile | float
        Resulting data of type `output_type`.
    """
    if ignore_nans:
        return _reduce(input, output_type, np.nanmin)
    return _reduce(input, output_type, np.min)


@overload
def max(
    input: "struct.Struct", output_type: Type[float] = float, ignore_nans: bool = True
) -> float:
    ...


@overload
def max(
    input: "struct.Struct",
    output_type: Type["struct.Map"] = float,
    ignore_nans: bool = True,
) -> "struct.Map":
    ...


@overload
def max(
    input: "struct.Struct",
    output_type: Type["struct.Profile"] = float,
    ignore_nans: bool = True,
) -> "struct.Profile":
    ...


def max(
    input: "struct.Struct",
    output_type: Type[Union["struct.Map", "struct.Profile", float]] = float,
    ignore_nans: bool = True,
) -> Union["struct.Struct", float]:
    """
    Computes the maximum value over the needed axes to obtain a data of type `output_type`.

    Parameters
    ----------
    input: struct.Struct
        Input multidimensional data.
    output_type: Type[Map] | Type[Profile] | Type[float]
        Type of output data. Determine the axis over which the operator has to be applied.
    operator:
        Operation on numpy array that reduce the number of dimensions.

    Returns
    -------
    Map | Profile | float
        Resulting data of type `output_type`.
    """
    if ignore_nans:
        return _reduce(input, output_type, np.nanmax)
    return _reduce(input, output_type, np.max)


@overload
def ptp(
    input: "struct.Struct", output_type: Type[float] = float, ignore_nans: bool = True
) -> float:
    ...


@overload
def ptp(
    input: "struct.Struct",
    output_type: Type["struct.Map"] = float,
    ignore_nans: bool = True,
) -> "struct.Map":
    ...


@overload
def ptp(
    input: "struct.Struct",
    output_type: Type["struct.Profile"] = float,
    ignore_nans: bool = True,
) -> "struct.Profile":
    ...


def ptp(
    input: "struct.Struct",
    output_type: Type[Union["struct.Map", "struct.Profile", float]] = float,
    ignore_nans: bool = True,
) -> Union["struct.Struct", float]:
    """
    Computes the peak-to-peak value over the needed axes to obtain a data of type `output_type`.

    Parameters
    ----------
    input: struct.Struct
        Input multidimensional data.
    output_type: Type[Map] | Type[Profile] | Type[float]
        Type of output data. Determine the axis over which the operator has to be applied.
    operator:
        Operation on numpy array that reduce the number of dimensions.

    Returns
    -------
    Map | Profile | float
        Resulting data of type `output_type`.
    """
    return input.max(output_type, ignore_nans) - input.min(output_type, ignore_nans)


@overload
def argmin(
    input: "struct.Struct", output_type: Type[float] = float, ignore_nans: bool = True
) -> tuple:
    ...


@overload
def argmin(
    input: "struct.Struct",
    output_type: Type["struct.Map"] = float,
    ignore_nans: bool = True,
) -> "struct.Map":
    ...


@overload
def argmin(
    input: "struct.Struct",
    output_type: Type["struct.Profile"] = float,
    ignore_nans: bool = True,
) -> "struct.Profile":
    ...


def argmin(
    input: "struct.Struct",
    output_type: Type[Union["struct.Map", "struct.Profile", float]] = float,
    ignore_nans: bool = True,
) -> Union["struct.Struct", Tuple[int], int]:
    """
    Computes an argmin over the needed axes to obtain a data of type `output_type`.
    In case of multiple occurrences of the minimum values, the indices corresponding to the first occurrence are returned.

    Parameters
    ----------
    input: struct.Struct
        Input multidimensional data.
    output_type: Type[Map] | Type[Profile] | Type[float]
        Type of output data. Determine the axis over which the operator has to be applied.
    operator:
        Operation on numpy array that reduce the number of dimensions.

    Returns
    -------
    Map | Profile | float
        Resulting data of type `output_type`.
    """
    if ignore_nans:
        return _arg(input, output_type, np.nanargmin)
    return _arg(input, output_type, np.argmin)


@overload
def argmax(
    input: "struct.Struct", output_type: Type[float] = float, ignore_nans: bool = True
) -> float:
    ...


@overload
def argmax(
    input: "struct.Struct",
    output_type: Type["struct.Map"] = float,
    ignore_nans: bool = True,
) -> "struct.Map":
    ...


@overload
def argmax(
    input: "struct.Struct",
    output_type: Type["struct.Profile"] = float,
    ignore_nans: bool = True,
) -> "struct.Profile":
    ...


def argmax(
    input: "struct.Struct",
    output_type: Type[Union["struct.Map", "struct.Profile", float]] = float,
    ignore_nans: bool = True,
) -> Union["struct.Struct", Tuple[int], int]:
    """
    Computes an argmax value over the needed axes to obtain a data of type `output_type`.
    In case of multiple occurrences of the maximum values, the indices corresponding
    to the first occurrence are returned.

    Parameters
    ----------
    input: struct.Struct
        Input multidimensional data.
    output_type: Type[Map] | Type[Profile] | Type[float]
        Type of output data. Determine the axis over which the operator has to be applied.
    operator:
        Operation on numpy array that reduce the number of dimensions.

    Returns
    -------
    Map | Profile | float
        Resulting data of type `output_type`.
    """
    if ignore_nans:
        return _arg(input, output_type, np.nanargmax)
    return _arg(input, output_type, np.argmax)


@overload
def mean(
    input: "struct.Struct", output_type: Type[float] = float, ignore_nans: bool = True
) -> float:
    ...


@overload
def mean(
    input: "struct.Struct",
    output_type: Type["struct.Map"] = float,
    ignore_nans: bool = True,
) -> "struct.Map":
    ...


@overload
def mean(
    input: "struct.Struct",
    output_type: Type["struct.Profile"] = float,
    ignore_nans: bool = True,
) -> "struct.Profile":
    ...


def mean(
    input: "struct.Struct",
    output_type: Type[Union["struct.Map", "struct.Profile", float]] = float,
    ignore_nans: bool = True,
) -> Union["struct.Struct", float]:
    """
    Computes the mean value over the needed axes to obtain a data of type `output_type`.

    Parameters
    ----------
    input: struct.Struct
        Input multidimensional data.
    output_type: Type[Map] | Type[Profile] | Type[float]
        Type of output data. Determine the axis over which the operator has to be applied.
    operator:
        Operation on numpy array that reduce the number of dimensions.

    Returns
    -------
    Map | Profile | float
        Resulting data of type `output_type`.
    """
    if ignore_nans:
        return _reduce(input, output_type, np.nanmean)
    return _reduce(input, output_type, np.mean)


@overload
def std(
    input: "struct.Struct", output_type: Type[float] = float, ignore_nans: bool = True
) -> float:
    ...


@overload
def std(
    input: "struct.Struct",
    output_type: Type["struct.Map"] = float,
    ignore_nans: bool = True,
) -> "struct.Map":
    ...


@overload
def std(
    input: "struct.Struct",
    output_type: Type["struct.Profile"] = float,
    ignore_nans: bool = True,
) -> "struct.Profile":
    ...


def std(
    input: "struct.Struct",
    output_type: Type[Union["struct.Map", "struct.Profile", float]] = float,
    ignore_nans: bool = True,
) -> Union["struct.Struct", float]:
    """
    Computes the standard deviation over the needed axes to obtain a data of type `output_type`.

    Parameters
    ----------
    input: struct.Struct
        Input multidimensional data.
    output_type: Type[Map] | Type[Profile] | Type[float]
        Type of output data. Determine the axis over which the operator has to be applied.
    operator:
        Operation on numpy array that reduce the number of dimensions.

    Returns
    -------
    Map | Profile | float
        Resulting data of type `output_type`.
    """
    if ignore_nans:
        return _reduce(input, output_type, np.nanstd)
    return _reduce(input, output_type, np.std)


@overload
def var(
    input: "struct.Struct", output_type: Type[float] = float, ignore_nans: bool = True
) -> float:
    ...


@overload
def var(
    input: "struct.Struct",
    output_type: Type["struct.Map"] = float,
    ignore_nans: bool = True,
) -> "struct.Map":
    ...


@overload
def var(
    input: "struct.Struct",
    output_type: Type["struct.Profile"] = float,
    ignore_nans: bool = True,
) -> "struct.Profile":
    ...


def var(
    input: "struct.Struct",
    output_type: Type[Union["struct.Map", "struct.Profile", float]] = float,
    ignore_nans: bool = True,
) -> Union["struct.Struct", float]:
    """
    Computes the variance over the needed axes to obtain a data of type `output_type`.

    Parameters
    ----------
    input: struct.Struct
        Input multidimensional data.
    output_type: Type[Map] | Type[Profile] | Type[float]
        Type of output data. Determine the axis over which the operator has to be applied.
    operator:
        Operation on numpy array that reduce the number of dimensions.

    Returns
    -------
    Map | Profile | float
        Resulting data of type `output_type`.
    """
    if ignore_nans:
        return _reduce(input, output_type, np.nanvar)
    return _reduce(input, output_type, np.var)


@overload
def moment(
    input: "struct.Struct",
    order: int,
    centered: bool = True,
    output_type: Type[float] = float,
    ignore_nans: bool = True,
) -> float:
    ...


@overload
def moment(
    input: "struct.Struct",
    order: int,
    centered: bool = True,
    output_type: Type["struct.Map"] = float,
    ignore_nans: bool = True,
) -> "struct.Map":
    ...


@overload
def moment(
    input: "struct.Struct",
    order: int,
    centered: bool = True,
    output_type: Type["struct.Profile"] = float,
    ignore_nans: bool = True,
) -> "struct.Profile":
    ...


def moment(
    input: "struct.Struct",
    order: int,
    centered: bool = True,
    output_type: Type[Union["struct.Map", "struct.Profile", float]] = float,
    ignore_nans: bool = True,
) -> Union["struct.Struct", float]:
    """
    Computes the statistical moment of order `order` over the needed axes to obtain a data of type `output_type`. If `centered` is True, the centered moment is computed.

    Parameters
    ----------
    input: struct.Struct
        Input multidimensional data.
    order: int
        Order of the statistical moment.
    centered: bool
        Whether the centered moment is computed. Default: True.
    output_type: Type[Map] | Type[Profile] | Type[float]
        Type of output data. Determine the axis over which the operator has to be applied.
    operator:
        Operation on numpy array that reduce the number of dimensions.

    Returns
    -------
    Map | Profile | float
        Resulting data of type `output_type`.
    """
    if not isinstance(order, int) or order <= 0:
        raise ValueError(f"order must be a positive integer (here {order})")
    if ignore_nans:
        mean = np.nanmean
    else:
        mean = np.mean
    if centered and isinstance(input, struct.Cube) and output_type is struct.Profile:
        op = lambda a, axis=None: mean(
            (a - np.expand_dims(mean(a, axis=axis), (1, 2))) ** order, axis=axis
        )
    elif centered:
        op = lambda a, axis=None: mean((a - mean(a, axis=axis)) ** order, axis=axis)
    else:
        op = lambda a, axis=None: np.mean(a**order, axis)
    return _reduce(input, output_type, op)


@overload
def median(
    input: "struct.Struct", output_type: Type[float] = float, ignore_nans: bool = True
) -> float:
    ...


@overload
def median(
    input: "struct.Struct",
    output_type: Type["struct.Map"] = float,
    ignore_nans: bool = True,
) -> "struct.Map":
    ...


@overload
def median(
    input: "struct.Struct",
    output_type: Type["struct.Profile"] = float,
    ignore_nans: bool = True,
) -> "struct.Profile":
    ...


def median(
    input: "struct.Struct",
    output_type: Type[Union["struct.Map", "struct.Profile", float]] = float,
    ignore_nans: bool = True,
) -> Union["struct.Struct", float]:
    """
    Computes median value over the needed axes to obtain a data of type `output_type`.

    Parameters
    ----------
    input: struct.Struct
        Input multidimensional data.
    output_type: Type[Map] | Type[Profile] | Type[float]
        Type of output data. Determine the axis over which the operator has to be applied.
    operator:
        Operation on numpy array that reduce the number of dimensions.

    Returns
    -------
    Map | Profile | float
        Resulting data of type `output_type`.
    """
    if ignore_nans:
        return _reduce(input, output_type, np.nanmedian)
    return _reduce(input, output_type, np.median)


@overload
def quantile(
    input: "struct.Struct",
    q: float,
    output_type: Type[float] = float,
    ignore_nans: bool = True,
) -> float:
    ...


@overload
def quantile(
    input: "struct.Struct",
    q: float,
    output_type: Type["struct.Map"] = float,
    ignore_nans: bool = True,
) -> "struct.Map":
    ...


@overload
def quantile(
    input: "struct.Struct",
    q: float,
    output_type: Type["struct.Profile"] = float,
    ignore_nans: bool = True,
) -> "struct.Profile":
    ...


def quantile(
    input: "struct.Struct",
    q: float,
    output_type: Type[Union["struct.Map", "struct.Profile", float]] = float,
    ignore_nans: bool = True,
) -> Union["struct.Struct", float]:
    """
    Computes the quantile `q` over the needed axes to obtain a data of type `output_type`.

    Parameters
    ----------
    input: struct.Struct
        Input multidimensional data.
    q: float
        Quantile (between 0 and 1).
    output_type: Type[Map] | Type[Profile] | Type[float]
        Type of output data. Determine the axis over which the operator has to be applied.
    operator:
        Operation on numpy array that reduce the number of dimensions.

    Returns
    -------
    Map | Profile | float
        Resulting data of type `output_type`.
    """
    if ignore_nans:
        op = lambda a, axis=None: np.nanquantile(a, q, axis=axis)
    else:
        op = lambda a, axis=None: np.quantile(a, q, axis=axis)
    return _reduce(input, output_type, op)


@overload
def percentile(
    input: "struct.Struct",
    p: float,
    output_type: Type[float] = float,
    ignore_nans: bool = True,
) -> float:
    ...


@overload
def percentile(
    input: "struct.Struct",
    p: float,
    output_type: Type["struct.Map"] = float,
    ignore_nans: bool = True,
) -> "struct.Map":
    ...


@overload
def percentile(
    input: "struct.Struct",
    p: float,
    output_type: Type["struct.Profile"] = float,
    ignore_nans: bool = True,
) -> "struct.Profile":
    ...


def percentile(
    input: "struct.Struct",
    p: float,
    output_type: Type[Union["struct.Map", "struct.Profile", float]] = float,
    ignore_nans: bool = True,
) -> Union["struct.Struct", float]:
    """
    Computes the percentile `p` over the needed axes to obtain a data of type `output_type`.

    Parameters
    ----------
    input: struct.Struct
        Input multidimensional data.
    p: float
        Percentile (between 0 and 100).
    output_type: Type[Map] | Type[Profile] | Type[float]
        Type of output data. Determine the axis over which the operator has to be applied.
    operator:
        Operation on numpy array that reduce the number of dimensions.

    Returns
    -------
    Map | Profile | float
        Resulting data of type `output_type`.
    """
    if ignore_nans:
        op = lambda a, axis=None: np.nanpercentile(a, p, axis=axis)
    else:
        op = lambda a, axis=None: np.percentile(a, p, axis=axis)
    return _reduce(input, output_type, op)


@overload
def rms(
    input: "struct.Struct", output_type: Type[float] = float, ignore_nans: bool = True
) -> float:
    ...


@overload
def rms(
    input: "struct.Struct",
    output_type: Type["struct.Map"] = float,
    ignore_nans: bool = True,
) -> "struct.Map":
    ...


@overload
def rms(
    input: "struct.Struct",
    output_type: Type["struct.Profile"] = float,
    ignore_nans: bool = True,
) -> "struct.Profile":
    ...


def rms(
    input: "struct.Struct",
    output_type: Type[Union["struct.Map", "struct.Profile", float]] = float,
    ignore_nans: bool = True,
) -> Union["struct.Struct", float]:
    """
    Computes the root mean squared (RMS) value over the needed axes to obtain a data of type `output_type`.

    Parameters
    ----------
    input: struct.Struct
        Input multidimensional data.
    output_type: Type[Map] | Type[Profile] | Type[float]
        Type of output data. Determine the axis over which the operator has to be applied.
    operator:
        Operation on numpy array that reduce the number of dimensions.

    Returns
    -------
    Map | Profile | float
        Resulting data of type `output_type`.
    """
    op = np.nanmean if ignore_nans else np.mean
    return _reduce(
        input, output_type, lambda a, axis=None: np.sqrt(op(a**2, axis=axis))
    )
