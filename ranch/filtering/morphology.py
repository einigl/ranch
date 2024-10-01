from typing import overload

import numpy as np
from scipy import ndimage

from .. import structures as struct
from . import filters

__all__ = [
    "StructuringElements",
    "closing",
    "dilation",
    "erosion",
    "filters",
    "gradient",
    "laplacian",
    "opening",
]


class StructuringElements:
    """Helpers to instantiate structuring elements"""

    @staticmethod
    def segment(lg: int, filled: bool = True) -> np.ndarray:
        """TODO"""
        se = np.ones(lg, dtype=np.bool)
        if not filled:
            se[1:-1] = False
        return se

    @staticmethod
    def disk(r: int, filled: bool = True) -> np.ndarray:
        """TODO"""
        x = np.arange(2 * r + 1) - r
        X, Y = np.meshgrid(x, x)
        if filled:
            return (X**2 + Y**2) <= r**2
        return (X**2 + Y**2) == r**2

    @staticmethod
    def ellipse(a: int, b: int, theta: float, filled: bool = True) -> np.ndarray:
        """TODO"""
        raise NotImplementedError()

    @staticmethod
    def square(c: int, filled: bool = True) -> np.ndarray:
        """TODO"""
        return StructuringElements.rectangle(c, c)

    @staticmethod
    def rectangle(a: int, b: int, filled: bool = True) -> np.ndarray:
        """TODO"""
        se = np.ones((a, b), dtype=np.bool)
        if filled:
            se[1:-1, 1:-1] = False
        return se

    @staticmethod
    def ball(r: int, filled: bool = True) -> np.ndarray:
        """TODO"""
        x = np.arange(2 * r + 1) - r
        X, Y, Z = np.meshgrid(x, x, x)
        if filled:
            return (X**2 + Y**2 + Z**2) <= r**2
        return (X**2 + Y**2 + Z**2) == r**2

    @staticmethod
    def ellipsoid(
        rx: int,
        ry: int,
        rz: int,
        theta: float = 0.0,
        phi: float = 0.0,
        filled: bool = True,
    ) -> np.ndarray:  # TODO
        """TODO"""
        r = max(rx, ry, rz)
        x = np.arange(2 * r + 1) - r
        Z, Y, X = np.meshgrid(x, x, x, indexing="ij")
        if filled:
            return (X**2 / rx**2 + Y**2 / ry**2 + Z**2 / rz**2) <= 1
        return (X**2 / rx**2 + Y**2 / ry**2 + Z**2 / rz**2) == 1

    @staticmethod
    def cube(c: int, filled=True) -> np.ndarray:
        """TODO"""
        return StructuringElements.cuboid(c, c, c, filled=filled)

    @staticmethod
    def cuboid(a: int, b: int, c: int, filled: bool = True) -> np.ndarray:
        """TODO"""
        se = np.ones(a, b, c, dtype=np.bool)
        if not filled:
            se[1:-1, 1:-1, 1:-1] = False
        return se


@overload
def dilation(input: "struct.Cube", se: np.ndarray) -> "struct.Cube":
    ...


@overload
def dilation(input: "struct.Map", se: np.ndarray) -> "struct.Map":
    ...


@overload
def dilation(input: "struct.Profile", se: np.ndarray) -> "struct.Profile":
    ...


def dilation(input: "struct.Struct", se: np.ndarray) -> "struct.Struct":
    """TODO"""
    if isinstance(input, struct.Cube):
        if se.ndim == 1:
            return input.filter_pixels(se, filtering_mode="max", padding_mode="reflect")
        if se.ndim == 2:
            return input.filter_channels(
                se, filtering_mode="max", padding_mode="reflect"
            )
        if se.ndim == 3:
            return input.filter(se, filtering_mode="max", padding_mode="reflect")
        raise ValueError(
            f"Structuring element se must have 1, 2 or 3 dimensions, not {se.ndim}, for input of type Cube"
        )
    elif isinstance(input, struct.Map):
        if se.ndim == 2:
            return input.filter(se, filtering_mode="max", padding_mode="reflect")
        raise ValueError(
            f"Structuring element se must have 2 dimensions, not {se.ndim}, for input of type Map"
        )
    elif isinstance(input, struct.Profile):
        if se.ndim == 1:
            return input.filter(se, filtering_mode="max", padding_mode="reflect")
        raise ValueError(
            f"Structuring element se must have 1 dimension, not {se.ndim}, for input of type Profile"
        )
    else:
        raise ValueError(
            f"input must an instance of Cube, Map or Profile, not {type(input)}"
        )


@overload
def erosion(input: "struct.Cube", se: np.ndarray) -> "struct.Cube":
    ...


@overload
def erosion(input: "struct.Map", se: np.ndarray) -> "struct.Map":
    ...


@overload
def erosion(input: "struct.Profile", se: np.ndarray) -> "struct.Profile":
    ...


def erosion(input: "struct.Struct", se: np.ndarray) -> "struct.Struct":
    """TODO"""
    if isinstance(input, struct.Cube):
        if se.ndim == 1:
            return input.filter_pixels(se, filtering_mode="min", padding_mode="reflect")
        if se.ndim == 2:
            return input.filter_channels(
                se, filtering_mode="min", padding_mode="reflect"
            )
        if se.ndim == 3:
            return input.filter(se, "min")
        raise ValueError(
            f"Structuring element se must have 1, 2 or 3 dimensions, not {se.ndim}, for input of type Cube"
        )
    elif isinstance(input, struct.Map):
        if se.ndim == 2:
            return input.filter(se, filtering_mode="min", padding_mode="reflect")
        raise ValueError(
            f"Structuring element se must have 2 dimensions, not {se.ndim}, for input of type Map"
        )
    elif isinstance(input, struct.Profile):
        if se.ndim == 1:
            return input.filter(se, filtering_mode="min", padding_mode="reflect")
        raise ValueError(
            f"Structuring element se must have 1 dimension, not {se.ndim}, for input of type Profile"
        )
    else:
        raise ValueError(
            f"input must an instance of Cube, Map or Profile, not {type(input)}"
        )


@overload
def closing(input: "struct.Cube", se: np.ndarray) -> "struct.Cube":
    ...


@overload
def closing(input: "struct.Map", se: np.ndarray) -> "struct.Map":
    ...


@overload
def closing(input: "struct.Profile", se: np.ndarray) -> "struct.Profile":
    ...


def closing(input: "struct.Struct", se: np.ndarray) -> "struct.Struct":
    """TODO"""
    return erosion(dilation(input, se), se)


@overload
def opening(input: "struct.Cube", se: np.ndarray) -> "struct.Cube":
    ...


@overload
def opening(input: "struct.Map", se: np.ndarray) -> "struct.Map":
    ...


@overload
def opening(input: "struct.Profile", se: np.ndarray) -> "struct.Profile":
    ...


def opening(input: "struct.Struct", se: np.ndarray) -> "struct.Struct":
    """TODO"""
    return dilation(erosion(input, se), se)


@overload
def gradient(input: "struct.Cube", se: np.ndarray) -> "struct.Cube":
    ...


@overload
def gradient(input: "struct.Map", se: np.ndarray) -> "struct.Map":
    ...


@overload
def gradient(input: "struct.Profile", se: np.ndarray) -> "struct.Profile":
    ...


def gradient(input: "struct.Struct", se: np.ndarray) -> "struct.Struct":
    """TODO"""
    return dilation(input, se) - erosion(input, se)


@overload
def laplacian(input: "struct.Cube", se: np.ndarray) -> "struct.Cube":
    ...


@overload
def laplacian(input: "struct.Map", se: np.ndarray) -> "struct.Map":
    ...


@overload
def laplacian(input: "struct.Profile", se: np.ndarray) -> "struct.Profile":
    ...


def laplacian(input: "struct.Struct", se: np.ndarray) -> "struct.Struct":
    """TODO"""
    return dilation(input, se) + erosion(input, se) - 2 * input
