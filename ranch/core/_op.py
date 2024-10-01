from numbers import Integral, Number
from typing import Callable, Tuple, Union, overload
from warnings import warn

import numpy as np

from .. import structures as struct
from . import header as hdr


@overload
def _binary_op(
    a: "struct.Cube",
    b: Union["struct.Cube", "struct.Map", "struct.Profile", float],
    fun: Callable,
) -> "struct.Cube":
    ...


@overload
def _binary_op(
    a: "struct.Map", b: Union["struct.Map", float], fun: Callable
) -> "struct.Map":
    ...


@overload
def _binary_op(
    a: "struct.Map", b: Union["struct.Cube", "struct.Profile"], fun: Callable
) -> "struct.Cube":
    ...


@overload
def _binary_op(
    a: "struct.Profile", b: Union["struct.Profile", float], fun: Callable
) -> "struct.Profile":
    ...


@overload
def _binary_op(
    a: "struct.Profile", b: Union["struct.Cube", "struct.Map"], fun: Callable
) -> "struct.Cube":
    ...


def _binary_op(
    a: "struct.Struct", b: Union["struct.Struct", float], fun: Callable
) -> "struct.Struct":
    """
    Convenience function for binary operators.

    Parameters
    ----------
    a : struct.Struct
        First input.
    b : Union[&#39;struct.Struct&#39;, float]
        Second input.
    fun : Callable
        Implementation of operator to be applied on numpy arrays.

    Returns
    -------
    struct.Struct
        Output multidimensional data.
    """
    if isinstance(b, Number):
        new_data = fun(a.data, b)
        return type(a)(new_data, a.header)
    new_header = hdr.merge_headers(a.header, b.header)
    if type(a) == type(b):
        new_data = fun(a.data, b.data)
        return type(a)(new_data, new_header)
    if isinstance(a, struct.Map) and isinstance(b, struct.Profile):
        new_data = fun(np.expand_dims(a.data, 0), np.expand_dims(b.data, [1, 2]))
    elif isinstance(a, struct.Profile) and isinstance(b, struct.Map):
        new_data = fun(np.expand_dims(a.data, [1, 2]), np.expand_dims(b.data, 0))
    elif isinstance(a, struct.Cube) and isinstance(b, struct.Profile):
        new_data = fun(a.data, np.expand_dims(b.data, [1, 2]))
    elif isinstance(a, struct.Profile) and isinstance(b, struct.Cube):
        new_data = fun(np.expand_dims(a.data, [1, 2]), b.data)
    else:
        new_data = fun(a.data, b.data)
    return struct.Cube(new_data, new_header)


# Unary float operators


@overload
def __neg__(self: "struct.Cube") -> "struct.Cube":
    ...


@overload
def __neg__(self: "struct.Map") -> "struct.Map":
    ...


@overload
def __neg__(self: "struct.Profile") -> "struct.Profile":
    ...


def __neg__(self: "struct.Struct") -> "struct.Struct":
    """Returns -self"""
    return type(self)(-self.data, self.header)


@overload
def __abs__(self: "struct.Cube") -> "struct.Cube":
    ...


@overload
def __abs__(self: "struct.Map") -> "struct.Map":
    ...


@overload
def __abs__(self: "struct.Profile") -> "struct.Profile":
    ...


def __abs__(self: "struct.Struct"):
    """Returns abs(self)"""
    return type(self)(-np.abs(self.data), self.header)


@overload
def __round__(self: "struct.Cube") -> "struct.Cube":
    ...


@overload
def __round__(self: "struct.Map") -> "struct.Map":
    ...


@overload
def __round__(self: "struct.Profile") -> "struct.Profile":
    ...


def __round__(self: "struct.Struct"):
    """Returns round(self)"""
    return type(self)(self.data.round(), self.header)


@overload
def __floor__(self: "struct.Cube") -> "struct.Cube":
    ...


@overload
def __floor__(self: "struct.Map") -> "struct.Map":
    ...


@overload
def __floor__(self: "struct.Profile") -> "struct.Profile":
    ...


def __floor__(self: "struct.Struct"):
    """Returns floor(self)"""
    return type(self)(np.floor(self.data), self.header)


@overload
def __ceil__(self: "struct.Cube") -> "struct.Cube":
    ...


@overload
def __ceil__(self: "struct.Map") -> "struct.Map":
    ...


@overload
def __ceil__(self: "struct.Profile") -> "struct.Profile":
    ...


def __ceil__(self: "struct.Struct"):
    """Returns ceil(self)"""
    return type(self)(np.ceil(self.data), self.header)


# Binary float operators


@overload
def __add__(
    self: "struct.Cube",
    other: Union["struct.Cube", "struct.Map", "struct.Profile", float],
) -> "struct.Cube":
    ...


@overload
def __add__(self: "struct.Map", other: Union["struct.Map", float]) -> "struct.Map":
    ...


@overload
def __add__(
    self: "struct.Map", other: Union["struct.Cube", "struct.Profile"]
) -> "struct.Cube":
    ...


@overload
def __add__(
    self: "struct.Profile", other: Union["struct.Profile", float]
) -> "struct.Profile":
    ...


@overload
def __add__(
    self: "struct.Profile", other: Union["struct.Cube", "struct.Map"]
) -> "struct.Cube":
    ...


def __add__(
    self: "struct.Struct", other: Union["struct.Struct", float]
) -> "struct.Struct":
    """Returns self + other"""
    return _binary_op(self, other, getattr(np.ndarray, "__add__"))


@overload
def __radd__(self: "struct.Cube", other: float) -> "struct.Cube":
    ...


@overload
def __radd__(self: "struct.Map", other: float) -> "struct.Map":
    ...


@overload
def __radd__(self: "struct.Profile", other: float) -> "struct.Profile":
    ...


def __radd__(self: "struct.Struct", other) -> "struct.Struct":
    """Returns other + self"""
    return self.__add__(other)


@overload
def __sub__(
    self: "struct.Cube",
    other: Union["struct.Cube", "struct.Map", "struct.Profile", float],
) -> "struct.Cube":
    ...


@overload
def __sub__(self: "struct.Map", other: Union["struct.Map", float]) -> "struct.Map":
    ...


@overload
def __sub__(
    self: "struct.Map", other: Union["struct.Cube", "struct.Profile"]
) -> "struct.Cube":
    ...


@overload
def __sub__(
    self: "struct.Profile", other: Union["struct.Profile", float]
) -> "struct.Profile":
    ...


@overload
def __sub__(
    self: "struct.Profile", other: Union["struct.Cube", "struct.Map"]
) -> "struct.Cube":
    ...


def __sub__(self: "struct.Struct", other) -> "struct.Struct":
    """Returns self - other"""
    return _binary_op(self, other, getattr(np.ndarray, "__sub__"))


@overload
def __rsub__(self: "struct.Cube", other: float) -> "struct.Cube":
    ...


@overload
def __rsub__(self: "struct.Map", other: float) -> "struct.Map":
    ...


@overload
def __rsub__(self: "struct.Profile", other: float) -> "struct.Profile":
    ...


def __rsub__(self: "struct.Struct", other) -> "struct.Struct":
    """Returns other - self"""
    return self.__sub__(other).__neg__()


@overload
def __mul__(
    self: "struct.Cube",
    other: Union["struct.Cube", "struct.Map", "struct.Profile", float],
) -> "struct.Cube":
    ...


@overload
def __mul__(self: "struct.Map", other: Union["struct.Map", float]) -> "struct.Map":
    ...


@overload
def __mul__(
    self: "struct.Map", other: Union["struct.Cube", "struct.Profile"]
) -> "struct.Cube":
    ...


@overload
def __mul__(
    self: "struct.Profile", other: Union["struct.Profile", float]
) -> "struct.Profile":
    ...


@overload
def __mul__(
    self: "struct.Profile", other: Union["struct.Cube", "struct.Map"]
) -> "struct.Cube":
    ...


def __mul__(self: "struct.Struct", other) -> "struct.Struct":
    """Returns self * other"""
    return _binary_op(self, other, getattr(np.ndarray, "__mul__"))


@overload
def __rmul__(self: "struct.Cube", other: float) -> "struct.Cube":
    ...


@overload
def __rmul__(self: "struct.Map", other: float) -> "struct.Map":
    ...


@overload
def __rmul__(self: "struct.Profile", other: float) -> "struct.Profile":
    ...


def __rmul__(self: "struct.Struct", other) -> "struct.Struct":
    """Returns other * self"""
    return self.__mul__(other)


@overload
def __truediv__(
    self: "struct.Cube",
    other: Union["struct.Cube", "struct.Map", "struct.Profile", float],
) -> "struct.Cube":
    ...


@overload
def __truediv__(self: "struct.Map", other: Union["struct.Map", float]) -> "struct.Map":
    ...


@overload
def __truediv__(
    self: "struct.Map", other: Union["struct.Cube", "struct.Profile"]
) -> "struct.Cube":
    ...


@overload
def __truediv__(
    self: "struct.Profile", other: Union["struct.Profile", float]
) -> "struct.Profile":
    ...


@overload
def __truediv__(
    self: "struct.Profile", other: Union["struct.Cube", "struct.Map"]
) -> "struct.Cube":
    ...


def __truediv__(self: "struct.Struct", other) -> "struct.Struct":
    """Returns self / other"""
    return _binary_op(self, other, getattr(np.ndarray, "__truediv__"))


@overload
def __rtruediv__(self: "struct.Cube", other: float) -> "struct.Cube":
    ...


@overload
def __rtruediv__(self: "struct.Map", other: float) -> "struct.Map":
    ...


@overload
def __rtruediv__(self: "struct.Profile", other: float) -> "struct.Profile":
    ...


def __rtruediv__(self: "struct.Struct", other) -> "struct.Struct":
    """Returns other / self"""
    return _binary_op(self, other, getattr(np.ndarray, "__rtruediv__"))


@overload
def __floordiv__(
    self: "struct.Cube",
    other: Union["struct.Cube", "struct.Map", "struct.Profile", float],
) -> "struct.Cube":
    ...


@overload
def __floordiv__(self: "struct.Map", other: Union["struct.Map", float]) -> "struct.Map":
    ...


@overload
def __floordiv__(
    self: "struct.Map", other: Union["struct.Cube", "struct.Profile"]
) -> "struct.Cube":
    ...


@overload
def __floordiv__(
    self: "struct.Profile", other: Union["struct.Profile", float]
) -> "struct.Profile":
    ...


@overload
def __floordiv__(
    self: "struct.Profile", other: Union["struct.Cube", "struct.Map"]
) -> "struct.Cube":
    ...


def __floordiv__(self: "struct.Struct", other) -> "struct.Struct":
    """Returns self // other"""
    return _binary_op(self, other, getattr(np.ndarray, "__floordiv__"))


@overload
def __mod__(
    self: "struct.Cube",
    other: Union["struct.Cube", "struct.Map", "struct.Profile", float],
) -> "struct.Cube":
    ...


@overload
def __mod__(self: "struct.Map", other: Union["struct.Map", float]) -> "struct.Map":
    ...


@overload
def __mod__(
    self: "struct.Map", other: Union["struct.Cube", "struct.Profile"]
) -> "struct.Cube":
    ...


@overload
def __mod__(
    self: "struct.Profile", other: Union["struct.Profile", float]
) -> "struct.Profile":
    ...


@overload
def __mod__(
    self: "struct.Profile", other: Union["struct.Cube", "struct.Map"]
) -> "struct.Cube":
    ...


def __mod__(self: "struct.Struct", other) -> "struct.Struct":
    """Returns self % other"""
    return _binary_op(self, other, getattr(np.ndarray, "__mod__"))


@overload
def __pow__(
    self: "struct.Cube",
    other: Union["struct.Cube", "struct.Map", "struct.Profile", float],
) -> "struct.Cube":
    ...


@overload
def __pow__(self: "struct.Map", other: Union["struct.Map", float]) -> "struct.Map":
    ...


@overload
def __pow__(
    self: "struct.Map", other: Union["struct.Cube", "struct.Profile"]
) -> "struct.Cube":
    ...


@overload
def __pow__(
    self: "struct.Profile", other: Union["struct.Profile", float]
) -> "struct.Profile":
    ...


@overload
def __pow__(
    self: "struct.Profile", other: Union["struct.Cube", "struct.Map"]
) -> "struct.Cube":
    ...


def __pow__(self: "struct.Struct", other) -> "struct.Struct":
    """Returns self ** other"""
    return _binary_op(self, other, getattr(np.ndarray, "__pow__"))


@overload
def __rpow__(self: "struct.Cube", other: float) -> "struct.Cube":
    ...


@overload
def __rpow__(self: "struct.Map", other: float) -> "struct.Map":
    ...


@overload
def __rpow__(self: "struct.Profile", other: float) -> "struct.Profile":
    ...


def __rpow__(self: "struct.Struct", other) -> "struct.Struct":
    """Returns other ** self"""
    return _binary_op(self, other, getattr(np.ndarray, "__rpow__"))


# Unary boolean operators


@overload
def __invert__(self: "struct.Cube") -> "struct.Cube":
    ...


@overload
def __invert__(self: "struct.Map") -> "struct.Map":
    ...


@overload
def __invert__(self: "struct.Profile") -> "struct.Profile":
    ...


def __invert__(self: "struct.Struct"):
    """Returns ~self"""
    if not self.is_logical():
        raise TypeError(
            "Self must be a logical structure i.e. to contain only 0 and 1 samples."
        )
    return type(self)(1 - self.data.astype("int"), self.header)


# Binary boolean operators


@overload
def __or__(
    self: "struct.Cube",
    other: Union["struct.Cube", "struct.Map", "struct.Profile", float],
) -> "struct.Cube":
    ...


@overload
def __or__(self: "struct.Map", other: Union["struct.Map", float]) -> "struct.Map":
    ...


@overload
def __or__(
    self: "struct.Map", other: Union["struct.Cube", "struct.Profile"]
) -> "struct.Cube":
    ...


@overload
def __or__(
    self: "struct.Profile", other: Union["struct.Profile", float]
) -> "struct.Profile":
    ...


@overload
def __or__(
    self: "struct.Profile", other: Union["struct.Cube", "struct.Map"]
) -> "struct.Cube":
    ...


def __or__(self: "struct.Struct", other) -> "struct.Struct":
    """Returns self | other"""
    if not self.is_logical():
        raise TypeError(
            "Self must be a logical structure i.e. to contain only 0 and 1 samples."
        )
    if not self.is_logical():
        raise TypeError(
            "Other must be a logical structure i.e. to contain only 0 and 1 samples."
        )
    return _binary_op(self, other, getattr(np, "maximum"))


@overload
def __ror__(self: "struct.Cube", other: float) -> "struct.Cube":
    ...


@overload
def __ror__(self: "struct.Map", other: float) -> "struct.Map":
    ...


@overload
def __ror__(self: "struct.Profile", other: float) -> "struct.Profile":
    ...


def __ror__(self: "struct.Struct", other) -> "struct.Struct":
    """Returns other | self"""
    return self.__or__(other)


@overload
def __and__(
    self: "struct.Cube",
    other: Union["struct.Cube", "struct.Map", "struct.Profile", float],
) -> "struct.Cube":
    ...


@overload
def __and__(self: "struct.Map", other: Union["struct.Map", float]) -> "struct.Map":
    ...


@overload
def __and__(
    self: "struct.Map", other: Union["struct.Cube", "struct.Profile"]
) -> "struct.Cube":
    ...


@overload
def __and__(
    self: "struct.Profile", other: Union["struct.Profile", float]
) -> "struct.Profile":
    ...


@overload
def __and__(
    self: "struct.Profile", other: Union["struct.Cube", "struct.Map"]
) -> "struct.Cube":
    ...


def __and__(self: "struct.Struct", other) -> "struct.Struct":
    """Returns self & other"""
    if not self.is_logical():
        raise TypeError(
            "Self must be a logical structure i.e. to contain only 0 and 1 samples."
        )
    if not self.is_logical():
        raise TypeError(
            "Other must be a logical structure i.e. to contain only 0 and 1 samples."
        )
    return _binary_op(self, other, getattr(np, "minimum"))


@overload
def __rand__(self: "struct.Cube", other: float) -> "struct.Cube":
    ...


@overload
def __rand__(self: "struct.Map", other: float) -> "struct.Map":
    ...


@overload
def __rand__(self: "struct.Profile", other: float) -> "struct.Profile":
    ...


def __rand__(self: "struct.Struct", other) -> "struct.Struct":
    """Returns other & self"""
    return self.__and__(other)


@overload
def __xor__(
    self: "struct.Cube",
    other: Union["struct.Cube", "struct.Map", "struct.Profile", float],
) -> "struct.Cube":
    ...


@overload
def __xor__(self: "struct.Map", other: Union["struct.Map", float]) -> "struct.Map":
    ...


@overload
def __xor__(
    self: "struct.Map", other: Union["struct.Cube", "struct.Profile"]
) -> "struct.Cube":
    ...


@overload
def __xor__(
    self: "struct.Profile", other: Union["struct.Profile", float]
) -> "struct.Profile":
    ...


@overload
def __xor__(
    self: "struct.Profile", other: Union["struct.Cube", "struct.Map"]
) -> "struct.Cube":
    ...


def __xor__(self: "struct.Struct", other) -> "struct.Struct":
    """Returns self ^ other"""
    if not self.is_logical():
        raise TypeError(
            "Self must be a logical structure i.e. to contain only 0 and 1 samples."
        )
    if not self.is_logical():
        raise TypeError(
            "Other must be a logical structure i.e. to contain only 0 and 1 samples."
        )
    return _binary_op(self, other, getattr(np.ndarray, "__ne__"))


@overload
def __rxor__(self: "struct.Cube", other: float) -> "struct.Cube":
    ...


@overload
def __rxor__(self: "struct.Map", other: float) -> "struct.Map":
    ...


@overload
def __rxor__(self: "struct.Profile", other: float) -> "struct.Profile":
    ...


def __rxor__(self: "struct.Struct", other) -> "struct.Struct":
    """Returns other ^ self"""
    return self.__xor__(other)


# Comparison operators


@overload
def __eq__(
    self: "struct.Cube",
    other: Union["struct.Cube", "struct.Map", "struct.Profile", float],
) -> "struct.Cube":
    ...


@overload
def __eq__(self: "struct.Map", other: Union["struct.Map", float]) -> "struct.Map":
    ...


@overload
def __eq__(
    self: "struct.Map", other: Union["struct.Cube", "struct.Profile"]
) -> "struct.Cube":
    ...


@overload
def __eq__(
    self: "struct.Profile", other: Union["struct.Profile", float]
) -> "struct.Profile":
    ...


@overload
def __eq__(
    self: "struct.Profile", other: Union["struct.Cube", "struct.Map"]
) -> "struct.Cube":
    ...


def __eq__(self: "struct.Struct", other) -> "struct.Struct":
    """Returns self == other"""
    return _binary_op(self, other, getattr(np.ndarray, "__eq__"))


@overload
def __ne__(
    self: "struct.Cube",
    other: Union["struct.Cube", "struct.Map", "struct.Profile", float],
) -> "struct.Cube":
    ...


@overload
def __ne__(self: "struct.Map", other: Union["struct.Map", float]) -> "struct.Map":
    ...


@overload
def __ne__(
    self: "struct.Map", other: Union["struct.Cube", "struct.Profile"]
) -> "struct.Cube":
    ...


@overload
def __ne__(
    self: "struct.Profile", other: Union["struct.Profile", float]
) -> "struct.Profile":
    ...


@overload
def __ne__(
    self: "struct.Profile", other: Union["struct.Cube", "struct.Map"]
) -> "struct.Cube":
    ...


def __ne__(self: "struct.Struct", other) -> "struct.Struct":
    """Returns self != other"""
    return _binary_op(self, other, getattr(np.ndarray, "__ne__"))


@overload
def __ge__(
    self: "struct.Cube",
    other: Union["struct.Cube", "struct.Map", "struct.Profile", float],
) -> "struct.Cube":
    ...


@overload
def __ge__(self: "struct.Map", other: Union["struct.Map", float]) -> "struct.Map":
    ...


@overload
def __ge__(
    self: "struct.Map", other: Union["struct.Cube", "struct.Profile"]
) -> "struct.Cube":
    ...


@overload
def __ge__(
    self: "struct.Profile", other: Union["struct.Profile", float]
) -> "struct.Profile":
    ...


@overload
def __ge__(
    self: "struct.Profile", other: Union["struct.Cube", "struct.Map"]
) -> "struct.Cube":
    ...


def __ge__(self: "struct.Struct", other) -> "struct.Struct":
    """Returns self >= other"""
    return _binary_op(self, other, getattr(np.ndarray, "__ge__"))


@overload
def __gt__(
    self: "struct.Cube",
    other: Union["struct.Cube", "struct.Map", "struct.Profile", float],
) -> "struct.Cube":
    ...


@overload
def __gt__(self: "struct.Map", other: Union["struct.Map", float]) -> "struct.Map":
    ...


@overload
def __gt__(
    self: "struct.Map", other: Union["struct.Cube", "struct.Profile"]
) -> "struct.Cube":
    ...


@overload
def __gt__(
    self: "struct.Profile", other: Union["struct.Profile", float]
) -> "struct.Profile":
    ...


@overload
def __gt__(
    self: "struct.Profile", other: Union["struct.Cube", "struct.Map"]
) -> "struct.Cube":
    ...


def __gt__(self: "struct.Struct", other) -> "struct.Struct":
    """Returns self > other"""
    return _binary_op(self, other, getattr(np.ndarray, "__gt__"))


@overload
def __le__(
    self: "struct.Cube",
    other: Union["struct.Cube", "struct.Map", "struct.Profile", float],
) -> "struct.Cube":
    ...


@overload
def __le__(self: "struct.Map", other: Union["struct.Map", float]) -> "struct.Map":
    ...


@overload
def __le__(
    self: "struct.Map", other: Union["struct.Cube", "struct.Profile"]
) -> "struct.Cube":
    ...


@overload
def __le__(
    self: "struct.Profile", other: Union["struct.Profile", float]
) -> "struct.Profile":
    ...


@overload
def __le__(
    self: "struct.Profile", other: Union["struct.Cube", "struct.Map"]
) -> "struct.Cube":
    ...


def __le__(self: "struct.Struct", other) -> "struct.Struct":
    """Returns self <= other"""
    return _binary_op(self, other, getattr(np.ndarray, "__le__"))


@overload
def __lt__(
    self: "struct.Cube",
    other: Union["struct.Cube", "struct.Map", "struct.Profile", float],
) -> "struct.Cube":
    ...


@overload
def __lt__(self: "struct.Map", other: Union["struct.Map", float]) -> "struct.Map":
    ...


@overload
def __lt__(
    self: "struct.Map", other: Union["struct.Cube", "struct.Profile"]
) -> "struct.Cube":
    ...


@overload
def __lt__(
    self: "struct.Profile", other: Union["struct.Profile", float]
) -> "struct.Profile":
    ...


@overload
def __lt__(
    self: "struct.Profile", other: Union["struct.Cube", "struct.Map"]
) -> "struct.Cube":
    ...


def __lt__(
    self: "struct.Struct", other: Union["struct.Struct", float]
) -> "struct.Struct":
    """Returns self < other"""
    return _binary_op(self, other, getattr(np.ndarray, "__lt__"))


# Getitem operator


@overload
def __getitem__(
    self: "struct.Cube",
    index: Union[
        int, Tuple[int], np.ndarray, "struct.Cube", "struct.Map", "struct.Profile"
    ],
) -> np.ndarray:
    ...


@overload
def __getitem__(
    self: "struct.Map", index: Union[Tuple[int], np.ndarray, "struct.Map"]
) -> np.ndarray:
    ...


@overload
def __getitem__(
    self: "struct.Profile", index: Union[int, np.ndarray, "struct.Profile"]
) -> np.ndarray:
    ...


def __getitem__(
    self: "struct.Struct", index: Union[int, Tuple[int], np.ndarray, "struct.Struct"]
) -> np.ndarray:
    """Returns self[index]"""
    if isinstance(index, struct.Struct):
        if not index.is_logical():
            warn("index is not a logical structure but it is casted as it")
        index_ = index.data.astype(bool)

    if isinstance(self, struct.Cube):
        if isinstance(index, Integral):
            return self.data[index]  # 2D array
        if isinstance(index, tuple):
            return self.data[:, index[1], index[0]]  # 1D array
        if isinstance(index, np.ndarray):
            return self.data[
                index
            ]  # returned array shape depends on index shape and type
        if isinstance(index, struct.Cube):
            return self.data[index_]  # 1D array
        if isinstance(index, struct.Map):
            return self.data[:, index_]  # 2D array
        if isinstance(index, struct.Profile):
            return self.data[index_, :, :]  # 2D array
        raise ValueError(
            "index must be an instance of int, tuple, ndarray, Cube, Map or Profile to be a valid index of type Cube"
        )
    if isinstance(self, struct.Map):
        if isinstance(index, tuple):
            return self.data[index[1], index[0]]  # 0D array
        if isinstance(index, np.ndarray):
            return self.data[
                index
            ]  # returned array shape depends on index shape and type
        if isinstance(index, struct.Map):
            return self.data[index_]  # 1D array
        raise ValueError(
            "index must be an instance of tuple, ndarray or Map to be a valid index of type Map"
        )
    if isinstance(self, struct.Profile):
        if isinstance(index, Integral):
            return self.data[index]  # 0D array
        if isinstance(index, np.ndarray):
            return self.data[
                index
            ]  # returned array shape depends on index shape and type
        if isinstance(index, struct.Profile):
            return self.data[index_]  # 1D array
        raise ValueError(
            "index must be an instance of int, ndarray or Profile to be a valid index of type Profile"
        )
