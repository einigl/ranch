from math import sqrt
from numbers import Number
from typing import Literal, Union, overload

import numpy as np

from .. import structures as struct

__all__ = ["additive_noise", "multiplicative_noise"]


def _std_broadcasting(
    input: "struct.Struct", std: Union["struct.Struct", float] = 1.0
) -> np.ndarray:
    std = std if std is not None else 1
    if isinstance(input, struct.Cube):
        if isinstance(std, struct.Cube):
            std_ = std.data
        elif isinstance(std, struct.Map):
            std_ = std.data[np.newaxis, :, :] * np.ones((input.nz, 1, 1))
        elif isinstance(std, struct.Profile):
            std_ = std.data[:, np.newaxis, np.newaxis] * np.ones(
                (1, input.ny, input.nx)
            )
        elif isinstance(std, Number):
            std_ = std * np.ones((input.nz, input.ny, input.nx))
        else:
            raise ValueError(
                f"std is of type {type(std)} which is incompatible for input of type {type(input)}"
            )
    elif isinstance(input, struct.Map):
        if isinstance(std, struct.Map):
            std_ = std.data
        elif isinstance(std, Number):
            std_ = std * np.ones((input.ny, input.nx))
        else:
            raise ValueError(
                f"std is of type {type(std)} which is incompatible for input of type {type(input)}"
            )
    elif isinstance(input, struct.Profile):
        if isinstance(std, struct.Profile):
            std_ = std.data
        elif isinstance(std, Number):
            std_ = std * np.ones(input.nz)
        else:
            raise ValueError(
                f"std is of type {type(std)} which is incompatible for input of type {type(input)}"
            )
    else:
        raise ValueError(
            f"input must be of type Cube, Map or Profile, not {type(input)}"
        )
    return std_


@overload
def additive_noise(
    input: "struct.Cube",
    noise_type: Literal["gaussian", "uniform"],
    std: Union["struct.Cube", "struct.Map", "struct.Profile", float],
) -> "struct.Cube":
    ...


@overload
def additive_noise(
    input: "struct.Map",
    noise_type: Literal["gaussian", "uniform"],
    std: Union["struct.Map", float],
) -> "struct.Map":
    ...


@overload
def additive_noise(
    input: "struct.Profile",
    noise_type: Literal["gaussian", "uniform"],
    std: Union["struct.Profile", float],
) -> "struct.Profile":
    ...


def additive_noise(
    input: "struct.Struct",
    noise_type: Literal["gaussian", "uniform"],
    std: Union["struct.Struct", float] = 1.0,
) -> "struct.Struct":
    """
    Return the input structure degraded with an additive noise of type `noise_type`.

    Parameters
    ----------
    input : Cube | Map | Profile
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
    std_ = _std_broadcasting(input, std)

    if noise_type == "gaussian":
        data = input.data + np.random.normal(0, std_)
    elif noise_type == "uniform":
        data = input.data + np.random.uniform(-sqrt(3) * std_, sqrt(3) * std_)
    else:
        raise ValueError(f"noise_type must be gaussian or uniform, not {noise_type}")

    return type(input)(data, input.header)


@overload
def multiplicative_noise(
    input: "struct.Cube",
    noise_type: Literal["gaussian", "uniform"],
    std: Union["struct.Cube", "struct.Map", "struct.Profile", float],
) -> "struct.Cube":
    ...


@overload
def multiplicative_noise(
    input: "struct.Map",
    noise_type: Literal["gaussian", "uniform"],
    std: Union["struct.Map", float],
) -> "struct.Map":
    ...


@overload
def multiplicative_noise(
    input: "struct.Profile",
    noise_type: Literal["gaussian", "uniform"],
    std: Union["struct.Profile", float],
) -> "struct.Profile":
    ...


def multiplicative_noise(
    input: "struct.Struct",
    noise_type: Literal["gaussian", "uniform"],
    std: Union["struct.Struct", float] = 1.0,
) -> "struct.Struct":
    """
    Return the input structure degraded with a multiplicative noise of type `noise_type`.

    Parameters
    ----------
    input : Cube | Map | Profile
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
    std_ = _std_broadcasting(input, std)

    if noise_type == "gaussian":
        data = input.data * (1 + np.random.normal(0, std_))
    elif noise_type == "uniform":
        data = input.data * (1 + np.random.uniform(-sqrt(3) * std_, sqrt(3) * std_))
    else:
        raise ValueError(f"noise_type must be gaussian or uniform, not {noise_type}")

    return type(input)(data, input.header)
