from typing import Optional, Sequence, Union

import numpy as np

from .. import structures as struct
from ..core import header as hdr

__all__ = [
    "EXCLUDED",
    "TRAIN",
    "VAL",
    "extract_indices",
    "nan_mask",
    "random_spatial_splitting",
    "random_splitting",
    "spatial_splitting",
    "stack_indices",
]

TRAIN = 0
VAL = 1
EXCLUDED = -1

# NaNs detection


def nan_mask(input: "struct.Struct", item: str) -> "struct.Struct":
    """
    Returns a binary mask equal to 1 if an item contains at least a NaN, else 0.
    """
    if item.strip().lower() not in ["pixel", "channel"]:
        raise ValueError(f"item must be 'pixel' or 'channel', not {item}")
    if isinstance(input, struct.Cube):
        if item == "pixel":
            new_data = np.isnan(input.data).max(axis=0)
            new_header = hdr.remove_header_axis(input.header, axis="spectral")
            return struct.Map(new_data, new_header)
        else:
            new_data = np.isnan(input.data).max(axis=(1, 2))
            new_header = hdr.remove_header_axis(input.header, axis="spatial")
            return struct.Map(new_data, new_header)
    elif isinstance(input, struct.Map):
        if item == "pixel":
            new_data = np.isnan(input.data)
            return struct.Map(new_data, input.header)
        else:
            raise ValueError("item = 'channel' is not compatible with a Map inputect")
    elif isinstance(input, struct.Profile):
        if item == "pixel":
            raise ValueError("item = 'pixel' is not compatible with a Profile inputect")
        else:
            new_data = np.isnan(input.data)
            return struct.Profile(new_data, input.header)
    else:
        raise TypeError(
            f"input must be an instance of Cube, Map or Profile, not {type(input)}"
        )


def _common_nan_mask(
    inputs: Union["struct.Struct", Sequence["struct.Struct"]], item: str
) -> "struct.Struct":
    """
    Returns a binary mask equal to 1 if an item contains at least a NaN, else 0.
    """
    if not isinstance(inputs, Sequence):
        inputs = [inputs]
    nans = nan_mask(inputs[0], item)
    for input in inputs[1:]:
        nans = nans | nan_mask(input, item)
    return nans


# Splitting methods


def random_splitting(
    inputs: Union["struct.Struct", Sequence["struct.Struct"]],
    item: str,
    frac_train: float,
    seed: Optional[int] = None,
    reject_nans: bool = True,
) -> "struct.Struct":
    """
    Returns a ternary structure equal to 1 if an item is in the training set,
    0 if it is in the validation set and -1 if the item contains at least one NaN.
    The splitting structure is valid for every inputect in inputs.
    """
    np.random.seed(seed)
    nans = _common_nan_mask(inputs, item)
    if reject_nans:
        indices = np.arange(nans.data.size)[(~nans.data.astype("bool")).flatten()]
    else:
        indices = np.arange(nans.data.size)
    np.random.shuffle(indices)
    train_indices = indices[: int(frac_train * len(indices))]
    val_indices = indices[int(frac_train * len(indices)) :]
    splitting_map = np.zeros_like(nans.data) + EXCLUDED
    if isinstance(nans, struct.Map):
        cols = splitting_map.shape[1]
        splitting_map[train_indices // cols, train_indices % cols] = TRAIN
        splitting_map[val_indices // cols, val_indices % cols] = VAL
    elif isinstance(nans, struct.Profile):
        splitting_map[train_indices] = TRAIN
        splitting_map[val_indices] = VAL
    else:
        raise RuntimeError("ERROR : should never been here")
    return type(nans)(splitting_map, nans.header)


def spatial_splitting(
    inputs: Union["struct.Struct", Sequence["struct.Struct"]], item: str
):
    """
    Returns a ternary structure equal to 1 if an item is in the training set,
    0 if it is in the validation set and -1 if the item contains at least one NaN
    """
    raise NotImplementedError()


def random_spatial_splitting(
    inputs: Union["struct.Struct", Sequence["struct.Struct"]], item: str
):
    """
    Returns a ternary structure equal to 1 if an item is in the training set,
    0 if it is in the validation set and -1 if the item contains at least one NaN
    """
    raise NotImplementedError()


# Indices extraction


def extract_indices(input: Union["struct.Map", "struct.Profile"]) -> tuple[np.ndarray]:
    """
    Returns the indices of the training, validation and full sets.
    """
    if not isinstance(input, (struct.Map, struct.Profile)):
        raise TypeError(f"input must be of type Map or Profile, not {type(input)}")
    data = input.data.flatten()
    indices = np.arange(data.size)
    train_indices = indices[data == TRAIN]
    val_indices = indices[data == VAL]
    return train_indices, val_indices, indices


def stack_indices(extracted: Sequence[tuple[np.ndarray]]) -> tuple[np.ndarray]:
    """
    Returns the stacked indices of the training, validation and full sets.
    """
    indices = np.array([])
    train_indices = np.array([])
    val_indices = np.array([])
    offset = 0
    for (train_ind, val_ind, ind) in extracted:
        offset += ind.max()
        indices = np.concatenate(indices, ind + offset)
        train_indices = np.concatenate(train_indices, train_ind + offset)
        val_indices = np.concatenate(val_indices, val_ind + offset)
    return train_indices, val_indices, indices
