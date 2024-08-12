from typing import Union, Sequence, Optional

import numpy as np

from .. import structures as struct
from ..core import header as hdr

__all__ = [
    'EXCLUDED', 'TRAIN', 'VAL', 'extract_indices', 'nan_mask', 'random_spatial_splitting', 'random_splitting', 'spatial_splitting', 'stack_indices'
]

TRAIN = 0
VAL = 1
EXCLUDED = -1

# NaNs detection

def nan_mask(obj : 'struct.Struct', item : str) -> 'struct.Struct' :
    """
    Returns a binary mask equal to 1 if an item contains at least a NaN, else 0.
    """
    if item.strip().lower() not in ['pixel', 'channel'] :
        raise ValueError(f"item must be 'pixel' or 'channel', not {item}")
    if isinstance(obj, struct.Cube) :
        if item == 'pixel' :
            new_data = np.isnan(obj.data).max(axis = 0)
            new_header = hdr.remove_header_axis(obj.header, axis = 'spectral')
            return struct.Map(new_data, new_header)
        else :
            new_data = np.isnan(obj.data).max(axis = (1,2))
            new_header = hdr.remove_header_axis(obj.header, axis = 'spatial')
            return struct.Map(new_data, new_header)
    elif isinstance(obj, struct.Map) :
        if item == 'pixel' :
            new_data = np.isnan(obj.data)
            return struct.Map(new_data, obj.header)
        else :
            raise ValueError("item = 'channel' is not compatible with a Map object")
    elif isinstance(obj, struct.Profile) :
        if item == 'pixel' :
            raise ValueError("item = 'pixel' is not compatible with a Profile object")
        else :
            new_data = np.isnan(obj.data)
            return struct.Profile(new_data, obj.header)
    else :
        raise TypeError(f"obj must be an instance of Cube, Map or Profile, not {type(obj)}")

def _common_nan_mask(objs : Union['struct.Struct', Sequence['struct.Struct']], item : str) -> 'struct.Struct' :
    """
    Returns a binary mask equal to 1 if an item contains at least a NaN, else 0.
    """
    if not isinstance(objs, Sequence) :
        objs = [objs]
    nans = nan_mask(objs[0], item)
    for obj in objs[1:] :
        nans = nans | nan_mask(obj, item)
    return nans
    
# Splitting methods

def random_splitting(objs : Union['struct.Struct', Sequence['struct.Struct']], item : str,
    frac_train : float, seed : Optional[int] = None, reject_nans : bool = True) -> 'struct.Struct' :
    """
    Returns a ternary structure equal to 1 if an item is in the training set,
    0 if it is in the validation set and -1 if the item contains at least one NaN.
    The splitting structure is valid for every object in objs.
    """ 
    np.random.seed(seed)
    nans = _common_nan_mask(objs, item)
    if reject_nans :
        indices = np.arange(nans.data.size)[(~nans.data.astype('bool')).flatten()]
    else :
        indices = np.arange(nans.data.size)
    np.random.shuffle(indices)
    train_indices = indices[:int(frac_train*len(indices))]
    val_indices = indices[int(frac_train*len(indices)):]
    splitting_map = np.zeros_like(nans.data) + EXCLUDED
    if isinstance(nans, struct.Map) :
        cols = splitting_map.shape[1]
        splitting_map[train_indices // cols, train_indices % cols] = TRAIN
        splitting_map[val_indices // cols, val_indices % cols] = VAL
    elif isinstance(nans, struct.Profile) :
        splitting_map[train_indices] = TRAIN
        splitting_map[val_indices] = VAL
    else :
        raise RuntimeError('ERROR : should never been here')
    return type(nans)(splitting_map, nans.header)

def spatial_splitting(objs : Union['struct.Struct', Sequence['struct.Struct']], item : str) :
    """
    Returns a ternary structure equal to 1 if an item is in the training set,
    0 if it is in the validation set and -1 if the item contains at least one NaN
    """
    raise NotImplementedError()

def random_spatial_splitting(objs : Union['struct.Struct', Sequence['struct.Struct']], item : str) :
    """
    Returns a ternary structure equal to 1 if an item is in the training set,
    0 if it is in the validation set and -1 if the item contains at least one NaN
    """
    raise NotImplementedError()

# Indices extraction

def extract_indices(obj : Union['struct.Map', 'struct.Profile']) -> tuple[np.ndarray] :
    """
    Returns the indices of the training, validation and full sets.
    """
    if not isinstance(obj, (struct.Map, struct.Profile)) :
        raise TypeError(f"obj must be of type Map or Profile, not {type(obj)}")
    data = obj.data.flatten()
    indices = np.arange(data.size)
    train_indices = indices[data == TRAIN]
    val_indices = indices[data == VAL]
    return train_indices, val_indices, indices

def stack_indices(extracted : Sequence[tuple[np.ndarray]]) -> tuple[np.ndarray] :
    """
    Returns the stacked indices of the training, validation and full sets.
    """
    indices = np.array([])
    train_indices = np.array([])
    val_indices = np.array([])
    offset = 0
    for (train_ind, val_ind, ind) in extracted :
        offset += ind.max()
        indices = np.concatenate(indices, ind + offset)
        train_indices = np.concatenate(train_indices, train_ind + offset)
        val_indices = np.concatenate(val_indices, val_ind + offset)
    return train_indices, val_indices, indices