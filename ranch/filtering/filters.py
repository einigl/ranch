from math import comb
from typing import Callable, Union
from warnings import warn

import numpy as np
from scipy import ndimage

from .. import structures as struct

__all__ = [
    'Kernels', 'filter_channels', 'filter_cube', 'filter_map', 'filter_pixels', 'filter_profile'
]

class Kernels :
    """ Helpers to instantiate structuring elements """

    @staticmethod
    def uniform_1d(n : int) -> np.ndarray :
        """ TODO """
        return 1/n * np.ones(n)

    @staticmethod
    def gaussian_1d(n : int) -> np.ndarray :
        """ TODO """
        return 1/2**(n-1) * np.array([comb(n-1, k) for k in range(n)])

    @staticmethod
    def assemble_separables(*h) -> np.ndarray :
        """ TODO """
        H = h[0]
        for i, hi in enumerate(h[1:], 1) :
            H = np.expand_dims(H, -1) * np.expand_dims(hi, tuple(range(i)))
        return H


def _is_binary_filter(h : np.ndarray) -> bool :
    """ Returns True if h is a binary filter i.e. h contains only 0 and 1, else returns False """
    return ((h == 0) | (h == 1)).all()

def filter_pixels(cube : 'struct.Cube', kernel : np.ndarray, filtering_mode : str = 'linear',
    padding_mode : str = 'constant', cval : float = 0) :
    """ TODO """
    if kernel.ndim != 1 :
        raise ValueError(f"kernel must have 1 dimension, not {kernel.ndim}")
    if filtering_mode not in ['lin', 'linear', 'med', 'median', 'min', 'minimum', 'max', 'maximum'] :
        raise ValueError(f"filtering_mode must be 'linear', 'median', 'min' or 'max', not {filtering_mode}")
    if filtering_mode not in ['lin', 'linear'] and not _is_binary_filter(kernel) :
        warn('kernel is not a binary array. Every non zero element are considered True.')

    kernel = np.expand_dims(kernel, (1,2))
    kwargs = {'mode': padding_mode, 'cval': cval}

    if filtering_mode in ['lin', 'linear'] :
        new_data = ndimage.filters.convolve(cube.data, kernel, **kwargs)
    elif filtering_mode in ['med', 'median'] :
        new_data = ndimage.filters.median_filter(cube.data, footprint = kernel, **kwargs)
    elif filtering_mode in ['min', 'minimum'] :
        new_data = ndimage.filters.minimum_filter(cube.data, footprint = kernel, **kwargs)
    elif filtering_mode in ['max', 'maximum'] :
        new_data = ndimage.filters.maximum_filter(cube.data, footprint = kernel, **kwargs)
    else :
        raise ValueError('Should never been here')
        
    return struct.Cube(new_data, cube.header)

def filter_channels(cube : 'struct.Cube', kernel : np.ndarray, filtering_mode : str = 'linear',
    padding_mode : str = 'constant', cval : float = 0) :
    """ TODO """
    if kernel.ndim != 2 :
        raise ValueError(f"kernel must have 2 dimensions, not {kernel.ndim}")
    if filtering_mode not in ['lin', 'linear', 'med', 'median', 'min', 'minimum', 'max', 'maximum'] :
        raise ValueError(f"filtering_mode must be 'linear', 'median', 'min' or 'max', not {filtering_mode}")
    if filtering_mode not in ['lin', 'linear'] and not _is_binary_filter(kernel) :
        warn('kernel is not a binary array. Every non zero element are considered True.')

    kernel = np.expand_dims(kernel, 0)
    kwargs = {'mode': padding_mode, 'cval': cval}

    print(filtering_mode)

    if filtering_mode in ['lin', 'linear'] :
        new_data = ndimage.filters.convolve(cube.data, kernel, **kwargs)
    elif filtering_mode in ['med', 'median'] :
        new_data = ndimage.filters.median_filter(cube.data, footprint = kernel, **kwargs)
    elif filtering_mode in ['min', 'minimum'] :
        new_data = ndimage.filters.minimum_filter(cube.data, footprint = kernel, **kwargs)
    elif filtering_mode in ['max', 'maximum'] :
        new_data = ndimage.filters.maximum_filter(cube.data, footprint = kernel, **kwargs)
    else :
        raise ValueError('Should never been here')

    return struct.Cube(new_data, cube.header)

def filter_cube(cube : 'struct.Cube', kernel : np.ndarray, filtering_mode : str = 'linear',
    padding_mode : str = 'constant', cval : float = 0) :
    """ TODO """
    if kernel.ndim != 3 :
        raise ValueError(f"kernel must have 3 dimensions, not {kernel.ndim}")
    if filtering_mode not in ['lin', 'linear', 'med', 'median', 'min', 'minimum', 'max', 'maximum'] :
        raise ValueError(f"filtering_mode must be 'linear', 'median', 'min' or 'max', not {filtering_mode}")
    if filtering_mode not in ['lin', 'linear'] and not _is_binary_filter(kernel) :
        warn('kernel is not a binary array. Every non zero element are considered True.')

    kwargs = {'mode': padding_mode, 'cval': cval}

    if filtering_mode in ['lin', 'linear'] :
        new_data = ndimage.filters.convolve(cube.data, kernel, **kwargs)
    elif filtering_mode in ['med', 'median'] :
        new_data = ndimage.filters.median_filter(cube.data, footprint = kernel, **kwargs)
    elif filtering_mode in ['min', 'minimum'] :
        new_data = ndimage.filters.minimum_filter(cube.data, footprint = kernel, **kwargs)
    elif filtering_mode in ['max', 'maximum'] :
        new_data = ndimage.filters.maximum_filter(cube.data, footprint = kernel, **kwargs)
    else :
        raise ValueError('Should never been here')
    
    return struct.Cube(new_data, cube.header)

def filter_map(map : 'struct.Map', kernel : np.ndarray, filtering_mode : str = 'linear',
    padding_mode : str = 'constant', cval : float = 0) :
    """ TODO """
    if kernel.ndim != 2 :
        raise ValueError(f"kernel must have 2 dimensions, not {kernel.ndim}")
    if filtering_mode not in ['lin', 'linear', 'med', 'median', 'min', 'minimum', 'max', 'maximum'] :
        raise ValueError(f"filtering_mode must be 'linear', 'median', 'min' or 'max', not {filtering_mode}")
    if filtering_mode not in ['lin', 'linear'] and not _is_binary_filter(kernel) :
        warn('kernel is not a binary array. Every non zero element are considered True.')

    kwargs = {'mode': padding_mode, 'cval': cval}

    if filtering_mode in ['lin', 'linear'] :
        new_data = ndimage.filters.convolve(map.data, kernel, **kwargs)
    elif filtering_mode in ['med', 'median'] :
        new_data = ndimage.filters.median_filter(map.data, footprint = kernel, **kwargs)
    elif filtering_mode in ['min', 'minimum'] :
        new_data = ndimage.filters.minimum_filter(map.data, footprint = kernel, **kwargs)
    elif filtering_mode in ['max', 'maximum'] :
        new_data = ndimage.filters.maximum_filter(map.data, footprint = kernel, **kwargs)
    else :
        raise ValueError('Should never been here')
    
    return struct.Map(new_data, map.header)

def filter_profile(profile : 'struct.Profile', kernel : np.ndarray, filtering_mode : str = 'linear',
    padding_mode : str = 'constant', cval : float = 0) :
    """ TODO """
    if kernel.ndim != 1 :
        raise ValueError(f"kernel must have 1 dimension, not {kernel.ndim}")
    if filtering_mode not in ['lin', 'linear', 'med', 'median', 'min', 'minimum', 'max', 'maximum'] :
        raise ValueError(f"filtering_mode must be 'linear', 'median', 'min' or 'max', not {filtering_mode}")
    if filtering_mode not in ['lin', 'linear'] and not _is_binary_filter(kernel) :
        warn('kernel is not a binary array. Every non zero element are considered True.')

    kwargs = {'mode': padding_mode, 'cval': cval}

    if filtering_mode in ['lin', 'linear'] :
        new_data = ndimage.filters.convolve(profile.data, kernel, **kwargs)
    elif filtering_mode in ['med', 'median'] :
        new_data = ndimage.filters.median_filter(profile.data, footprint = kernel, **kwargs)
    elif filtering_mode in ['min', 'minimum'] :
        new_data = ndimage.filters.minimum_filter(profile.data, footprint = kernel, **kwargs)
    elif filtering_mode in ['max', 'maximum'] :
        new_data = ndimage.filters.maximum_filter(profile.data, footprint = kernel, **kwargs)
    else :
        raise ValueError('Should never been here')
    
    return struct.Profile(new_data, profile.header)
