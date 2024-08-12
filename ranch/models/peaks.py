from typing import Union, overload

from scipy.ndimage import maximum_filter1d

from .. import structures as struct
from ..core import header as hdr

__all__ = [
    'peaks_detection'
]

@overload
def peaks_detection(obj: 'struct.Cube', delta_peaks: int = 1,
    peaks_min: Union[float, 'struct.Struct'] = -float('inf')) -> 'struct.Cube': ...
@overload
def peaks_detection(obj: 'struct.Profile', delta_peaks: int = 1,
    peaks_min: Union[float, 'struct.Struct'] = -float('inf')) -> 'struct.Profile': ...

def peaks_detection(obj: Union['struct.Cube', 'struct.Profile'], delta_peaks: int = 1,
    peaks_min: Union[float, 'struct.Struct'] = -float('inf')) -> 'struct.Struct':
    """ axis must be 'spectral', 'spatial' or 'all' """

    peaks = type(obj).from_numpy(maximum_filter1d(obj.data, 2*delta_peaks+1, axis = 0), obj.header)
    peaks = (peaks == obj) & (obj > peaks_min)

    return peaks