from typing import Union, Sequence, Tuple, List

from .. import structures as struct
from ..core import header as hdr

__all__ = [
    'get_channel', 'get_channels', 'get_pixel', 'get_pixels'
]

def get_channel(cube : 'struct.Cube', z : Union[int, float],
    unit = 'index') -> 'struct.Map' :
    """ TODO """
    return cube.get_channels([z], unit = unit)[0]

def get_channels(cube : 'struct.Cube', z : Sequence[Union[int, float]],
    unit = 'index') -> List['struct.Map'] :
    """ TODO """
    unit = unit.lower()
    if unit == 'index' :
        k_ = z
    elif unit == 'velocity' :
        k_ = [hdr.velocity_to_index(cube.header, v) for v in z]
    elif unit == 'frequency' :
        k_ = [hdr.frequency_to_index(cube.header, f) for f in z]
    else :
        raise ValueError(f"unit must be 'index', 'velocity' or 'frequency', not '{unit}'")
    
    new_header = hdr.remove_header_axis(cube.header, 'spectral')
    out = []
    for k in k_ :
        new_data = cube.data[k]
        out.append(struct.Map(new_data, new_header))
    return out

def get_pixel(cube : 'struct.Cube', xy : Tuple[Union[int, float]],
    unit = 'index') -> 'struct.Profile' :
    """ TODO """
    return cube.get_pixels([xy], unit = unit)[0]

def get_pixels(cube : 'struct.Cube', xy : Sequence[Tuple[Union[int, float]]],
    unit = 'index') -> List['struct.Profile'] :
    """ TODO """
    unit = unit.lower()
    if unit == 'index' :
        ij_ = xy
    elif unit == 'angle' :
        ij_ = [hdr.coordinates_to_indices(cube.header, x, y) for (x,y) in xy ]
    else :
        raise ValueError(f"unit must be 'index' or 'angle', not '{unit}'")

    new_header = hdr.remove_header_axis(cube.header, 'spatial')
    out = []
    for (i,j) in ij_ :
        new_data = cube.data[:, j, i]
        out.append(struct.Profile(new_data, new_header))
    return out
