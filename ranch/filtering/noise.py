from math import sqrt, comb
from typing import List, Optional
from warnings import warn

from scipy.special import erfinv

from .. import structures as struct
from . import Kernels

__all__ = [
    'noise_clipping'
]

def noise_clipping(cube : 'struct.Cube', percent : float,
    noise_map : Optional['struct.Map'] = None, signal_channels : Optional[List[slice]] = None,
    p_xy : int = 5, p_z : int = 3, type_xy : str = 'gaussian', type_z : str = 'gaussian',
    nans : bool = False, verbose : bool = False) :

    if noise_map is not None and signal_channels is not None :
        warn('noise_map and signal_channels are both not None so signal_channels will be ignored')
    elif noise_map is None and signal_channels is not None :
        noise_map = cube.noise_map(signal_channels)
    elif signal_channels is None and noise_map is not None :
        pass
    else :
        raise ValueError('noise_map and signal_channels must not be both None')

    # Spatial and spectral clipping
    if type_xy.lower() == 'gaussian' :
        h_xy = Kernels.gaussian_1d(p_xy)
        sigma_xy = 1/2**(p_xy-1)*sqrt(comb(2*(p_xy-1), p_xy-1))
    elif type_xy.lower() == 'uniform' :
        h_xy = Kernels.uniform_1d(p_xy)
        sigma_xy = sqrt(1/p_xy)

    if type_z.lower() == 'gaussian' :
        h_z = Kernels.gaussian_1d(p_z)
        sigma_z = 1/2**(p_z-1)*sqrt(comb(2*(p_z-1), p_z-1))
    elif type_z.lower() == 'uniform' :
        h_z = Kernels.uniform_1d(p_z)
        sigma_z = sqrt(1/p_z)
        
    h = Kernels.assemble_separables(h_z, h_xy, h_xy)
    
    sigma : float = sigma_xy**2 * sigma_z
    thresh : float = sqrt(2) * sigma * erfinv(2*percent/100-1)
    if verbose :
        print('Threshold : {:.2f}'.format(thresh))

    m = (cube/noise_map).filter(h)
    mask = m > thresh
    cube_clipped = mask.where(cube, 0. if not nans else float('nan'))

    return cube_clipped, mask
