""" Header module """

from typing import Optional, Union, Literal

from datetime import datetime

import numpy as np

from astropy.io import fits

__all__ = [
    'C_LIGHT', 'bound_coordinates', 'bound_frequencies', 'bound_velocities', 'change_coordinates', 'check_header', 'coordinates_to_indices', 'create_header', 'frequency_to_index', 'index_to_frequency', 'index_to_velocity', 'indices_to_coordinates', 'merge_headers', 'move_header_axes', 'remove_header_axis', 'update_header', 'velocity_to_index'
]

def check_header(data, header) :
    """ Returns True if data is compatible with the header, else raise a `ValueError` """
    if header['NAXIS'] != data.ndim :
        raise ValueError(f"header must have card NAXIS = {data.ndim}, not {header['NAXIS']}")
    h_shape = tuple((header[f'NAXIS{k}'] for k in range(data.ndim, 0, -1)))
    if data.shape != h_shape :
        raise ValueError(f"data dimensions {data.shape} do not match the header cards {h_shape}")
    return True

def update_header(data : np.ndarray, header : fits.Header) :
    """ Updates header cards to match data. Data and header are first verified by `check_header`
    i.e. their dimensions must match. Missing cards are also added to avoid future errors. """
    check_header(data, header)
    new_header = header.copy()
    # Update data extrema (if all values are NaNs, min and max are 0 by default)
    if np.isfinite(data).any() :
        new_header['DATAMIN'] = np.nanmin(data)
        new_header['DATAMAX'] = np.nanmax(data)
    else :
        new_header['DATAMIN'] = 0.
        new_header['DATAMAX'] = 0.
    # Update date
    new_header['DATE'] = str(datetime.now())
    # Update origin
    if 'ORIGIN' in new_header.keys() :
        if '- MODIFIED' not in new_header['ORIGIN'].strip().upper() :
            new_header['ORIGIN'] = new_header['ORIGIN'].strip().upper() + ' - MODIFIED'
    # Update several cards
    if 'OBJECT' not in new_header.keys() :
        new_header['OBJECT'] = 'UNKNOWN'
    if 'BUNIT' not in new_header.keys() :
        new_header['BUNIT'] = 'UNKNOWN'
    if 'RA' not in new_header.keys() :
        new_header['RA'] = 0.
    if 'DEC' not in new_header.keys() :
        new_header['DEC'] = 0.
    if 'RESTFREQ' not in new_header.keys() :
        new_header['RESTFREQ'] = 1.
    # Axis cards
    if data.ndim not in (1, 2, 3) :
        raise ValueError(f'data dimension must be 1, 2 or 3, not {data.ndim}')
    for axis in range(1, data.ndim+1) :
        if f'CTYPE{axis}' not in new_header.keys() :
            new_header[f'CTYPE{axis}'] = 'INDICES'
        if f'CRVAL{axis}' not in new_header.keys() :
            new_header[f'CRVAL{axis}'] = 1
        if f'CDELT{axis}' not in new_header.keys() :
            new_header[f'CDELT{axis}'] = 1
        if f'CRPIX{axis}' not in new_header.keys() :
            new_header[f'CRPIX{axis}'] = 1
        if f'CROTA{axis}' not in new_header.keys() :
            new_header[f'CROTA{axis}'] = 0
    return new_header

def merge_headers(header_1 : fits.Header, header_2 : fits.Header) :
    """
    Returns a header compatible with `header_1` and `header_2`.
    The returned header can have a different number of axis than the inputs.
    """
    n1, n2 = header_1['NAXIS'], header_2['NAXIS']
    cards = ['NAXIS{}', 'CTYPE{}', 'CRVAL{}', 'CDELT{}', 'CRPIX{}', 'CROTA{}']

    if n1 == n2 :
        # Case 1 : header have the same number of axis
        h_shape_1 = tuple((header_1[f'NAXIS{k}'] for k in range(header_1['NAXIS'], 0, -1)))
        h_shape_2 = tuple((header_2[f'NAXIS{k}'] for k in range(header_1['NAXIS'], 0, -1)))
        if h_shape_1 != h_shape_2 :
            raise ValueError(f"header_1 and header_2 must have the same axis dimensions ({h_shape_1} and {h_shape_2})")
        # By default, the new header is a copy of header_1
        new_header = header_1.copy()
    elif (n1 == 2 and n2 == 1) or (n1 == 1 and n2 == 2) :
        # Case 2 : one has two spatial axis and the other has one spectral axis
        (map_header, profile_header) = (header_1.copy(), header_2.copy())\
            if n1 > n2 else (header_2.copy(), header_1.copy())
        map_header['NAXIS'] = 3
        for card in cards :
            map_header[card.format(3)] = profile_header[card.format(1)]
        new_header = map_header
    elif (n1 == 3 and n2 == 2) or (n1 == 2 and n2 == 3) :
        # Case 3 : one has three axis and the other has two spatial axis
        # By default, the new header is a copy of the cube header
        new_header = header_1.copy() if n1 > n2 else header_2.copy()
    elif (n1 == 3 and n2 == 1) or (n1 == 1 and n2 == 3) :
        # Case 4 : one has three axis and the other has one spectral axis
        # By default, the new header is a copy of the profile header
        new_header = header_1.copy() if n1 > n2 else header_2.copy()
    else :
        # Else return an error
        raise ValueError("Headers cannot be merged.")

    # Check somes cards
    try :
        if header_1['LINE'].strip().lower() != header_2['LINE'].strip().lower() :
            new_header['LINE'] = 'COMBINATION'
    except :
        pass
    # Remove unit
    if 'BUNIT' in new_header.keys() :
        new_header['BUNIT'] = 'UNKNOWN'
    # Update origin
    if 'ORIGIN' in new_header.keys() :
        if '- MODIFIED' not in new_header['ORIGIN'].strip().upper() :
            new_header['ORIGIN'] = new_header['ORIGIN'].strip().upper() + ' - MODIFIED'

    return new_header

def create_header(struct : str, ref_header : Optional[fits.Header] = None,
    nx : Optional[int] = None, ny : Optional[int] = None, nz : Optional[int] = None) -> fits.Header:
    """
    TODO
    """
    struct = struct.lower()
    if struct not in ['cube', 'map', 'profile'] :
        raise ValueError(f"struct must be 'cube', 'map' or 'profile', not {struct}")

    if ref_header is None :
        new_header = fits.Header()
        new_header['SIMPLE'] = 'T'
        new_header['BITPIX'] = 32     
        #new_header['BSCALE'] = 0.3341172671867E-08
        #new_header['BZERO'] = 0.3305371761322E+01
        #new_header['BLANK'] = 2147483647
        new_header['DATAMIN'] = 0.
        new_header['DATAMAX'] = 0.
        new_header['BUNIT'] = 'UNKNOWN'
        new_header['OBJECT'] = 'UNKNOWN'
        new_header['RADESYS'] = 'FK5'
        #new_header['RA'] =  0.8522612499999E+02
        #new_header['DEC'] = -0.2466666666667E+01
        #new_header['EQUINOX'] =  0.2000000000000E+04
        new_header['LINE'] = 'UNKNOWN'
        #new_header['ALTRPIX'] = 0.1200000000000E+03
        #new_header['ALTRVAL'] = 0.1134869952662E+12
        #new_header['RESTFREQ'] = 0.1134909702000E+12
        #new_header['IMAGFREQ'] = 0.9400874075270E+11
        #new_header['VELO-LSR'] = 0.1050000000000E+05
        #new_header['VELREF'] = 257
        #new_header['SPECSYS'] = 'LSRK'
        #new_header['BMAJ'] =  0.6346730671426E-02
        #new_header['BMIN'] =  0.6346730671426E-02
        #new_header['BPA'] =  0.0000000000000E+00
        new_header['ORIGIN'] = 'IRAM'
        new_header['DATE'] = str(datetime.now())
        if struct == 'cube' :
            if nx is None or ny is None or nz is None :
                raise ValueError('nx, ny and nz cannot be None to create a cube header without reference')
            new_header['NAXIS'] = 3
            new_header['NAXIS1'] = nx
            new_header['NAXIS2'] = ny
            new_header['NAXIS3'] = nz
            new_header['CTYPE1'] = 'INDICES'
            new_header['CTYPE2'] = 'INDICES'
            new_header['CTYPE3'] = 'INDICES'
            new_header['CRVAL1'] = 1
            new_header['CDELT1'] = 1
            new_header['CRPIX1'] = 1
            new_header['CROTA1'] = 0
            new_header['CRVAL2'] = 1
            new_header['CDELT2'] = 1
            new_header['CRPIX2'] = 1
            new_header['CROTA2'] = 0
            new_header['CRVAL3'] = 1
            new_header['CDELT3'] = 1
            new_header['CRPIX3'] = 1
            new_header['CROTA3'] = 0
            return new_header
        if struct == 'map' :
            if nx is None or ny is None :
                raise ValueError('nx and ny cannot be None to create a map header without reference')
            new_header['NAXIS'] = 2
            new_header['NAXIS1'] = nx
            new_header['NAXIS2'] = ny
            new_header['CTYPE1'] = 'INDICES'
            new_header['CTYPE2'] = 'INDICES'
            new_header['CRVAL1'] = 1
            new_header['CDELT1'] = 1
            new_header['CRPIX1'] = 1
            new_header['CROTA1'] = 0
            new_header['CRVAL2'] = 1
            new_header['CDELT2'] = 1
            new_header['CRPIX2'] = 1
            new_header['CROTA2'] = 0
            return new_header
        if struct == 'profile' :
            if nz is None :
                raise ValueError('nz cannot be None to create a profile header without reference')
            new_header['NAXIS'] = 1
            new_header['NAXIS1'] = nz
            new_header['CTYPE1'] = 'INDICES'
            new_header['CRVAL1'] = 1
            new_header['CDELT1'] = 1
            new_header['CRPIX1'] = 1
            new_header['CROTA1'] = 0
            return new_header

    if ref_header['NAXIS'] == 1 :
        ref_struct = 'profile'
    elif ref_header['NAXIS'] == 2 :
        ref_struct = 'map'
    elif ref_header['NAXIS'] == 3 :
        ref_struct = 'cube'
    else :
        raise ValueError(f"ref_header['NAXIS'] must be equal to 1, 2 or 3, not {ref_header['NAXIS']}")

    if struct == 'cube' and ref_struct == 'cube' :
        return ref_header.copy()
    if struct == 'cube' and ref_struct == 'map' :
        return merge_headers(ref_header, create_header('profile', nz = nz))
    if struct == 'cube' and ref_struct == 'profile' :
        return merge_headers(ref_header, create_header('map', nx = nx, ny = ny))
    
    raise ValueError(f'Cannot create {struct} header using a {ref_struct} header as reference')
    

# Cards

def move_header_axes(header : fits.Header, order : Literal['xyz', 'yxz', 'zxy', 'zyx', 'xy', 'yx']) -> fits.Header :
    """
    Return a version of header where the order of axes if xyz.
    Input header axes order is `order`.
    """
    order = order.lower()

    # If header is a cube header
    if header['NAXIS'] == 3 :
        if order == 'xyz' :
            indices = (1, 2, 3)
        elif order == 'yxz' :
            indices = (2, 1, 3)
        elif order == 'zxy' :
            indices = (3, 1, 2)
        elif order == 'zyx' :
            indices = (3, 2, 1)
        else :
            raise ValueError(f"order must be 'xyz', 'yxz', 'zxy' or 'zyx', not '{order}'")
    elif header['NAXIS'] == 2 :
        if order == 'xy' :
            indices = (1, 2)
        elif order == 'yx' :
            indices = (2, 1)
        else :
            raise ValueError(f"order must be 'xy' or 'yx', not '{order}'")
    else :
        raise ValueError('move_header_axes is not defined for a number of axis different than 2 or 3')
    
    new_header = header.copy()
    cards = ['NAXIS{}', 'CTYPE{}', 'CRVAL{}', 'CDELT{}', 'CRPIX{}', 'CROTA{}']
    for i, j in enumerate(indices, 1) :
        for card in cards :
            new_header[card.format(j)] = header[card.format(i)]
    return new_header

def remove_header_axis(header : fits.Header, axis : str) -> fits.Header :
    """
    Axis must be 'spatial' or 'spectral'.
    Header must be a cube header i.e. have header['NAXIS'] = 3.
    """
    axis = axis.strip().lower()
    cards = ['NAXIS{}', 'CTYPE{}', 'CRVAL{}', 'CDELT{}', 'CRPIX{}', 'CROTA{}']
    new_header = header.copy()
    if axis == 'spatial' :
        for card in cards :
            new_header[card.format(1)] = new_header[card.format(3)]
            del new_header[card.format(2)]
            del new_header[card.format(3)]
        new_header['NAXIS'] = 1
    elif axis == 'spectral' :
        for card in cards :
            del new_header[card.format(3)]
        new_header['NAXIS'] = 2
    else :
        raise ValueError(f"axis must be 'spatial' or 'spectral', not {axis}")
    return new_header

def _replace_none(x, value) :
    """ Returns x+1 if x is not None, else value """
    return x+1 if x is not None else value

def change_coordinates(header, x = None, y = None, z = None) -> fits.Header :
    """ Coordinates must begin at 0 """
    new_header = header.copy()

    if header['NAXIS'] == 3 : # Cubes handling

        if x is None :
            x = [1, header['NAXIS1']]
        else :
            x = [_replace_none(x[0], 1), _replace_none(x[1], header['NAXIS1'])]
        if y is None :
            y = [1, header['NAXIS2']]
        else :
            y = [_replace_none(y[0], 1), _replace_none(y[1], header['NAXIS2'])]
        if z is None :
            z = [1, header['NAXIS3']]
        else :
            z = [_replace_none(z[0], 1), _replace_none(z[1], header['NAXIS3'])]

        new_header['CRPIX1'] = new_header['CRPIX1'] - x[0]
        new_header['CRPIX2'] = new_header['CRPIX2'] - y[0]
        new_header['CRPIX3'] = new_header['CRPIX3'] - z[0]

        new_header['NAXIS1'] = x[1] - x[0] + 1
        new_header['NAXIS2'] = y[1] - y[0] + 1
        new_header['NAXIS3'] = z[1] - z[0] + 1

    elif header['NAXIS'] == 2 : # Maps handling

        if x is None :
            x = [1, header['NAXIS1']]
        else :
            x = [_replace_none(x[0], 1), _replace_none(x[1], header['NAXIS1'])]
        if y is None :
            y = [1, header['NAXIS2']]
        else :
            y = [_replace_none(y[0], 1), _replace_none(y[1], header['NAXIS2'])]

        new_header['CRPIX1'] = new_header['CRPIX1'] - x[0]
        new_header['CRPIX2'] = new_header['CRPIX2'] - y[0]

        new_header['NAXIS1'] = x[1] - x[0] + 1
        new_header['NAXIS2'] = y[1] - y[0] + 1

    elif header['NAXIS'] == 1 : # Profiles handling

        if z is None :
            z = [1, header['NAXIS1']]
        else :
            z = [_replace_none(z[0], 1), _replace_none(z[1], header['NAXIS1'])]

        new_header['CRPIX1'] = new_header['CRPIX1'] - z[0]

        new_header['NAXIS1'] = z[1] - z[0] + 1

    else :

        raise ValueError(f"'NAXIS' card must be 1, 2 or 3 (not {header['NAXIS']})")
   
    return new_header

# Coordinates
# Unit for index is numpy index (beginning at 0)
# Unit for velocity is km/s
# Unit for frequency is GHz
# Unit for angle is ' (minute of arc)

C_LIGHT = 299792458 # In m/s

def indices_to_coordinates(header : fits.Header, i : Union[int, np.ndarray],
    j : Union[int, np.ndarray], absolute : bool = False) :
    """ Returns the absolute coordinates in degrees of the (i,j) point in pixels (beginning at 0) """
    if header['NAXIS'] not in [2, 3] : # If header of profile
        raise ValueError("header must be the header of a cube or a map")
    if (np.array(i) < 0).any() or (np.array(j) < 0).any() :
        raise ValueError('i and j must be non-negative indices')
    if (np.array(i) >= header['NAXIS1']).any() :
        raise ValueError(f"i must be lower than {header['NAXIS1']}")
    if (np.array(j) >= header['NAXIS2']).any() :
        raise ValueError(f"j must be lower than {header['NAXIS2']}")

    if absolute :
        x = (i+1-header['CRPIX1']) * header['CDELT1'] + header['CRVAL1']
        y = (j+1-header['CRPIX2']) * header['CDELT2'] + header['CRVAL2']
    else :
        x = (i+1-header['CRPIX1']) * header['CDELT1']
        y = (j+1-header['CRPIX2']) * header['CDELT2']
    x, y = 60*x, 60*y # Conversion from degrees to minutes of arc
    return x, y

def coordinates_to_indices(header : fits.Header, x : Union[float, np.ndarray],
    y : Union[float, np.ndarray], absolute : bool = False) :
    """ TODO """
    if header['NAXIS'] == 1 :
        raise ValueError("header must be the header of a cube or a map")
    xmin, ymin = indices_to_coordinates(header, 0, 0, absolute = absolute)
    xmax, ymax = indices_to_coordinates(header, header['NAXIS1']-1, header['NAXIS2']-1, absolute = absolute)
    if (np.array(x) < xmin).any() :
        raise ValueError(f"x must be greater than {xmin}'")
    if (np.array(y) < ymin).any() :
        raise ValueError(f"y must be greater than {ymin}'")
    if (np.array(x) > xmax).any() :
        raise ValueError(f"x must be lower than {xmax}'")
    if (np.array(y) > ymax).any() :
        raise ValueError(f"y must be lower than {ymax}'")

    if absolute :
        i = (x - header['CRVAL1']) / header['CDELT1'] + header['CRPIX1'] - 1
        j = (y - header['CRVAL2']) / header['CDELT2'] + header['CRPIX2'] - 1
    else :
        i = x / header['CDELT1'] + header['CRPIX1'] - 1
        j = y / header['CDELT2'] + header['CRPIX2'] - 1
    return round(i), round(j)

def bound_coordinates(header : fits.Header, absolute : bool = False) :
    """ Returns the absolute coordinates bounds """
    xymin = indices_to_coordinates(header, 0, 0, absolute = absolute)
    xymax = indices_to_coordinates(header, header['NAXIS1']-1, header['NAXIS2']-1, absolute = absolute)
    xbounds = xymin[0], xymax[0]
    ybounds = xymin[1], xymax[1]
    return xbounds, ybounds


def index_to_velocity(header : fits.Header, k : Union[int, np.ndarray]) :
    """ Returns the velocity of the k-th spectral image by Doppler effect (beginning at 0) """
    if header['NAXIS'] not in [1, 3] : # If header of map
        raise ValueError("header must be the header of a cube or a profile")
    if (np.array(k) < 0).any() :
        raise ValueError('k must be a non-negative index')
    
    if header['NAXIS'] == 1 : # Header of profile
        if (np.array(k) >= header['NAXIS1']).any() :
            raise ValueError(f"k must be lower than {header['NAXIS1']}")
        v = (k+1-header['CRPIX1']) * header['CDELT1'] + header['CRVAL1']
    else : # Header of cube
        if (np.array(k) >= header['NAXIS3']).any() :
            raise ValueError(f"k must be lower than {header['NAXIS3']}")
        v = (k+1-header['CRPIX3']) * header['CDELT3'] + header['CRVAL3']
    return v / 1e3 # Convert to km/s

def velocity_to_index(header : fits.Header, v : Union[float, np.ndarray]) :
    """ Returns the velocity channel index (beginning at 0) corresponding to a velocity v  """
    if header['NAXIS'] not in [1, 3] :
        raise ValueError("header must be the header of a cube or a profile")
    vmin = index_to_velocity(header, 0)
    if (np.array(v) < vmin).any() :
        raise ValueError(f'v must be greater than {vmin}')

    if header['NAXIS'] == 1 : # Header of profile
        vmax = index_to_velocity(header, header['NAXIS1']-1)
        if (np.array(v) > vmax).any() :
            raise ValueError(f'v must be lower than {vmax}')
        v_ = v * 1e3 # Convert to m/s
        k = (v_ - header['CRVAL1']) / header['CDELT1'] + header['CRPIX1'] - 1
    else : # Header of cube
        vmax = index_to_velocity(header, header['NAXIS3']-1)
        if (np.array(v) > vmax).any() :
            raise ValueError(f'v must be lower than {vmax}')
        v_ = v * 1e3 # Convert to m/s
        k = (v_ - header['CRVAL3']) / header['CDELT3'] + header['CRPIX3'] - 1
    return round(k)

def bound_velocities(header : fits.Header) :
    """ Returns the min and max velocities """
    vmin = index_to_velocity(header, 0)
    vmax = index_to_velocity(header, header[f"NAXIS{header['NAXIS']}"]-1)
    return vmin, vmax
    

def index_to_frequency(header : fits.Header, k : Union[int, np.ndarray]) :
    """ Returns the frequency of the k-th spectal image (beginning at 0) """
    if header['NAXIS'] not in [1, 3] : # If header of map
        raise ValueError("header must be the header of a cube or a profile")
    if (np.array(k) < 0).any() :
        raise ValueError('k must be a non-negative index')
    
    if header['NAXIS'] == 1 :
        if (np.array(k) >= header['NAXIS1']).any() :
            raise ValueError(f"k must be lower than {header['NAXIS1']}")
        df = -(header['CDELT1']/C_LIGHT) * header['RESTFREQ']
        f = (k+1-header['CRPIX1']) * df + header['RESTFREQ']
    else :
        if (np.array(k) >= header['NAXIS3']).any() :
            raise ValueError(f"k must be lower than {header['NAXIS3']}")
        df = -(header['CDELT3']/C_LIGHT) * header['RESTFREQ']
        f = (k+1-header['CRPIX3']) * df + header['RESTFREQ']
    return f / 1e9 # Convert to GHz

def frequency_to_index(header : fits.Header, f : Union[float, np.ndarray]) :
    """ Returns the velocity channel index (beginning at 0) corresponding to a frequency f  """
    if header['NAXIS'] not in [1, 3] :
        raise ValueError("header must be the header of a cube or a profile")
    fmin = index_to_frequency(header, 0)
    if (np.array(f) < fmin).any() :
        raise ValueError(f'f must be greater than {fmin}')

    if header['NAXIS'] == 1 :
        fmax = index_to_velocity(header, header['NAXIS1']-1)
        if (np.array(f) > fmax).any() :
            raise ValueError(f'f must be lower than {fmax}')
        f_ = f * 1e9 # Convert to Hz
        df = -(header['CDELT1']/C_LIGHT) * header['RESTFREQ']
        k = (f_ - header['RESTFREQ']) / df + header['CRPIX1'] - 1
    else :
        fmax = index_to_velocity(header, header['NAXIS3']-1)
        if (np.array(f) > fmax).any() :
            raise ValueError(f'f must be lower than {fmax}')
        f_ = f * 1e9 # Convert to Hz
        df = -(header['CDELT3']/C_LIGHT) * header['RESTFREQ']
        k = (f_ - header['RESTFREQ']) / df + header['CRPIX3'] - 1
    return round(k)

def bound_frequencies(header : fits.Header) :
    """ Returns the min and max frequencies """
    fmin = index_to_frequency(header, header[f"NAXIS{header['NAXIS']}"]-1)
    fmax = index_to_frequency(header, 0)
    return fmin, fmax