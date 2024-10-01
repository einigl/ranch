from math import ceil, pi, sqrt
from typing import Literal, Optional, Tuple, Union

import numpy as np

from .. import structures as struct

__all__ = ["kde"]


def kde(
    input: "struct.Struct",
    axis: Optional[Literal["spatial", "spectral"]] = None,
    h: Optional[Union[float, np.ndarray]] = None,
    t: Optional[np.ndarray] = None,
    t_bounds: Optional[Tuple[float]] = None,
    t_step: float = 0.05,
) -> Tuple[np.ndarray]:
    """If axis == 'spectral' then the kde is computed over the spectral axis i.e. pixel-wise.
    Use a gaussian kernel of parameter h

    Parameters
    ----------
    input : Cube | Map | Profile
        Input structure.
    axis : str, optional
        Axis over which to compute the PDF. Must be set only if input is an instance of Cube.
        If axis is None, the PDF is computed over all the flattened structure so the output
        `pdf` is a 1D-array
    h : float or ndarray, optional
        Kernel parameter. Must be a scalar if input is an instance of Map or Profile.
        If input is an instance of Cube, `h` must be an array of shape (nz,) if axis == 'spectral'
        or (ny, nx) if axis == 'spatial'.
    t : ndarray, optional
        Variable of the estimated PDF. Must be set if a specific range of value is needed.
    t_bounds : tuple of float, optional
        Bounds of `t`. Ignored if t is not None. Default : None.
    t_step : float, optional
        Step between two values of `t`. Ignored if t is not None. Default : 0.05.
        If `t_step` is not compatible with `t_step`, the right bound could be modified.

    Returns
    -------

    t : ndarray
        Variable of the estimated PDF.
    pdf : ndarray
        Estimated PDF.
    """
    if axis is not None and not isinstance(input, struct.Cube):
        raise ValueError("axis must only be set if input is an instance of Cube")
    elif axis is not None and isinstance(input, struct.Cube):
        axis = axis.lower()
        if axis == "spectral":
            data = np.moveaxis(input.data, 0, -1)
        elif axis == "spatial":
            data = input.data.reshape(input.nz, -1)
        else:
            raise ValueError(f"axis must only be 'spectral' or 'spatial', not {axis}")
    else:
        data = input.data.flatten()

    if h is None:
        h = 1.06 * np.nanstd(data, axis=-1) * data.shape[-1] ** (-1 / 5)
    else:
        h = float(h)

    if t_bounds is None:  # 5 sigma bound
        t_bounds = (data.min() - 5 * h.min(), data.max() + 5 * h.max())
    if t is None:
        n_t = ceil((t_bounds[1] - t_bounds[0]) / t_step)
        t = np.arange(n_t) * t_step + t_bounds[0]

    t_ = np.expand_dims(t, axis=tuple(range(data.ndim)))
    data_ = np.expand_dims(data, axis=-1)
    h_ = h if isinstance(h, float) else np.expand_dims(h, axis=(-1, -2))

    # pdf = norm.pdf(t_, loc = data_, scale = h_).mean(axis = -2)
    pdf = 1 / (sqrt(2 * pi) * h_) * np.exp(-((t_ - data_) ** 2) / (2 * h_**2))
    pdf = np.nanmean(pdf, axis=-2)

    return t, pdf
