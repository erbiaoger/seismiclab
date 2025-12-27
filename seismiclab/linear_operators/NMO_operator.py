"""
NMO (Normal Moveout) operator for seismic data processing.

转换自 MATLAB: codes/linear_operators/NMO_operator.m
"""

import numpy as np
from typing import Any


def NMO_operator(in_data: np.ndarray, Param: Any, flag: int) -> np.ndarray:
    """
    Perform forward and adjoint NMO operator.

    Parameters
    ----------
    in_data : np.ndarray
        Input data (nt x nx)
        - If flag==1: Data after NMO
        - If flag==-1: Data before NMO
    Param : object
        Parameters with attributes:
        - h: offset array (nx,)
        - dt: sampling interval
        - vnmo: NMO velocity array (nt,)
    flag : int
        1 for inverse NMO (NMO-corrected to original), -1 for forward NMO

    Returns
    -------
    out : np.ndarray
        Output data (nt x nx)
        - If flag==1: Data before NMO
        - If flag==-1: Data after NMO

    Notes
    -----
    The NMO operator applies the Normal Moveout correction used in seismic
    processing. The adjoint operation maps from NMO-corrected time to
    original time using linear interpolation.

    Examples
    --------
    >>> import numpy as np
    >>> from types import SimpleNamespace
    >>> nt, nx = 1000, 50
    >>> dt = 0.004
    >>> h = np.arange(nx) * 100  # offsets
    >>> vnmo = 2000 * np.ones(nt)  # constant velocity
    >>> Param = SimpleNamespace(h=h, dt=dt, vnmo=vnmo)
    >>> data_nmo = np.random.randn(nt, nx)
    >>> data_original = NMO_operator(data_nmo, Param, 1)
    """
    h = Param.h
    dt = Param.dt
    vnmo = Param.vnmo

    v2 = vnmo ** 2
    h2 = h ** 2
    nt, nx = in_data.shape

    if flag == 1:
        # Inverse NMO: NMO-corrected to original time
        m = in_data
        d = np.zeros((nt, nx))

        for k in range(nx):
            for it0 in range(nt):
                t0 = it0 * dt
                t = np.sqrt(t0 ** 2 + h2[k] / v2[it0])
                its = t / dt
                it1 = int(np.floor(its))
                it2 = it1 + 1
                a = its - it1

                if it2 < nt:
                    d[it1, k] += (1 - a) * m[it0, k]
                    d[it2, k] += a * m[it0, k]

        out = d

    else:
        # Forward NMO: original to NMO-corrected time
        d = in_data
        ma = np.zeros((nt, nx))

        for k in range(nx):
            for it0 in range(nt):
                t0 = it0 * dt
                t = np.sqrt(t0 ** 2 + h2[k] / v2[it0])
                its = t / dt
                it1 = int(np.floor(its))
                it2 = it1 + 1
                a = its - it1

                if it2 < nt:
                    ma[it0, k] = (1.0 - a) * d[it1, k] + a * d[it2, k]

        out = ma

    return out
