"""
Apex-shifted hyperbolic Radon transform operators.

转换自 MATLAB: codes/linear_operators/ash_radon_tx.m
"""

from __future__ import annotations

import numpy as np
from typing import Union
from ..bp_filter import bp_filter


def ash_radon_tx(In: np.ndarray, Par: dict, itype: int) -> np.ndarray:
    """
    Operators for t-x, tau-v-a forward and adjoint Radon transforms
    to compute the apex shifted hyperbolic Radon transform.

    转换自 MATLAB: codes/linear_operators/ash_radon_tx.m

    Parameters
    ----------
    In : np.ndarray
        Radon coefficients (nt, nv, na) if itype = 1 (Forward transform)
        CSG or CRG gather (nt, nh) if itype = -1 (Adjoint transform)
    Par : dict
        Parameters dictionary containing:
        - h: vector containing the nh offsets in meters
        - v: vector containing the nv velocities in m/s
        - a: vector containing the na apexes in meters
        - dt: sampling interval
        - f: frequency corners of BP operator [f1, f2, f3, f4]
    itype : int
        1 for forward transform, -1 for adjoint transform

    Returns
    -------
    Out : np.ndarray
        CSG or CRG gather (nt, nh) if itype = 1 (Forward)
        Radon coefficients (nt, nv, na) if itype = -1 (Adjoint)

    Notes
    -----
    This function calls bp_filter. It is like doing the deconvoluted RT
    with a band-pass zero phase wavelet.

    Reference: M.D. Sacchi, msacchi@ualberta.ca

    Examples
    --------
    >>> import numpy as np
    >>> Par = {'h': np.arange(0, 1000, 20), 'v': np.array([2000, 2500, 3000]),
    ...        'a': np.array([0, 100, 200]), 'dt': 0.004, 'f': [5, 10, 40, 50]}
    >>> m = np.random.randn(500, 3, 3)
    >>> d = ash_radon_tx(m, Par, 1)  # Forward transform
    """
    h = Par['h']
    v = Par['v']
    a = Par['a']
    dt = Par['dt']
    f = Par['f']

    f1, f2, f3, f4 = f[0], f[1], f[2], f[3]

    nv = len(v)
    na = len(a)
    nh = len(h)

    if itype == 1:
        m = In.copy()
        nt, nv_in, na_in = m.shape
        d = np.zeros((nt, nh), dtype=complex)
        m = bp_filter(m, dt, f1, f2, f3, f4)
    else:  # itype == -1
        d = In.copy()
        nt, nh = d.shape
        m = np.zeros((nt, nv, na), dtype=complex)

    # Time function: sqrt(tau^2 - (h-a)^2/v^2)
    def time_func(x1, x2, x3, x4):
        val = x1**2 - (x2 - x3)**2 / x4**2
        # Only compute sqrt for positive values
        result = np.zeros_like(val)
        mask = val >= 0
        result[mask] = np.sqrt(val[mask])
        return result

    for it in range(nt):
        for iv in range(nv):
            for ia in range(na):
                for ih in range(nh):
                    tau = time_func((it) * dt, h[ih], a[ia], v[iv])

                    if tau >= 0:
                        itau = tau / dt
                        itau1 = int(np.floor(itau))
                        itau2 = itau1 + 1
                        alpha = itau - itau1

                        if itau2 < nt and itau1 >= 0:
                            if itype == 1:
                                d[it, ih] += (1 - alpha) * m[itau1, iv, ia] + alpha * m[itau2, iv, ia]
                            else:  # itype == -1
                                m[itau1, iv, ia] += (1 - alpha) * d[it, ih]
                                m[itau2, iv, ia] += alpha * d[it, ih]

    if itype == 1:
        Out = d
    else:  # itype == -1
        Out = bp_filter(m, dt, f1, f2, f3, f4)

    return np.real(Out)
