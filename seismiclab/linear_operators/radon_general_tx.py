"""
General Radon transform operators for linear, parabolic, or hyperbolic moveout.

转换自 MATLAB: codes/linear_operators/radon_tx.m
"""

from __future__ import annotations

import numpy as np
from typing import Union
from ..bp_filter import bp_filter


def radon_general_tx(In: np.ndarray, Par: dict, itype: int) -> np.ndarray:
    """
    Operators for t-x, tau-v forward and adjoint Radon transforms
    to compute linear, parabolic or hyperbolic Radon transform.

    转换自 MATLAB: codes/linear_operators/radon_tx.m

    Parameters
    ----------
    In : np.ndarray
        Radon coefficients (nt, np) if itype = 1 (Forward transform)
        CMP gather (nt, nh) if itype = -1 (Adjoint transform)
    Par : dict
        Parameters dictionary containing:
        - h: vector containing the nh offsets
        - p: vector containing the np parameters:
            * velocities in m/s if transf='hyperb'
            * curvatures normalized by far offset if transf='parab'
            * dips in s/m if transf='linear'
        - dt: sampling interval
        - f: frequency corners of BP operator [f1, f2, f3, f4]
        - transf: 'parab', 'linear', or 'hyperb'
    itype : int
        1 for forward transform, -1 for adjoint transform

    Returns
    -------
    Out : np.ndarray
        CMP gather (nt, nh) if itype = 1 (Forward)
        Radon coefficients (nt, np) if itype = -1 (Adjoint)

    Notes
    -----
    This function calls bp_filter. It is like doing the deconvoluted RT
    with a band-pass zero phase wavelet.

    Reference: M.D. Sacchi, msacchi@ualberta.ca

    Examples
    --------
    >>> import numpy as np
    >>> Par = {'h': np.arange(0, 1000, 20), 'p': np.array([0.1, 0.2, 0.3]),
    ...        'dt': 0.004, 'f': [5, 10, 40, 50], 'transf': 'parab'}
    >>> m = np.random.randn(500, 3)
    >>> d = radon_general_tx(m, Par, 1)  # Forward transform
    """
    h = Par['h']
    p = Par['p']
    dt = Par['dt']
    f = Par['f']
    transf = Par.get('transf', 'parab')

    f1, f2, f3, f4 = f[0], f[1], f[2], f[3]

    np_len = len(p)
    nh = len(h)

    if itype == 1:
        m = In.copy()
        nt, np_in = m.shape
        d = np.zeros((nt, nh), dtype=complex)
        m = bp_filter(m, dt, f1, f2, f3, f4)
    else:  # itype == -1
        d = In.copy()
        nt, nh = d.shape
        m = np.zeros((nt, np_len), dtype=complex)

    # Define time function based on transform type
    if transf == 'parab':
        hmax = np.max(np.abs(h))
        def time_func(a, b, c):
            return a + (b / hmax)**2 * c
    elif transf == 'linear':
        def time_func(a, b, c):
            return a + b * c
    elif transf == 'hyperb':
        def time_func(a, b, c):
            return np.sqrt(a**2 + (b**2 / c**2))
    else:
        raise ValueError(f"Unknown transform type: {transf}. Use 'parab', 'linear', or 'hyperb'")

    tmax = (nt - 1) * dt
    tau = np.arange(nt) * dt

    if itype == 1:
        # Forward transform
        for itau in range(nt):
            for ih in range(nh):
                for ip in range(np_len):
                    t = time_func(tau[itau], h[ih], p[ip])
                    if t > 0 and t < tmax:
                        it = t / dt
                        it1 = int(np.floor(it))
                        it2 = it1 + 1
                        alpha = it - it1
                        if it2 < nt:
                            d[it1, ih] += (1 - alpha) * m[itau, ip]
                            d[it2, ih] += alpha * m[itau, ip]
    else:
        # Adjoint transform
        for itau in range(nt):
            for ih in range(nh):
                for ip in range(np_len):
                    t = time_func(tau[itau], h[ih], p[ip])
                    if t > 0 and t < tmax:
                        it = t / dt
                        it1 = int(np.floor(it))
                        it2 = it1 + 1
                        alpha = it - it1
                        if it2 < nt:
                            m[itau, ip] += (1 - alpha) * d[it1, ih] + alpha * d[it2, ih]

    if itype == 1:
        Out = d
    else:  # itype == -1
        Out = bp_filter(m, dt, f1, f2, f3, f4)

    return np.real(Out)
