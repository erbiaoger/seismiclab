"""
F-X Radon transform operators for seismic data processing.

转换自 MATLAB: codes/linear_operators/radon_fx.m
"""

import numpy as np
from typing import Any
from scipy.fft import fft, ifft


def _nextpow2(n: int) -> int:
    """Next power of 2 greater than or equal to n."""
    return int(2 ** np.ceil(np.log2(n)))


def _bp_filter(data: np.ndarray, dt: float, f1: float, f2: float,
               f3: float, f4: float) -> np.ndarray:
    """
    Band-pass filter wrapper.

    Imports from bp_filter module when available.
    """
    try:
        from ..bp_filter import bp_filter
        return bp_filter(data, dt, f1, f2, f3, f4)
    except ImportError:
        # Fallback: return data unchanged if bp_filter not available
        return data


def radon_fx(In: np.ndarray, Par: Any, itype: int) -> np.ndarray:
    """
    F-X Radon transform operators (forward and adjoint).

    Computes linear or parabolic Radon transform in frequency domain.

    Parameters
    ----------
    In : np.ndarray
        Input data
        - If itype=1 (forward): Radon coefficients (np x nt)
        - If itype=-1 (adjoint): CMP gather (nh x nt)
    Par : object
        Parameters with attributes:
        - h: offsets (nh,)
        - p: curvatures (parabolic) or dips (linear), normalized
        - dt: sampling interval
        - f: frequency corners [f1, f2, f3, f4] for BP filter
        - transf: 'parab', 'linear', or 'hyperb'
    itype : int
        1 for forward transform, -1 for adjoint transform

    Returns
    -------
    Out : np.ndarray
        Output data
        - If itype=1: CMP gather (nh x nt)
        - If itype=-1: Radon coefficients (np x nt)

    Notes
    -----
    This function performs frequency-domain Radon transform with optional
    band-pass filtering. The forward transform maps from Radon domain to
    space domain, while the adjoint maps from space to Radon domain.

    Examples
    --------
    >>> import numpy as np
    >>> from types import SimpleNamespace
    >>> nh, nt, np = 50, 1000, 30
    >>> h = np.linspace(0, 2000, nh)
    >>> p = np.linspace(-50, 50, np) * 1e-6
    >>> dt = 0.004
    >>> f = [5, 10, 40, 60]  # frequency corners
    >>> Par = SimpleNamespace(h=h, p=p, dt=dt, f=f, transf='parab')
    >>> # Forward transform
    >>> m = np.random.randn(np, nt)
    >>> d = radon_fx(m, Par, 1)
    """
    dt = Par.dt
    h = Par.h
    p = Par.p
    f = Par.f

    f1, f2, f3, f4 = f[0], f[1], f[2], f[3]

    # Normalize offsets for parabolic transform
    if Par.transf == 'parab':
        hmax = np.max(np.abs(h))
        h = (h / hmax) ** 2

    nh = len(h)
    np_param = len(p)

    # Forward transform: Radon to space
    if itype == 1:
        nt = In.shape[0]
        m = In
        m = _bp_filter(m, dt, f1, f2, f3, f4)
    else:
        # Adjoint transform: space to Radon
        nt = In.shape[0]
        d = In

    nfft = 2 * (2 ** _nextpow2(nt))
    ilow = int(np.floor(f[0] * dt * nfft)) + 1
    ihigh = int(np.floor(f[3] * dt * nfft)) + 1

    # Phase shift matrix
    A = np.outer(h, p)

    if itype == 1:
        # Forward transform
        M = fft(m, n=nfft, axis=0)
        D = np.zeros((nfft, nh), dtype=complex)

        for ifreq in range(ilow, ihigh + 1):
            w = 2.0 * np.pi * (ifreq - 1) / nfft / dt
            L = np.exp(1j * w * A)
            x = M[ifreq, :].T
            y = L @ x
            D[ifreq, :] = y.T

        # Enforce Hermitian symmetry for real output
        for k in range(nfft // 2 + 1, nfft):
            D[k, :] = np.conj(D[nfft - k + 1, :])

        d = ifft(D, axis=0)
        d = np.real(d[:nt, :])

        Out = d

    else:
        # Adjoint transform
        D = fft(d, n=nfft, axis=0)
        M = np.zeros((nfft, np_param), dtype=complex)

        for ifreq in range(ilow, ihigh + 1):
            w = 2.0 * np.pi * (ifreq - 1) / nfft / dt
            L = np.exp(1j * w * A)
            y = D[ifreq, :].T
            x = L.conj().T @ y
            M[ifreq, :] = x.T

        # Enforce Hermitian symmetry
        for k in range(nfft // 2 + 1, nfft):
            M[k, :] = np.conj(M[nfft - k + 1, :])

        m = ifft(M, axis=0)
        m = np.real(m[:nt, :])

        Out = m

    # Apply BP filter for adjoint
    if itype == -1:
        m = _bp_filter(m, dt, f1, f2, f3, f4)
        Out = m

    return Out
